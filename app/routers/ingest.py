"""Episode ingestion endpoints — called by the pg-boss worker in Engage."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.models.episode import (
    BootstrapRequest,
    IngestEpisodeRequest,
    IngestEpisodeResponse,
)
from app.provenance_contract import LEGACY_EPISODE_CONTRACT_VERSION
from app.services import graphiti_client, ingest_jobs

logger = logging.getLogger("graphiti_service")

router = APIRouter()


async def _perform_ingest(req: IngestEpisodeRequest) -> dict:
    """Run the extraction and return the response payload as a plain dict.

    Shared by the synchronous and asynchronous routes so the two cannot drift.
    Raises on failure; each route decides how to report that.
    """
    anchor_mode = (
        req.anchor_mode.value
        if req.anchor_mode is not None
        else LEGACY_EPISODE_CONTRACT_VERSION
    )
    producer_contract_version = (
        req.producer_contract_version
        if req.producer_contract_version is not None
        else LEGACY_EPISODE_CONTRACT_VERSION
    )

    # Build descriptive episode name
    episode_name = f"{req.episode_type.value}: {req.source_type}/{req.source_id}"
    source_desc = (
        f"Engagement {req.engagement_id} — "
        f"{req.episode_type.value} from {req.source_type}"
    )

    # Add metadata context to content for richer extraction
    enriched_content = req.content
    if req.metadata:
        meta_lines = [f"  {k}: {v}" for k, v in req.metadata.items()]
        enriched_content = f"Metadata:\n{''.join(meta_lines)}\n\n{req.content}"

    result = await graphiti_client.add_episode(
        client_slug=req.client_slug,
        engagement_id=req.engagement_id,
        name=episode_name,
        content=enriched_content,
        source_description=source_desc,
        reference_time=req.timestamp,
        metadata=req.metadata,
        source_id=req.source_id,
        source_type=req.source_type,
        episode_type=req.episode_type.value,
        anchor_mode=anchor_mode,
        producer_contract_version=producer_contract_version,
    )

    graph_name = graphiti_client._graph_name_for_client(req.client_slug)

    logger.info(
        f"[graphiti] Ingested episode for {req.client_slug}: "
        f"{req.episode_type.value} ({result.get('elapsed_ms', 0):.0f}ms)"
    )

    return {
        "episode_id": result.get("episode_id", ""),
        "entities_extracted": result.get("entities_extracted", 0),
        "facts_created": result.get("facts_created", 0),
        "graph_name": graph_name,
    }


@router.post("/episode", response_model=IngestEpisodeResponse)
async def ingest_episode(req: IngestEpisodeRequest):
    """Ingest an episode synchronously.

    RETAINED, BUT PREFER /episode/async. Extraction is 25-50 sequential model
    calls and regularly outlives the edge proxy's request ceiling; on
    2026-08-27 every call of this shape died at ~301s with a 520 while this
    service was still working. Callers that can poll should start a job
    instead.
    """
    try:
        return IngestEpisodeResponse(**await _perform_ingest(req))
    except Exception as error:
        logger.error(
            "[graphiti] Ingestion failed error_type=%s",
            type(error).__name__,
        )
        raise HTTPException(status_code=500, detail="Ingestion failed")


# Strong references to in-flight tasks. Without this the event loop is the only
# owner and a task can be garbage-collected mid-extraction.
_running: set[asyncio.Task] = set()


@router.post("/episode/async", status_code=202)
async def ingest_episode_async(req: IngestEpisodeRequest):
    """Start an ingestion and return a handle immediately.

    The response is 202 with a job_id; poll GET /ingest/jobs/{job_id} for the
    outcome. This exists because the work reliably outlives the proxy's request
    ceiling — see app/services/ingest_jobs for the full account.
    """
    job = ingest_jobs.create(req.client_slug)
    content = req.content

    async def _run() -> None:
        try:
            ingest_jobs.mark_succeeded(job.job_id, await _perform_ingest(req))
        except asyncio.CancelledError as error:
            # CancelledError is a BaseException (3.8+), so `except Exception`
            # did NOT catch it. A cancelled task therefore left its job pinned
            # at "running" forever: the poller kept asking, got "running" every
            # time, and spun until its whole budget was spent — for work that
            # had already stopped. Record it, then re-raise so cancellation
            # still behaves like cancellation.
            logger.error("[graphiti] Async ingestion cancelled job=%s", job.job_id)
            ingest_jobs.mark_failed(job.job_id, error, content)
            raise
        except Exception as error:  # noqa: BLE001 - recorded on the job
            logger.error(
                "[graphiti] Async ingestion failed job=%s error_type=%s",
                job.job_id,
                type(error).__name__,
            )
            ingest_jobs.mark_failed(job.job_id, error, content)

    task = asyncio.create_task(_run())
    _running.add(task)
    task.add_done_callback(_running.discard)

    return {"job_id": job.job_id, "status": job.status}


class IngestJobStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(..., description="Handle returned by POST /ingest/episode/async")
    # Required, and not merely for auth: it keeps the poll tenant-scoped. A body
    # without client_slug would have to claim platform scope ("*"), which is far
    # broader than reading one job. See app/auth.py verify_request.
    client_slug: str = Field(..., description="Client that started the job")


@router.post("/jobs/status")
async def ingest_job_status(req: IngestJobStatusRequest):
    """Report an ingestion job's outcome.

    POST, not GET, and deliberately so: service auth signs the request BODY, so
    a protected GET would need method-aware signing in both clients first. That
    rule is pinned by
    test_graph_stats_auth::test_every_protected_business_route_uses_the_supported_post_contract.

    A 404 means this process has no record of the job — it finished long enough
    ago to be pruned, or the service restarted. Callers must treat that as a
    RETRYABLE failure, never as success: the registry is in-process and the
    durable record lives in the caller's own queue.
    """
    job = ingest_jobs.get(req.job_id, req.client_slug)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown ingestion job")
    return job.to_dict()


@router.post("/bootstrap")
async def bootstrap_graph(req: BootstrapRequest):
    """Seed a client graph with ground-truth documents (org chart, RFP, etc.).

    Called during engagement setup to solve the cold-start problem.
    """
    try:
        results = []
        for doc in req.documents:
            result = await graphiti_client.add_episode(
                client_slug=req.client_slug,
                engagement_id=req.engagement_id,
                name=f"bootstrap: {doc.get('name', 'unknown')}",
                content=doc.get("content", ""),
                source_description=f"Bootstrap document: {doc.get('category', 'general')}",
                reference_time=doc.get("timestamp") or __import__("datetime").datetime.utcnow(),
            )
            results.append(
                {
                    "document": doc.get("name"),
                    "episode_id": result.get("episode_id", ""),
                }
            )

        graph_name = graphiti_client._graph_name_for_client(req.client_slug)

        logger.info(
            f"[graphiti] Bootstrapped {len(results)} documents for {req.client_slug}"
        )

        return {
            "graph_name": graph_name,
            "documents_processed": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"[graphiti] Bootstrap failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Bootstrap failed: {str(e)}")
