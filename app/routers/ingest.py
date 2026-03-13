"""Episode ingestion endpoint — called by pg-boss worker from Engage backend."""

import logging

from fastapi import APIRouter, HTTPException

from app.models.episode import (
    BootstrapRequest,
    IngestEpisodeRequest,
    IngestEpisodeResponse,
)
from app.services import graphiti_client

logger = logging.getLogger("graphiti_service")

router = APIRouter()


@router.post("/episode", response_model=IngestEpisodeResponse)
async def ingest_episode(req: IngestEpisodeRequest):
    """Ingest an episode into the client's knowledge graph.

    Called by the pg-boss worker in the Engage backend after AI operations complete.
    Graphiti extracts entities and relationships from the content using Claude Haiku.
    """
    try:
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
        )

        graph_name = graphiti_client._graph_name_for_client(req.client_slug)

        logger.info(
            f"[graphiti] Ingested episode for {req.client_slug}: "
            f"{req.episode_type.value} ({result.get('elapsed_ms', 0):.0f}ms)"
        )

        return IngestEpisodeResponse(
            episode_id=result.get("episode_id", ""),
            entities_extracted=result.get("entities_extracted", 0),
            facts_created=result.get("facts_created", 0),
            graph_name=graph_name,
        )

    except Exception as e:
        logger.error(f"[graphiti] Ingestion failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


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
