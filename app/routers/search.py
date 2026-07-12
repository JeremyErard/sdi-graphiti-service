"""Provenance-aware KG context retrieval for backend authority resolution."""

import logging
import time
import uuid as uuidlib

from fastapi import APIRouter, HTTPException

from app.models.search import (
    ChainStatus,
    FactResult,
    FactSource,
    ProvenanceSummary,
    SearchContextRequest,
    SearchContextResponse,
)
from app.provenance_contract import (
    LEGACY_EPISODE_CONTRACT_VERSION,
    LEGACY_STRUCTURED_CONTRACT_VERSION,
    V2_ANCHOR_MODES,
    V2_PRODUCER_CONTRACT_VERSIONS,
)
from app.services import graphiti_client

logger = logging.getLogger("graphiti_service")

router = APIRouter()

_OVERFETCH_FACTOR = 3
_MAX_OVERFETCH = 150


def _fact_uuid(value: object) -> str | None:
    try:
        return str(uuidlib.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def _anchor_is_complete(
    source: graphiti_client.ResolvedEpisodeAnchor,
) -> bool:
    if source.malformed:
        return False
    common = all(
        (
            source.episode_uuid,
            source.episode_name,
            source.source_description,
            source.source_type,
            source.source_id,
            source.engagement_id,
            source.episode_type,
            source.anchor_mode,
            source.producer_contract_version,
        )
    )
    if not common:
        return False
    if (
        source.anchor_mode == "engagement"
        and source.source_id != source.engagement_id
    ):
        return False
    if source.source_type == "engagement" and (
        source.anchor_mode != "engagement"
        or source.source_id != source.engagement_id
    ):
        return False
    if source.producer_contract_version == LEGACY_STRUCTURED_CONTRACT_VERSION:
        return False
    if source.producer_contract_version == LEGACY_EPISODE_CONTRACT_VERSION:
        return source.anchor_mode == LEGACY_EPISODE_CONTRACT_VERSION
    return bool(
        source.producer_contract_version in V2_PRODUCER_CONTRACT_VERSIONS
        and source.anchor_mode in V2_ANCHOR_MODES
    )


def _claims_v2_anchor(source: graphiti_client.ResolvedEpisodeAnchor) -> bool:
    if source.malformed:
        return True
    if source.producer_contract_version == LEGACY_STRUCTURED_CONTRACT_VERSION:
        return False
    return (
        (
            source.producer_contract_version is not None
            and source.producer_contract_version != LEGACY_EPISODE_CONTRACT_VERSION
        )
        or source.anchor_mode in V2_ANCHOR_MODES
    )


def _structurally_malformed(edge: graphiti_client.ResolvedSearchEdge) -> bool:
    return edge.malformed or not all(
        (
            edge.fact_id,
            edge.subject_uuid,
            edge.subject_name,
            edge.predicate,
            edge.object_uuid,
            edge.object_name,
            edge.fact,
        )
    )


@router.post("/context", response_model=SearchContextResponse)
async def search_context(req: SearchContextRequest):
    """Return only same-engagement, structurally chained graph facts.

    Graphiti resolves graph identities and anchors but never claims tenant-row
    existence or content grounding. Backend PR-B performs those authority checks.
    """

    if req.include_segment:
        raise HTTPException(
            status_code=409,
            detail="Segment context requires a governed pattern contract",
        )

    started = time.time()
    try:
        graph_name = graphiti_client._graph_name_for_client(req.client_slug)
        overfetch_limit = min(
            max(req.max_results * _OVERFETCH_FACTOR, req.max_results),
            _MAX_OVERFETCH,
        )
        retrieval_edges, retrieval_path = await graphiti_client.search_with_path(
            client_slug=req.client_slug,
            query=req.query,
            max_results=overfetch_limit,
        )
        if isinstance(retrieval_edges, list):
            oversized_response_events = int(
                len(retrieval_edges) > overfetch_limit
            )
            edges = retrieval_edges[:overfetch_limit]
            resolved_by_id, resolution_response_events = (
                await graphiti_client.resolve_search_provenance(
                    client_slug=req.client_slug,
                    edges=edges,
                )
            )
            malformed_response_events = min(
                oversized_response_events + resolution_response_events,
                len(edges),
            )
        else:
            # A producer-shape failure has no stable fact identity and therefore
            # stays outside the candidate equation.
            edges = []
            resolved_by_id = {}
            malformed_response_events = 1

        facts: list[FactResult] = []
        seen_fact_ids: set[str] = set()
        malformed_item_suppressed = 0
        expired_suppressed = 0
        pre_chain_suppressed = 0
        cross_engagement_suppressed = 0

        # Deterministic ordered scanned prefix: stop at K forwardable facts or
        # exhaust the bounded pool. Unexamined tail items never enter candidates.
        for raw_edge in edges:
            if len(facts) >= req.max_results:
                break
            fact_id = _fact_uuid(getattr(raw_edge, "uuid", None))
            if not fact_id or fact_id in seen_fact_ids:
                continue
            seen_fact_ids.add(fact_id)
            edge = resolved_by_id.get(fact_id)
            if edge is None or _structurally_malformed(edge):
                malformed_item_suppressed += 1
                continue
            if edge.expired_at is not None:
                expired_suppressed += 1
                continue

            complete_sources = [
                source for source in edge.sources if _anchor_is_complete(source)
            ]
            malformed_anchor_claim = any(
                _claims_v2_anchor(source) and not _anchor_is_complete(source)
                for source in edge.sources
            )
            if malformed_anchor_claim and not complete_sources:
                malformed_item_suppressed += 1
                continue
            if not complete_sources:
                pre_chain_suppressed += 1
                continue

            same_engagement_sources = [
                source
                for source in complete_sources
                if source.engagement_id == req.engagement_id
            ]
            if not same_engagement_sources:
                cross_engagement_suppressed += 1
                continue

            facts.append(
                FactResult(
                    fact_id=edge.fact_id,
                    subject=edge.subject_uuid,
                    subject_name=edge.subject_name,
                    predicate=edge.predicate,
                    object=edge.object_uuid,
                    object_name=edge.object_name,
                    fact=edge.fact,
                    episodes=[source.episode_uuid for source in same_engagement_sources],
                    sources=[
                        FactSource(
                            episode_uuid=source.episode_uuid,
                            episode_name=source.episode_name,
                            source_description=source.source_description,
                            source_type=source.source_type or "",
                            source_id=source.source_id or "",
                            engagement_id=source.engagement_id or "",
                            episode_type=source.episode_type or "",
                            anchor_mode=source.anchor_mode or "",
                            producer_contract_version=(
                                source.producer_contract_version or ""
                            ),
                            valid_at=source.valid_at,
                        )
                        for source in same_engagement_sources
                    ],
                    chain_status=ChainStatus.CHAINED.value,
                    valid_from=edge.valid_at,
                    valid_to=edge.invalid_at,
                    expired_at=edge.expired_at,
                )
            )

        suppressed = (
            malformed_item_suppressed
            + expired_suppressed
            + pre_chain_suppressed
            + cross_engagement_suppressed
        )
        service_forwarded = len(facts)
        candidates = service_forwarded + suppressed
        summary = ProvenanceSummary(
            candidates=candidates,
            service_forwarded=service_forwarded,
            malformed_item_suppressed=malformed_item_suppressed,
            expired_suppressed=expired_suppressed,
            pre_chain_suppressed=pre_chain_suppressed,
            cross_engagement_suppressed=cross_engagement_suppressed,
            malformed_response_events=malformed_response_events,
            retrieval_path=retrieval_path,
            requested_results=req.max_results,
            overfetch_limit=overfetch_limit,
            # The bounded returned pool was exhausted whenever K was not met,
            # including a naturally short or wholly malformed producer response.
            starved_at_service=service_forwarded < req.max_results,
        )
        if summary.candidates != (
            summary.service_forwarded
            + summary.malformed_item_suppressed
            + summary.expired_suppressed
            + summary.pre_chain_suppressed
            + summary.cross_engagement_suppressed
        ):
            raise RuntimeError("graph provenance accounting invariant failed")

        elapsed_ms = (time.time() - started) * 1000
        logger.info(
            "[graphiti] Search in %s path=%s: %s/%s facts forwarded "
            "(%sms)",
            graph_name,
            retrieval_path,
            service_forwarded,
            candidates,
            round(elapsed_ms),
        )
        return SearchContextResponse(
            facts=facts,
            segment_insights=[],
            graph_name=graph_name,
            search_time_ms=elapsed_ms,
            provenance_summary=summary,
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.error(
            "[graphiti] Search failed error_type=%s",
            type(error).__name__,
        )
        raise HTTPException(status_code=500, detail="Search failed")
