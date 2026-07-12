"""Provenance-aware KG context retrieval for backend authority resolution."""

import logging
import time
import uuid as uuidlib

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.search import (
    ChainStatus,
    FactResult,
    FactSource,
    LegacyFactResult,
    LegacySearchContextResponse,
    ProvenanceShadow,
    ProvenanceSummary,
    SearchContextRequest,
    SearchContextResponse,
    ShadowSearchContextResponse,
)
from app.provenance_contract import (
    LEGACY_EPISODE_CONTRACT_VERSION,
    LEGACY_STRUCTURED_CONTRACT_VERSION,
    PROVENANCE_WRITE_STATE_COMPLETE,
    PROVENANCE_WRITE_STATE_STAGING,
    STRUCTURED_PROVENANCE_CONTRACT_VERSION,
    V2_ANCHOR_MODES,
    V2_PRODUCER_CONTRACT_VERSIONS,
)
from app.services import graphiti_client

logger = logging.getLogger("graphiti_service")

router = APIRouter()

_OVERFETCH_FACTOR = 3
_MAX_OVERFETCH = 150
_LEGACY_RESPONSE_CAP = 15
_MAX_SOURCES_PER_FACT = 64
_MAX_SOURCES_PER_RESPONSE = 500


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
    if (
        source.producer_contract_version == STRUCTURED_PROVENANCE_CONTRACT_VERSION
        and source.provenance_write_state != PROVENANCE_WRITE_STATE_COMPLETE
    ):
        return False
    return bool(
        source.producer_contract_version in V2_PRODUCER_CONTRACT_VERSIONS
        and source.anchor_mode in V2_ANCHOR_MODES
    )


def _claims_v2_anchor(source: graphiti_client.ResolvedEpisodeAnchor) -> bool:
    if source.malformed:
        return True
    if source.producer_contract_version == LEGACY_STRUCTURED_CONTRACT_VERSION:
        return False
    if (
        source.producer_contract_version == STRUCTURED_PROVENANCE_CONTRACT_VERSION
        and source.provenance_write_state == PROVENANCE_WRITE_STATE_STAGING
    ):
        # A known in-progress staged write is structurally pre-chain, not a
        # malformed producer claim. It remains suppressed until finalization.
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


def _legacy_facts(retrieval_edges: object) -> list[LegacyFactResult]:
    """Serialize the exact pre-P1 fact shape, order, and hard cap."""

    if not isinstance(retrieval_edges, list):
        raise TypeError("legacy search producer returned an unsupported shape")
    facts = [
        LegacyFactResult(
            subject=getattr(edge, "source_node_uuid", ""),
            predicate=getattr(edge, "name", ""),
            object=getattr(edge, "target_node_uuid", ""),
            fact=getattr(edge, "fact", ""),
            valid_from=getattr(edge, "valid_at", None),
            valid_to=getattr(edge, "invalid_at", None),
            expired_at=getattr(edge, "expired_at", None),
        )
        for edge in retrieval_edges
    ]
    return facts[:_LEGACY_RESPONSE_CAP]


async def _evaluate_provenance(
    *,
    req: SearchContextRequest,
    retrieval_edges: object,
    retrieval_path: graphiti_client.RetrievalPath,
    overfetch_limit: int,
) -> tuple[list[FactResult], ProvenanceSummary]:
    """Compute the Graphiti-owned v3 facts and exact terminal algebra."""

    if isinstance(retrieval_edges, list):
        oversized_response_events = int(len(retrieval_edges) > overfetch_limit)
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
    source_anchors_forwarded = 0

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
        if (
            len(same_engagement_sources) > _MAX_SOURCES_PER_FACT
            or source_anchors_forwarded + len(same_engagement_sources)
            > _MAX_SOURCES_PER_RESPONSE
        ):
            # Never truncate authority. The whole fact is structurally malformed
            # when its anchor set cannot fit the frozen bounded wire contract.
            malformed_item_suppressed += 1
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
        source_anchors_forwarded += len(same_engagement_sources)

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
        # Starvation is filtering loss, not a naturally short result set.
        starved_at_service=(service_forwarded < req.max_results and suppressed > 0),
    )
    if summary.candidates != (
        summary.service_forwarded
        + summary.malformed_item_suppressed
        + summary.expired_suppressed
        + summary.pre_chain_suppressed
        + summary.cross_engagement_suppressed
    ):
        raise RuntimeError("graph provenance accounting invariant failed")
    if sum(len(fact.sources) for fact in facts) > _MAX_SOURCES_PER_RESPONSE:
        raise RuntimeError("graph provenance source-anchor bound failed")
    return facts, summary


def _failed_shadow_summary(
    *,
    req: SearchContextRequest,
    retrieval_path: graphiti_client.RetrievalPath,
    overfetch_limit: int,
) -> ProvenanceSummary:
    return ProvenanceSummary(
        candidates=0,
        service_forwarded=0,
        malformed_response_events=1,
        retrieval_path=retrieval_path,
        requested_results=req.max_results,
        overfetch_limit=overfetch_limit,
        starved_at_service=False,
    )


@router.post(
    "/context",
    response_model=(
        SearchContextResponse
        | ShadowSearchContextResponse
        | LegacySearchContextResponse
    ),
)
async def search_context(req: SearchContextRequest):
    """Serve compatibility, non-enforcing shadow, or enforced v3 provenance."""

    if req.include_segment:
        raise HTTPException(
            status_code=409,
            detail="Segment context requires a governed pattern contract",
        )

    started = time.time()
    try:
        graph_name = graphiti_client._graph_name_for_client(req.client_slug)
        mode = settings.graphiti_provenance_mode
        if mode in {"legacy", "shadow"}:
            legacy_edges, legacy_path = await graphiti_client.search_with_path(
                client_slug=req.client_slug,
                query=req.query,
                max_results=req.max_results,
            )
            compatibility_facts = _legacy_facts(legacy_edges)
            if mode == "legacy":
                elapsed_ms = (time.time() - started) * 1000
                logger.info(
                    "[graphiti] compatibility search mode=legacy path=%s facts=%s",
                    legacy_path,
                    len(compatibility_facts),
                )
                return LegacySearchContextResponse(
                    facts=compatibility_facts,
                    segment_insights=[],
                    graph_name=graph_name,
                    search_time_ms=elapsed_ms,
                )

            overfetch_limit = min(
                max(req.max_results * _OVERFETCH_FACTOR, req.max_results),
                _MAX_OVERFETCH,
            )
            try:
                preview_edges, preview_path = await graphiti_client.search_with_path(
                    client_slug=req.client_slug,
                    query=req.query,
                    max_results=overfetch_limit,
                )
                preview_facts, preview_summary = await _evaluate_provenance(
                    req=req,
                    retrieval_edges=preview_edges,
                    retrieval_path=preview_path,
                    overfetch_limit=overfetch_limit,
                )
            except Exception as error:
                logger.warning(
                    "[graphiti] provenance shadow failed error_type=%s",
                    type(error).__name__,
                )
                preview_facts = []
                preview_summary = _failed_shadow_summary(
                    req=req,
                    retrieval_path=legacy_path,
                    overfetch_limit=overfetch_limit,
                )
            elapsed_ms = (time.time() - started) * 1000
            logger.info(
                "[graphiti] provenance shadow path=%s forwarded=%s candidates=%s "
                "suppressed=%s",
                preview_summary.retrieval_path,
                preview_summary.service_forwarded,
                preview_summary.candidates,
                preview_summary.candidates - preview_summary.service_forwarded,
            )
            return ShadowSearchContextResponse(
                facts=compatibility_facts,
                segment_insights=[],
                graph_name=graph_name,
                search_time_ms=elapsed_ms,
                provenance_shadow=ProvenanceShadow(
                    facts=preview_facts,
                    provenance_summary=preview_summary,
                ),
            )

        if mode != "enforce":  # Settings rejects this; retain a runtime fence.
            raise RuntimeError("unsupported provenance mode")
        overfetch_limit = min(
            max(req.max_results * _OVERFETCH_FACTOR, req.max_results),
            _MAX_OVERFETCH,
        )
        retrieval_edges, retrieval_path = await graphiti_client.search_with_path(
            client_slug=req.client_slug,
            query=req.query,
            max_results=overfetch_limit,
        )
        facts, summary = await _evaluate_provenance(
            req=req,
            retrieval_edges=retrieval_edges,
            retrieval_path=retrieval_path,
            overfetch_limit=overfetch_limit,
        )
        elapsed_ms = (time.time() - started) * 1000
        logger.info(
            "[graphiti] enforced provenance path=%s forwarded=%s candidates=%s "
            "suppressed=%s",
            summary.retrieval_path,
            summary.service_forwarded,
            summary.candidates,
            summary.candidates - summary.service_forwarded,
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
