"""Search endpoint — synchronous KG context retrieval for AI services."""

import logging
import time

from fastapi import APIRouter, HTTPException

from app.models.search import (
    FactResult,
    SearchContextRequest,
    SearchContextResponse,
)
from app.services import graphiti_client

logger = logging.getLogger("graphiti_service")

router = APIRouter()


@router.post("/context", response_model=SearchContextResponse)
async def search_context(req: SearchContextRequest):
    """Search the knowledge graph for relevant context.

    Called synchronously by the Engage backend before each AI service call.
    Returns facts (edges) and segment insights.

    Graphiti search returns EntityEdge objects — each represents a
    temporal fact/relationship between two entities.
    """
    start = time.time()

    try:
        graph_name = graphiti_client._graph_name_for_client(req.client_slug)

        # Search client graph — returns list[EntityEdge]
        edges = await graphiti_client.search(
            client_slug=req.client_slug,
            query=req.query,
            max_results=req.max_results,
        )

        # Parse EntityEdge objects into structured facts
        facts: list[FactResult] = []
        for edge in edges:
            facts.append(
                FactResult(
                    subject=getattr(edge, "source_node_uuid", ""),
                    predicate=getattr(edge, "name", ""),
                    object=getattr(edge, "target_node_uuid", ""),
                    fact=getattr(edge, "fact", ""),
                    valid_from=getattr(edge, "valid_at", None),
                    valid_to=getattr(edge, "invalid_at", None),
                    expired_at=getattr(edge, "expired_at", None),
                )
            )

        # Search segment graph if requested
        segment_insights: list[str] = []
        if req.include_segment:
            try:
                segment_edges = await graphiti_client.search_segment(
                    industry="tribal_gaming",  # TODO: look up from client metadata
                    query=req.query,
                    max_results=5,
                )
                for edge in segment_edges:
                    fact_text = getattr(edge, "fact", None)
                    if fact_text:
                        segment_insights.append(fact_text)
            except Exception as e:
                logger.warning(f"[graphiti] Segment search skipped: {e}")

        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"[graphiti] Search for '{req.query[:50]}...' in {graph_name}: "
            f"{len(facts)} facts, "
            f"{len(segment_insights)} segment insights ({elapsed_ms:.0f}ms)"
        )

        return SearchContextResponse(
            facts=facts[:15],  # Top-K limit per plan
            segment_insights=segment_insights[:5],
            graph_name=graph_name,
            search_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"[graphiti] Search failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
