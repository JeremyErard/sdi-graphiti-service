"""Search endpoint — synchronous KG context retrieval for AI services."""

import logging
import time

from fastapi import APIRouter, HTTPException

from app.models.search import (
    ContradictionResult,
    EntityResult,
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
    Returns entities, facts, contradictions, and segment insights.
    """
    start = time.time()

    try:
        graph_name = graphiti_client._graph_name_for_client(req.client_slug)

        # Search client graph
        result = await graphiti_client.search(
            client_slug=req.client_slug,
            engagement_id=req.engagement_id,
            query=req.query,
            max_results=req.max_results,
        )

        # Parse nodes into structured results
        nodes: list[EntityResult] = []
        facts: list[FactResult] = []
        contradictions: list[ContradictionResult] = []

        raw_nodes = result.get("nodes", [])
        for node in raw_nodes:
            if hasattr(node, "name") and hasattr(node, "summary"):
                nodes.append(
                    EntityResult(
                        name=node.name if hasattr(node, "name") else str(node),
                        entity_type=getattr(node, "label", "Unknown"),
                        properties={"summary": node.summary}
                        if hasattr(node, "summary")
                        else {},
                        relevance_score=getattr(node, "score", 0.0),
                    )
                )
            elif hasattr(node, "fact"):
                facts.append(
                    FactResult(
                        subject=getattr(node, "source_node_name", ""),
                        predicate=getattr(node, "name", ""),
                        object=getattr(node, "target_node_name", ""),
                        valid_from=getattr(node, "valid_at", None),
                        valid_to=getattr(node, "invalid_at", None),
                        confidence=getattr(node, "score", 0.0),
                        source_description=getattr(node, "fact", ""),
                    )
                )

        # Search segment graph if requested
        segment_insights: list[str] = []
        if req.include_segment:
            try:
                segment_results = await graphiti_client.search_segment(
                    industry="tribal_gaming",  # TODO: look up from client metadata
                    query=req.query,
                    max_results=5,
                )
                for item in segment_results:
                    if hasattr(item, "summary"):
                        segment_insights.append(item.summary)
                    elif hasattr(item, "fact"):
                        segment_insights.append(item.fact)
            except Exception as e:
                logger.warning(f"[graphiti] Segment search skipped: {e}")

        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"[graphiti] Search for '{req.query[:50]}...' in {graph_name}: "
            f"{len(nodes)} nodes, {len(facts)} facts, "
            f"{len(segment_insights)} segment insights ({elapsed_ms:.0f}ms)"
        )

        return SearchContextResponse(
            nodes=nodes[:req.max_results],
            facts=facts[:15],  # Top-K limit per Gemini review
            contradictions=contradictions,
            segment_insights=segment_insights[:5],
            graph_name=graph_name,
            search_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"[graphiti] Search failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
