"""Segment knowledge graph management.

Handles promotion of approved insights into segment-level graphs.
Only called after human review and approval in the SDI dashboard.
"""

import logging
from datetime import datetime

from app.services import graphiti_client

logger = logging.getLogger("graphiti_service")


async def promote_insight(
    industry: str,
    content: str,
    source_description: str,
) -> dict:
    """Promote an approved, anonymized insight into the segment graph.

    This should ONLY be called after a consultant has reviewed and approved
    the insight in the SDI dashboard.
    """
    client = await graphiti_client.get_segment_client(industry)

    result = await client.add_episode(
        name=f"segment_insight: {industry}",
        episode_body=content,
        source_description=source_description,
        reference_time=datetime.utcnow(),
        group_id=f"segment:{industry}",
    )

    graph_name = graphiti_client._segment_graph_name(industry)
    logger.info(f"[graphiti] Promoted insight to segment graph: {graph_name}")

    return {
        "graph_name": graph_name,
        "episode_id": str(result.uuid) if hasattr(result, "uuid") else str(id(result)),
    }
