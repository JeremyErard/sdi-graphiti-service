"""Graphiti core wrapper — manages graph connections, episodes, and search."""

import logging
import time
from datetime import datetime
from typing import Any

import yaml
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType as GraphitiEpisodeType
from graphiti_core.driver.falkordb_driver import FalkorDriver

from app.config import settings

logger = logging.getLogger("graphiti_service")

# Cache of initialized Graphiti clients per graph name
_clients: dict[str, Graphiti] = {}


def _graph_name_for_client(client_slug: str) -> str:
    """Map client slug to isolated graph name."""
    safe_slug = "".join(c for c in client_slug if c.isalnum() or c == "_").lower()
    return f"client_{safe_slug}"


def _segment_graph_name(industry: str) -> str:
    """Map industry to segment graph name."""
    safe_industry = "".join(c for c in industry if c.isalnum() or c == "_").lower()
    return f"segment_{safe_industry}"


def _load_entity_types() -> list[dict[str, str]]:
    """Load entity types from config.yaml."""
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        return config.get("entity_types", [])
    except Exception as e:
        logger.warning(f"Failed to load entity types from config.yaml: {e}")
        return []


def _create_driver(graph_name: str) -> FalkorDriver:
    """Create a FalkorDB driver targeting a specific named graph."""
    return FalkorDriver(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
        database=graph_name,
    )


async def get_client(client_slug: str) -> Graphiti:
    """Get or create a Graphiti client for a specific client graph.

    Each client gets a separate FalkorDB named graph via the driver's
    `database` parameter, providing full data isolation.
    """
    graph_name = _graph_name_for_client(client_slug)

    if graph_name not in _clients:
        logger.info(f"[graphiti] Initializing graph: {graph_name}")
        driver = _create_driver(graph_name)
        client = Graphiti(graph_driver=driver)
        _clients[graph_name] = client

    return _clients[graph_name]


async def get_segment_client(industry: str) -> Graphiti:
    """Get or create a Graphiti client for a segment graph."""
    graph_name = _segment_graph_name(industry)

    if graph_name not in _clients:
        logger.info(f"[graphiti] Initializing segment graph: {graph_name}")
        driver = _create_driver(graph_name)
        client = Graphiti(graph_driver=driver)
        _clients[graph_name] = client

    return _clients[graph_name]


async def init_graph(client_slug: str) -> str:
    """Initialize a new graph for a client (create indices and constraints)."""
    client = await get_client(client_slug)
    await client.build_indices_and_constraints()
    graph_name = _graph_name_for_client(client_slug)
    logger.info(f"[graphiti] Graph initialized: {graph_name}")
    return graph_name


async def add_episode(
    client_slug: str,
    engagement_id: str,
    name: str,
    content: str,
    source_description: str,
    reference_time: datetime,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add an episode to the client's knowledge graph.

    Uses EpisodeType.text for all ingestion (plain text content).
    group_id is set to the graph name to prevent driver re-cloning.
    """
    client = await get_client(client_slug)
    graph_name = _graph_name_for_client(client_slug)

    start = time.time()

    result = await client.add_episode(
        name=name,
        episode_body=content,
        source_description=source_description,
        reference_time=reference_time,
        source=GraphitiEpisodeType.text,
        group_id=graph_name,
    )

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"[graphiti] Episode added to {graph_name} "
        f"in {elapsed_ms:.0f}ms: {name}"
    )

    # AddEpisodeResults has .episode, .nodes, .edges attributes
    entities_extracted = len(result.nodes) if hasattr(result, "nodes") else 0
    facts_created = len(result.edges) if hasattr(result, "edges") else 0
    episode_id = ""
    if hasattr(result, "episode") and hasattr(result.episode, "uuid"):
        episode_id = str(result.episode.uuid)

    return {
        "episode_id": episode_id,
        "entities_extracted": entities_extracted,
        "facts_created": facts_created,
        "elapsed_ms": elapsed_ms,
    }


async def search(
    client_slug: str,
    query: str,
    max_results: int = 10,
) -> list[Any]:
    """Search the client's knowledge graph.

    Returns list of EntityEdge objects. Each edge has:
    - fact: human-readable fact string
    - name: relationship label
    - source_node_uuid / target_node_uuid
    - valid_at / invalid_at / expired_at (temporal)
    - episodes: list of source episode UUIDs
    """
    client = await get_client(client_slug)
    graph_name = _graph_name_for_client(client_slug)

    start = time.time()

    edges = await client.search(
        query=query,
        num_results=max_results,
        group_ids=[graph_name],
    )

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"[graphiti] Search in {graph_name}: {len(edges)} edges ({elapsed_ms:.0f}ms)"
    )

    return edges


async def search_segment(
    industry: str,
    query: str,
    max_results: int = 5,
) -> list[Any]:
    """Search the segment knowledge graph. Returns list of EntityEdge."""
    try:
        client = await get_segment_client(industry)
        graph_name = _segment_graph_name(industry)
        results = await client.search(
            query=query,
            num_results=max_results,
            group_ids=[graph_name],
        )
        return results
    except Exception as e:
        logger.warning(f"[graphiti] Segment search failed for {industry}: {e}")
        return []


async def close_all():
    """Close all cached Graphiti clients."""
    for name, client in _clients.items():
        try:
            await client.close()
            logger.info(f"[graphiti] Closed graph client: {name}")
        except Exception as e:
            logger.warning(f"[graphiti] Error closing {name}: {e}")
    _clients.clear()
