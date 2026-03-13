"""Graphiti core wrapper — manages graph connections, episodes, and search."""

import logging
import time
from datetime import datetime
from typing import Any

import yaml
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from app.config import settings

logger = logging.getLogger("graphiti_service")

# Cache of initialized Graphiti clients per graph name
_clients: dict[str, Graphiti] = {}


def _build_falkordb_uri() -> str:
    """Build FalkorDB connection URI."""
    if settings.falkordb_password:
        return f"falkor://default:{settings.falkordb_password}@{settings.falkordb_host}:{settings.falkordb_port}"
    return f"falkor://{settings.falkordb_host}:{settings.falkordb_port}"


def _graph_name_for_client(client_slug: str) -> str:
    """Map client slug to isolated graph name."""
    # Sanitize slug to prevent injection
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


async def get_client(client_slug: str) -> Graphiti:
    """Get or create a Graphiti client for a specific client graph."""
    graph_name = _graph_name_for_client(client_slug)

    if graph_name not in _clients:
        uri = _build_falkordb_uri()
        logger.info(f"[graphiti] Initializing graph: {graph_name}")

        client = Graphiti(
            uri=uri,
            graph_name=graph_name,
        )

        _clients[graph_name] = client

    return _clients[graph_name]


async def get_segment_client(industry: str) -> Graphiti:
    """Get or create a Graphiti client for a segment graph."""
    graph_name = _segment_graph_name(industry)

    if graph_name not in _clients:
        uri = _build_falkordb_uri()
        logger.info(f"[graphiti] Initializing segment graph: {graph_name}")

        client = Graphiti(
            uri=uri,
            graph_name=graph_name,
        )

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
    """Add an episode to the client's knowledge graph."""
    client = await get_client(client_slug)
    group_id = f"{client_slug}:{engagement_id}"

    start = time.time()

    result = await client.add_episode(
        name=name,
        episode_body=content,
        source_description=source_description,
        reference_time=reference_time,
        group_id=group_id,
    )

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"[graphiti] Episode added to {_graph_name_for_client(client_slug)} "
        f"in {elapsed_ms:.0f}ms: {name}"
    )

    return {
        "episode_id": str(result.uuid) if hasattr(result, "uuid") else str(id(result)),
        "elapsed_ms": elapsed_ms,
    }


async def search(
    client_slug: str,
    engagement_id: str,
    query: str,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search the client's knowledge graph using hybrid search."""
    client = await get_client(client_slug)
    group_id = f"{client_slug}:{engagement_id}"

    start = time.time()

    # Retrieve nodes (entities)
    nodes = await client.search(
        query=query,
        num_results=max_results,
        group_ids=[group_id],
    )

    elapsed_ms = (time.time() - start) * 1000

    return {
        "nodes": nodes,
        "elapsed_ms": elapsed_ms,
    }


async def search_segment(
    industry: str,
    query: str,
    max_results: int = 5,
) -> list[Any]:
    """Search the segment knowledge graph."""
    try:
        client = await get_segment_client(industry)
        results = await client.search(
            query=query,
            num_results=max_results,
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
