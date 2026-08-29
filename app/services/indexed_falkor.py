"""Give graphiti's ENTITY dedup the same HNSW treatment we already gave edges.

graphiti-core builds only range + fulltext indexes for FalkorDB. Its node dedup
therefore runs, per extracted entity:

    MATCH (n:Entity) WHERE n.group_id IN $group_ids
    WITH n, (2 - vec.cosineDistance(n.name_embedding, vecf32($v)))/2 AS score
    WHERE score > $min_score ... ORDER BY score DESC LIMIT $limit

a full scan of every Entity in the graph with a 1024-dim cosine per node, then a
sort. `LIMIT` bounds what comes BACK, not what is scanned, so no configuration
avoids it -- and it is run 25-50 times per episode, once per extracted entity.
That is what pushed ingestion past a 120s socket timeout and made a healthy,
53MB, ~1s-BGSAVE FalkorDB look like a dead one.

Verified against 0.29.3 (latest): identical query, still no `db.idx.vector`
call anywhere in the package. Upgrading does not fix this.

We already solved exactly this for RELATES_TO.fact_embedding -- see
`_ensure_edge_vector_index`, whose docstring notes the same O(N) scan taking
~28s on a 7k-edge graph. This applies that proven approach to Entity nodes.

Fallback discipline is deliberate and matches the edge path: if the index does
not exist, or this FalkorDB build lacks vector support, the override returns to
graphiti's own scan. Slower is acceptable; wrong or crashed is not.
"""

import logging
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.driver.record_parsers import entity_node_from_record
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import EntityNode

logger = logging.getLogger("graphiti_service")

_node_vindex_ensured: set[str] = set()


def ensure_node_vector_index(graph: Any, graph_name: str, dim: int) -> None:
    """Idempotently ensure an HNSW index on Entity.name_embedding.

    Creating one that exists raises, and so does an unsupported FalkorDB build;
    both are caught, because the query path falls back either way. Cached per
    process so this costs one round trip per graph, not one per episode.
    """
    if graph_name in _node_vindex_ensured:
        return
    try:
        graph.query(
            f"CREATE VECTOR INDEX FOR (n:Entity) ON (n.name_embedding) "
            f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
        )
        logger.info(
            f"[graphiti] created Entity.name_embedding vector index on {graph_name} (dim={dim})"
        )
    except Exception as e:  # noqa: BLE001 - "already exists" is the common case
        logger.debug(f"[graphiti] node vector index ensure on {graph_name}: {e}")
    _node_vindex_ensured.add(graph_name)


async def ensure_node_vector_index_via(executor: Any, group_key: str, dim: int) -> None:
    """Ensure the index using a connection that is ALREADY open.

    Deliberately not a separate select_graph on the ingest path: an earlier
    revision did that and made an unanchored ingest open a FalkorDB connection
    it had never needed. Going through the executor the search is about to use
    adds no connection and no lifecycle change -- and it means an EXISTING
    graph heals itself on its next dedup, rather than waiting for someone to
    re-run init-graph by hand.
    """
    if group_key in _node_vindex_ensured:
        return
    _node_vindex_ensured.add(group_key)  # mark first: one attempt, not one per call
    try:
        await executor.execute_query(
            f"CREATE VECTOR INDEX FOR (n:Entity) ON (n.name_embedding) "
            f"OPTIONS {{dimension:{int(dim)}, similarityFunction:'cosine'}}"
        )
        logger.info(f"[graphiti] created Entity.name_embedding vector index ({group_key})")
    except Exception as e:  # noqa: BLE001 - "already exists" is the common case
        # INFO, not DEBUG. At DEBUG this was invisible in production, so
        # "did the index get built?" could not be answered from the logs --
        # and that question was load-bearing while diagnosing the outage.
        logger.info(f"[graphiti] node vector index not created ({group_key}): {e}")


class IndexedFalkorSearchOperations(FalkorSearchOperations):
    """FalkorSearchOperations, but entity dedup goes through the vector index."""

    async def node_similarity_search(
        self,
        executor: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityNode]:
        # Only the plain group-scoped case is safe to reroute. A populated
        # SearchFilters composes extra WHERE clauses that the index procedure
        # does not accept, and silently dropping them would widen the search
        # rather than speed it up.
        if _has_filters(search_filter):
            return await super().node_similarity_search(
                executor, search_vector, search_filter, group_ids, limit, min_score
            )

        from app.config import settings  # noqa: PLC0415 - avoids an import cycle

        await ensure_node_vector_index_via(
            executor, group_ids[0] if group_ids else "*", int(settings.embedding_dim)
        )

        try:
            gid_clause = " WHERE node.group_id IN $group_ids" if group_ids else ""
            cypher = (
                f"CALL db.idx.vector.queryNodes('Entity', 'name_embedding', {int(limit)}, vecf32($search_vector)) "
                f"YIELD node, score{gid_clause} "
                f"WITH node AS n, score WHERE score > $min_score RETURN "
                + get_entity_node_return_query(GraphProvider.FALKORDB)
                + " ORDER BY score DESC"
            )
            records, _, _ = await executor.execute_query(
                cypher,
                search_vector=search_vector,
                min_score=min_score,
                **({"group_ids": group_ids} if group_ids else {}),
            )
        except Exception as e:  # noqa: BLE001 - no index / no vector support
            logger.debug(f"[graphiti] node vector search unavailable, falling back to scan: {e}")
            return await super().node_similarity_search(
                executor, search_vector, search_filter, group_ids, limit, min_score
            )

        return [entity_node_from_record(r) for r in records]


def _has_filters(search_filter: Any) -> bool:
    """True when the caller asked for more than a group scope."""
    if search_filter is None:
        return False
    try:
        data = search_filter.model_dump(exclude_none=True)
    except Exception:  # noqa: BLE001 - not a pydantic model
        return True
    return any(v for v in data.values())


class IndexedFalkorDriver(FalkorDriver):
    """FalkorDriver whose search operations use the Entity vector index."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._search_ops = IndexedFalkorSearchOperations()
