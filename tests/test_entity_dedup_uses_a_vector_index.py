"""Entity dedup must go through the index, and must degrade rather than break.

graphiti-core builds only range + fulltext indexes for FalkorDB, so its node
dedup scans every Entity in the graph with an inline 1024-dim cosine -- once per
extracted entity, 25-50 times per episode. That is what pushed ingestion past a
120s socket timeout while FalkorDB itself sat idle at 53MB.

Verified against 0.29.3 (latest): same query, still no db.idx.vector call.
Upgrading does not fix it, so we route around it the same way we already did for
RELATES_TO.fact_embedding.
"""

import asyncio

import pytest

from app.services import indexed_falkor
from app.services.indexed_falkor import IndexedFalkorSearchOperations, ensure_node_vector_index


class _Executor:
    def __init__(self, fail: bool = False):
        self.queries: list[str] = []
        self.params: list[dict] = []
        self.fail = fail

    async def execute_query(self, cypher, **params):
        self.queries.append(cypher)
        self.params.append(params)
        # Fail ONLY the index procedure. A stub that failed the fallback too
        # would prove the fallback never runs, which is the opposite of the point.
        if self.fail and "db.idx.vector.queryNodes" in cypher:
            raise RuntimeError("Unknown procedure 'db.idx.vector.queryNodes'")
        return [], None, None


class _Graph:
    def __init__(self, raises: bool = False):
        self.queries: list[str] = []
        self.raises = raises

    def query(self, q):
        self.queries.append(q)
        if self.raises:
            raise RuntimeError("index already exists")


def _search(ops, executor, group_ids=None, search_filter=None):
    from graphiti_core.search.search_filters import SearchFilters

    indexed_falkor._node_vindex_ensured.clear()

    return asyncio.run(
        ops.node_similarity_search(
            executor, [0.1, 0.2], search_filter or SearchFilters(), group_ids, 10, 0.6
        )
    )


def test_it_queries_the_index_instead_of_scanning_every_entity():
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    vq = [q for q in ex.queries if "db.idx.vector.queryNodes" in q]
    assert vq, "dedup must go through the index"
    assert "cosineDistance" not in vq[0], "an inline cosine means it is still scanning"


def test_it_still_scopes_to_the_group():
    """Without this the dedup would look across every client's graph."""
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    vq = [q for q in ex.queries if "db.idx.vector.queryNodes" in q]
    assert "group_id IN $group_ids" in vq[0]
    assert any(p.get("group_ids") == ["client_pokagon"] for p in ex.params)


def test_it_falls_back_to_the_scan_when_the_index_is_missing():
    """Slower is acceptable. Crashing an ingest over a missing index is not."""
    ex = _Executor(fail=True)
    out = _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    assert out == []
    assert any("cosineDistance" in q for q in ex.queries), "it must retry via graphiti's own scan"


def test_a_filtered_search_is_left_to_graphiti():
    """SearchFilters compose WHERE clauses the index procedure cannot take.
    Dropping them silently would WIDEN the search, not speed it up."""
    from graphiti_core.search.search_filters import SearchFilters

    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["g"], SearchFilters(node_labels=["Person"]))
    assert not any("db.idx.vector.queryNodes" in q for q in ex.queries)
    assert any("cosineDistance" in q for q in ex.queries)


def test_the_index_is_created_once_per_graph():
    g = _Graph()
    ensure_node_vector_index(g, "graph_created_once", 1024)
    ensure_node_vector_index(g, "graph_created_once", 1024)
    assert len(g.queries) == 1, "cached per process; one round trip per graph"
    assert "CREATE VECTOR INDEX" in g.queries[0]
    assert "n.name_embedding" in g.queries[0]
    assert "dimension:1024" in g.queries[0]


def test_an_index_that_already_exists_is_not_an_error():
    """CREATE raises when one exists, and on builds without vector support."""
    g = _Graph(raises=True)
    ensure_node_vector_index(g, "graph_already_indexed", 1024)  # must not raise


def test_an_existing_graph_gets_its_index_on_the_next_dedup():
    """pokagon predates this change. It must not need a hand-run init-graph."""
    indexed_falkor._node_vindex_ensured.clear()
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    assert any("CREATE VECTOR INDEX" in q for q in ex.queries)


def test_the_index_is_ensured_once_not_once_per_dedup():
    """25-50 dedups per episode must not mean 25-50 CREATE round trips."""
    indexed_falkor._node_vindex_ensured.clear()
    ex = _Executor()
    from graphiti_core.search.search_filters import SearchFilters

    for _ in range(3):
        asyncio.run(
            IndexedFalkorSearchOperations().node_similarity_search(
                ex, [0.1], SearchFilters(), ["client_pokagon"], 10, 0.6
            )
        )
    assert len([q for q in ex.queries if "CREATE VECTOR INDEX" in q]) == 1
