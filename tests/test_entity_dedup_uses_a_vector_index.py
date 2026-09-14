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
import re

import pytest

from app.services import indexed_falkor
from app.services.indexed_falkor import IndexedFalkorSearchOperations, ensure_node_vector_index



from tests.conftest import assert_with_scopes_are_sound  # noqa: E402


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
    assert vq[0].lstrip().startswith("CALL db.idx.vector.queryNodes"), "the index bounds the candidates"
    assert "MATCH (n:Entity)" not in vq[0], "a bare Entity MATCH means it is still scanning"
    # The cosine is computed only for the yielded candidates, never as a scan,
    # and it reads `node.`: an alias cannot be referenced inside the WITH that
    # defines it (FalkorDB: "'n' not defined", live on 2026-09-14).
    assert "WITH node AS n, (2 - vec.cosineDistance(node.name_embedding, vecf32($search_vector)))/2 AS score" in vq[0]
    assert "cosineDistance(n.name_embedding" not in vq[0]


def test_the_threshold_uses_the_explicit_cosine_not_the_procedure_score():
    """The procedure's own score is not relied on anywhere: candidates are
    re-scored with graphiti's cosine so min_score and ordering keep their meaning."""
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    vq = [q for q in ex.queries if "db.idx.vector.queryNodes" in q][0]
    assert "YIELD node, score AS index_score" in vq, "every procedure field yielded, the score aliased away"
    assert "index_score" not in vq.split("index_score", 1)[1], "the procedure score is never used after the YIELD"
    assert "'name_embedding', 40, vecf32($search_vector)" in vq
    assert "WHERE score > $min_score" in vq
    assert vq.rstrip().endswith("ORDER BY score DESC LIMIT $limit")
    assert any(p.get("limit") == 10 and p.get("min_score") == 0.6 for p in ex.params)


def test_it_still_scopes_to_the_group():
    """Without this the dedup would look across every client's graph."""
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    vq = [q for q in ex.queries if "db.idx.vector.queryNodes" in q]
    assert "group_id IN $group_ids" in vq[0]
    assert any(p.get("group_ids") == ["client_pokagon"] for p in ex.params)


def test_no_projection_references_its_own_alias_and_nothing_after_a_with_reads_a_dropped_variable():
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    for q in ex.queries:
        assert_with_scopes_are_sound(q)


def test_the_scope_check_catches_both_live_failure_classes():
    with pytest.raises(AssertionError):
        assert_with_scopes_are_sound("CALL p() YIELD node WITH node AS n, (2 - vec.cosineDistance(n.name_embedding, vecf32($v)))/2 AS score WHERE score > $m RETURN n.uuid")
    with pytest.raises(AssertionError):
        assert_with_scopes_are_sound("CALL p() YIELD node WITH node AS n, 1 AS score WHERE score > $m RETURN node.uuid AS uuid")
    assert_with_scopes_are_sound("CALL p() YIELD node WITH node AS node, (2 - vec.cosineDistance(node.name_embedding, vecf32($v)))/2 AS score WHERE score > $m RETURN node.uuid")
    assert_with_scopes_are_sound("MATCH (n)-[e]->(m) WITH e, n, m, (2 - vec.cosineDistance(e.fact_embedding, vecf32($v)))/2 AS score WHERE score > $m RETURN e.uuid, n.uuid, m.uuid ORDER BY score DESC")
    # bare (undotted) use of a sibling alias, and an alias chain
    with pytest.raises(AssertionError):
        assert_with_scopes_are_sound("MATCH (x) WITH x AS n, size(n) AS s RETURN s")
    with pytest.raises(AssertionError):
        assert_with_scopes_are_sound("MATCH (a) WITH a AS b, b AS c RETURN c")
    # legal forms the bare-identifier rule must not reject
    assert_with_scopes_are_sound("MATCH (n) WITH n.uuid AS uuid, n AS n RETURN uuid, n.name")
    assert_with_scopes_are_sound("CALL p() YIELD node WITH node AS n, node.score AS score RETURN n.uuid, score")
    assert_with_scopes_are_sound("MATCH ()-[e]->() WITH e, startNode(e) AS s, size(e.episodes) AS size RETURN e.uuid, s.uuid, size")
    assert_with_scopes_are_sound("MATCH (n) WITH DISTINCT n RETURN n.group_id")
    # a MATCH after a WITH binds new variables the following clauses may read
    assert_with_scopes_are_sound("CALL p() YIELD relationship AS rel, score WITH rel, score ORDER BY score DESC LIMIT 200 MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity) WHERE e.group_id IN $g WITH e, score, n, m RETURN e.uuid, n.uuid, m.uuid ORDER BY score DESC LIMIT $l")


def test_a_dead_override_is_visible_once_at_warning_then_debug(caplog):
    import logging
    indexed_falkor._fallback_warned.clear()
    ex = _Executor(fail=True)
    with caplog.at_level(logging.DEBUG, logger="graphiti_service"):
        _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
        _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    lines = [r for r in caplog.records if "node vector search unavailable, falling back to scan" in r.getMessage()]
    assert [r.levelname for r in lines] == ["WARNING", "DEBUG"], "one WARNING per graph per process, then DEBUG"
    assert len([q for q in ex.queries if "db.idx.vector" in q]) == 2, "the index is still attempted every time"


def test_a_different_failure_on_the_same_graph_warns_again(caplog):
    """Keyed on the error too: at INFO level a second failure mode behind the
    first's DEBUG line would be the invisible failure of 2026-09-14 again."""
    import logging
    indexed_falkor._fallback_warned.clear()
    with caplog.at_level(logging.DEBUG, logger="graphiti_service"):
        indexed_falkor._log_fallback("node", "client_pokagon", RuntimeError("vector index not found"))
        indexed_falkor._log_fallback("node", "client_pokagon", RuntimeError("'n' not defined"))
        indexed_falkor._log_fallback("node", "client_pokagon", RuntimeError("'n' not defined"))
    lines = [r for r in caplog.records if "node vector search unavailable" in r.getMessage()]
    assert [r.levelname for r in lines] == ["WARNING", "WARNING", "DEBUG"]
    assert "'n' not defined" in lines[2].getMessage(), "the DEBUG line carries the error text"


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
