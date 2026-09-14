"""Edge dedup must go through the RELATES_TO vector index, and degrade rather than break.

graphiti's edge resolution runs a full cosine scan over every RELATES_TO in the
group for every extracted edge (see the query in FalkorSearchOperations.
edge_similarity_search). Measured 2026-09-14 on client_pokagon (25.9k edges):
those scans, dozens per episode, are what queue the service's own searches past
their 2.5 s budget while an ingest runs. The search path has used the HNSW index
since the graph had 7k edges; the ingest path now gets the same treatment.

The procedure's own score is never used for the threshold: the k candidates are
re-scored with the cosine expression graphiti's scan uses, so `min_score` and
the ordering keep their meaning.
"""

import asyncio
import re

import pytest

from graphiti_core.search.search_filters import SearchFilters

from app.services import indexed_falkor
from app.services.indexed_falkor import VECTOR_OVERFETCH, IndexedFalkorSearchOperations

SCAN = "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)"



from tests.conftest import assert_with_scopes_are_sound  # noqa: E402


class _Executor:
    def __init__(self, fail: bool = False):
        self.queries: list[str] = []
        self.params: list[dict] = []
        self.fail = fail

    async def execute_query(self, cypher, **params):
        self.queries.append(cypher)
        self.params.append(params)
        if self.fail and "db.idx.vector.queryRelationships" in cypher:
            raise RuntimeError("Unknown procedure 'db.idx.vector.queryRelationships'")
        return [], None, None


def _search(ops, executor, group_ids=None, search_filter=None, source=None, target=None, limit=10):
    indexed_falkor._edge_vindex_ensured_via.clear()
    return asyncio.run(
        ops.edge_similarity_search(
            executor, [0.1, 0.2], source, target, search_filter or SearchFilters(), group_ids, limit, 0.6
        )
    )


def _index_queries(ex):
    return [q for q in ex.queries if "db.idx.vector.queryRelationships" in q]


def test_it_queries_the_index_instead_of_scanning_every_edge():
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    (q,) = _index_queries(ex)
    assert not any(SCAN in x for x in ex.queries), "a bare endpoint MATCH means it is still scanning"
    assert "{uuid: rel.uuid}" not in q, "no join by uuid: it was planned as a label scan per candidate"
    assert "WITH e, startNode(e) AS n, endNode(e) AS m" in q, "endpoints come from the relationship itself"


def test_the_candidates_are_rescored_with_the_scan_cosine_and_the_procedure_score_is_not_used():
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"], limit=10)
    (q,) = _index_queries(ex)
    assert "YIELD relationship AS e, score AS index_score " in q, "every procedure field yielded, the score aliased away"
    assert "index_score" not in q.split("index_score ", 1)[1], "the procedure score is never used after the YIELD"
    assert "vec.cosineDistance(e.fact_embedding, vecf32($search_vector))" in q
    assert "WHERE score > $min_score" in q
    assert q.rstrip().endswith("ORDER BY score DESC LIMIT $limit")
    assert "'fact_embedding', 40, vecf32($search_vector)" in q, "four times the requested 10, so the post-filters keep recall"
    assert VECTOR_OVERFETCH == 4
    p = [p for p in ex.params if "min_score" in p][0]
    assert p["min_score"] == 0.6 and p["limit"] == 10


def test_it_still_scopes_to_the_group():
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    (q,) = _index_queries(ex)
    assert "WHERE score > $min_score AND e.group_id IN $group_ids" in q
    assert any(p.get("group_ids") == ["client_pokagon"] for p in ex.params)


def test_it_ensures_the_index_through_the_same_executor_once():
    ex = _Executor()
    ops = IndexedFalkorSearchOperations()
    _search(ops, ex, ["client_pokagon"])
    creates = [q for q in ex.queries if "CREATE VECTOR INDEX FOR ()-[r:RELATES_TO]->() ON (r.fact_embedding)" in q]
    assert len(creates) == 1
    asyncio.run(ops.edge_similarity_search(ex, [0.1, 0.2], None, None, SearchFilters(), ["client_pokagon"], 10, 0.6))
    creates = [q for q in ex.queries if "CREATE VECTOR INDEX" in q]
    assert len(creates) == 1, "one attempt per process, not one per call"


def test_a_dead_override_is_visible_once_at_warning_then_debug(caplog):
    import logging
    indexed_falkor._fallback_warned.clear()
    ex = _Executor(fail=True)
    with caplog.at_level(logging.DEBUG, logger="graphiti_service"):
        _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
        _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    lines = [r for r in caplog.records if "edge vector search unavailable, falling back to scan" in r.getMessage()]
    assert [r.levelname for r in lines] == ["WARNING", "DEBUG"], "one WARNING per graph per process, then DEBUG"
    assert len([q for q in ex.queries if "db.idx.vector" in q]) == 2, "the index is still attempted every time"


def test_a_different_failure_on_the_same_graph_warns_again(caplog):
    """Keyed on the error too: at INFO level a second failure mode behind the
    first's DEBUG line would be the invisible failure of 2026-09-14 again."""
    import logging
    indexed_falkor._fallback_warned.clear()
    with caplog.at_level(logging.DEBUG, logger="graphiti_service"):
        indexed_falkor._log_fallback("edge", "client_pokagon", RuntimeError("vector index not found"))
        indexed_falkor._log_fallback("edge", "client_pokagon", RuntimeError("'n' not defined"))
        indexed_falkor._log_fallback("edge", "client_pokagon", RuntimeError("'n' not defined"))
    levels = [r.levelname for r in caplog.records if "edge vector search unavailable" in r.getMessage()]
    assert levels == ["WARNING", "WARNING", "DEBUG"]
    assert "'n' not defined" in caplog.text


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
    # a MATCH after a WITH binds new variables the following clauses may read
    assert_with_scopes_are_sound("CALL p() YIELD relationship AS rel, score WITH rel, score ORDER BY score DESC LIMIT 200 MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity) WHERE e.group_id IN $g WITH e, score, n, m RETURN e.uuid, n.uuid, m.uuid ORDER BY score DESC LIMIT $l")


def test_it_falls_back_to_the_scan_when_the_index_is_missing():
    ex = _Executor(fail=True)
    out = _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    assert out == []
    assert _index_queries(ex), "the index was tried first"
    assert any(SCAN in q and "cosineDistance" in q for q in ex.queries), "then graphiti's scan ran"


def test_an_endpoint_constrained_call_keeps_the_scan():
    """The scan path is bounded by the Entity.uuid index there; the vector
    top-k could miss an edge between two specific nodes."""
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"], source="s-1", target="t-1")
    assert not _index_queries(ex)
    (q,) = [q for q in ex.queries if SCAN in q]
    assert "n.uuid = $source_uuid" in q and "m.uuid = $target_uuid" in q


def test_an_empty_edge_uuids_filter_is_a_filter_and_keeps_the_scan():
    """graphiti's related-edges search passes edge_uuids=[] for a node pair
    with no edges yet (the common case in incremental ingest) and expects
    nothing back. Read as "no filter", the index would hand the duplicate
    resolver the nearest edges of the whole graph."""
    ex = _Executor()
    out = _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"], search_filter=SearchFilters(edge_uuids=[]))
    assert out == []
    assert not _index_queries(ex)
    (q,) = [q for q in ex.queries if SCAN in q]
    assert "e.uuid in $edge_uuids" in q.lower().replace("$edge_uuids", "$edge_uuids")
    assert any(p.get("edge_uuids") == [] for p in ex.params)


def test_the_index_is_marked_ensured_only_after_the_create_returns():
    """A concurrent first burst must not find the key marked while the CREATE
    is still in flight, query a missing index, and fall back to the scan."""
    seen: list[bool] = []

    class _Probe(_Executor):
        async def execute_query(self, cypher, **params):
            if "CREATE VECTOR INDEX" in cypher:
                seen.append("client_pokagon" in indexed_falkor._edge_vindex_ensured_via)
                raise RuntimeError("already exists")
            return await super().execute_query(cypher, **params)

    ex = _Probe()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    assert seen == [False], "not marked before the attempt"
    assert "client_pokagon" in indexed_falkor._edge_vindex_ensured_via, "marked after it, even when it raised"


def test_a_filtered_call_keeps_the_scan():
    from datetime import datetime, timezone
    from graphiti_core.search.search_filters import DateFilter, ComparisonOperator

    ex = _Executor()
    flt = SearchFilters(valid_at=[[DateFilter(date=datetime.now(timezone.utc), comparison_operator=ComparisonOperator.less_than)]])
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"], search_filter=flt)
    assert not _index_queries(ex)
    assert any(SCAN in q for q in ex.queries)


def test_the_search_legs_and_the_fulltext_override_pass_the_scope_check():
    from app.services import graphiti_client

    for q in graphiti_client.fast_search_leg_queries(20):
        assert_with_scopes_are_sound(q)
