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

from graphiti_core.search.search_filters import SearchFilters

from app.services import indexed_falkor
from app.services.indexed_falkor import VECTOR_OVERFETCH, IndexedFalkorSearchOperations

SCAN = "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)"


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
    assert "RELATES_TO {uuid: rel.uuid}" in q, "candidates are joined by uuid, not scanned"


def test_the_candidates_are_rescored_with_the_scan_cosine_and_the_procedure_score_is_not_used():
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"], limit=10)
    (q,) = _index_queries(ex)
    assert "YIELD relationship AS rel " in q and "YIELD relationship AS rel, score" not in q
    assert "vec.cosineDistance(e.fact_embedding, vecf32($search_vector))" in q
    assert "WHERE score > $min_score" in q
    assert q.rstrip().endswith("ORDER BY score DESC LIMIT $limit")
    assert f"'fact_embedding', {10 * VECTOR_OVERFETCH}, vecf32($search_vector)" in q
    p = [p for p in ex.params if "min_score" in p][0]
    assert p["min_score"] == 0.6 and p["limit"] == 10


def test_it_still_scopes_to_the_group():
    ex = _Executor()
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"])
    (q,) = _index_queries(ex)
    assert "WHERE e.group_id IN $group_ids" in q
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


def test_a_filtered_call_keeps_the_scan():
    from datetime import datetime, timezone
    from graphiti_core.search.search_filters import DateFilter, ComparisonOperator

    ex = _Executor()
    flt = SearchFilters(valid_at=[[DateFilter(date=datetime.now(timezone.utc), comparison_operator=ComparisonOperator.less_than)]])
    _search(IndexedFalkorSearchOperations(), ex, ["client_pokagon"], search_filter=flt)
    assert not _index_queries(ex)
    assert any(SCAN in q for q in ex.queries)
