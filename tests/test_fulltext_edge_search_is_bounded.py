"""The fulltext edge search must bound BEFORE the join, not after it.

Measured on client_pokagon via FalkorDB's slowlog:

    576.3s  CALL db.idx.fulltext.queryRelationships('RELATES_TO', $query) ...
      0.4s  CALL db.idx.vector.queryRelationships('RELATES_TO','fact_embedding',2,...)

Same graph, same relationship type, 1400x apart. The structural difference is
where the bound goes. graphiti's get_relationships_query() takes a `limit` and
uses it for Kuzu (TOP :=) and Neo4j ({limit: ...}) but drops it for FalkorDB,
so the procedure yields EVERY match and the generated Cypher then runs
`MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)` per hit, applying
LIMIT only at the end -- after the expensive part.
"""

import asyncio

from app.services.indexed_falkor import FULLTEXT_OVERFETCH, IndexedFalkorSearchOperations


class _Executor:
    def __init__(self, fail=False):
        self.queries: list[str] = []
        self.fail = fail

    async def execute_query(self, cypher, **params):
        self.queries.append(cypher)
        # Fail ONLY the bounded form (its LIMIT precedes the MATCH). Failing the
        # fallback too would prove the fallback never runs -- the opposite of
        # what this asserts.
        flat = " ".join(cypher.split())
        bounded = "LIMIT" in flat and "MATCH (n:Entity)" in flat and flat.index("LIMIT") < flat.index("MATCH (n:Entity)")
        if self.fail and bounded:
            raise RuntimeError("unsupported")
        return [], None, None


def _run(ops, ex, limit=10):
    from graphiti_core.search.search_filters import SearchFilters

    return asyncio.run(
        ops.edge_fulltext_search(ex, "monthly close", SearchFilters(), ["client_pokagon"], limit)
    )


def test_the_bound_comes_before_the_match():
    ex = _Executor()
    _run(IndexedFalkorSearchOperations(), ex)
    q = " ".join(ex.queries[0].split())
    assert "LIMIT" in q, "the procedure output must be bounded"
    assert q.index("LIMIT") < q.index("MATCH (n:Entity)"), (
        "bounding after the join is the whole defect: every matched relationship "
        "gets re-matched by uuid before the limit applies"
    )


def test_it_overfetches_so_post_join_filters_do_not_starve_the_result():
    """group_id / uuid filters apply AFTER the bound, so a bare top-N returns short."""
    ex = _Executor()
    _run(IndexedFalkorSearchOperations(), ex, limit=10)
    q = " ".join(ex.queries[0].split())
    assert f"LIMIT {10 * FULLTEXT_OVERFETCH}" in q


def test_the_final_limit_is_still_the_callers():
    ex = _Executor()
    _run(IndexedFalkorSearchOperations(), ex, limit=10)
    q = " ".join(ex.queries[0].split())
    assert q.rstrip().endswith("LIMIT $limit"), "callers must still get what they asked for"


def test_it_still_scopes_to_the_group():
    ex = _Executor()
    _run(IndexedFalkorSearchOperations(), ex)
    assert "e.group_id IN $group_ids" in ex.queries[0]


def test_it_falls_back_rather_than_losing_the_search():
    ex = _Executor(fail=True)
    out = _run(IndexedFalkorSearchOperations(), ex)
    assert out == []
    assert len(ex.queries) == 2, "it must retry via graphiti's own query"
