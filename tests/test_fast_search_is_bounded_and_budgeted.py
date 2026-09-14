"""The fast search bounds its BM25 leg before the join and budgets both legs.

Measured 2026-09-14 on client_pokagon after the Phase 1 record backfill (25.9k
edges): "Who approved the Table Fill Inspection process map, and when?" took
68 s on the FAST path while the backend gives a search 8 s. The BM25 leg joined
every relationship matching "approved" and "process" back to its endpoints
before applying its limit; the hybrid path had the same defect and the same
fix. Each leg now also carries a FalkorDB query TIMEOUT: a BM25 leg past its
budget makes the request vector-only, a vector leg past its budget makes the
request answer empty on the fast path rather than fall back to the slower
hybrid search.
"""

import asyncio
from types import SimpleNamespace

import pytest
from redis.exceptions import ResponseError

from app.config import settings
from app.services import graphiti_client

ROW = ["e1", "The Finance Team owns the Monthly Close process.", "OWNS", "s", "Finance Team", "t", "Monthly Close", "['ep']", None, None, None]


class FakeGraph:
    def __init__(self, fail: str | None = None):
        self.calls: list[tuple[str, int | None]] = []
        self.fail = fail

    def _run(self, q):
        if self.fail == "bm25" and "fulltext" in q:
            raise ResponseError("Query timed out")
        if self.fail == "vector" and "db.idx.vector" in q:
            raise ResponseError("Query timed out")
        if self.fail == "vector-other" and "db.idx.vector" in q:
            raise ResponseError("Invalid input")
        return SimpleNamespace(result_set=[ROW])

    def query(self, q, params=None, timeout=None):
        self.calls.append((q, timeout))
        return self._run(q)

    def ro_query(self, q, params=None, timeout=None):
        self.calls.append(("ro:" + q, timeout))
        return self._run(q)


class FakeEmbedder:
    async def create(self, input_data):
        return [0.1, 0.2, 0.3]


@pytest.fixture
def fast(monkeypatch):
    def install(fail: str | None = None) -> FakeGraph:
        graph = FakeGraph(fail)
        monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: SimpleNamespace(select_graph=lambda name: graph))
        monkeypatch.setattr(graphiti_client, "_ensure_edge_vector_index", lambda g, n: None)
        monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: FakeEmbedder())
        monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", False)
        monkeypatch.setattr(settings, "search_bm25_timeout_ms", 2500)
        monkeypatch.setattr(settings, "search_vector_timeout_ms", 4000)
        return graph
    return install


def test_the_bm25_leg_is_bounded_before_the_join_and_both_legs_carry_their_budget(fast):
    graph = fast()
    edges = asyncio.run(graphiti_client._search_fast("pokagon", "who approved the process map", 10))
    assert [e.uuid for e in edges] == ["e1"]
    vector, bm25 = graph.calls
    assert "db.idx.vector.queryRelationships" in vector[0]
    assert vector[1] == 4000
    q = " ".join(bm25[0].split())
    assert "db.idx.fulltext.queryRelationships" in q
    assert bm25[1] == 2500
    bound = f"WITH rel, score ORDER BY score DESC LIMIT {10 * 2 * graphiti_client.FULLTEXT_OVERFETCH} MATCH (a:Entity)"
    assert bound in q, "the bound must precede the join"
    assert q.endswith("ORDER BY score DESC LIMIT 20"), "the caller's pool is still the final limit"
    assert q.index("LIMIT") < q.index("MATCH (a:Entity)")


def test_a_bm25_leg_past_its_budget_makes_the_request_vector_only(fast):
    graph = fast(fail="bm25")
    edges = asyncio.run(graphiti_client._search_fast("pokagon", "who approved the process map", 10))
    assert [e.uuid for e in edges] == ["e1"]
    assert len(graph.calls) == 2


def test_a_vector_leg_past_its_budget_answers_empty_on_the_fast_path_without_the_hybrid_fallback(fast, monkeypatch):
    fast(fail="vector")

    async def never(*_args, **_kwargs):
        raise AssertionError("the hybrid fallback must not run for a budgeted request")

    monkeypatch.setattr(graphiti_client, "get_client", never)
    edges, path = asyncio.run(graphiti_client.search_with_path("pokagon", "who approved the process map", 10))
    assert edges == []
    assert path == "fast"


def test_a_vector_leg_error_that_is_not_a_timeout_still_falls_back_to_hybrid(fast, monkeypatch):
    fast(fail="vector-other")
    called: list[str] = []

    class FakeClient:
        async def search(self, query, num_results, group_ids):
            called.append(query)
            return []

    async def get_client(_slug):
        return FakeClient()

    monkeypatch.setattr(graphiti_client, "get_client", get_client)
    edges, path = asyncio.run(graphiti_client.search_with_path("pokagon", "who approved the process map", 10))
    assert path == "hybrid_fallback"
    assert called == ["who approved the process map"]


def test_a_budget_exceeded_in_probe_mode_is_still_a_probe_failure(fast, monkeypatch):
    fast(fail="vector")
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", True)
    monkeypatch.setattr(graphiti_client, "_select_existing_probe_graph", lambda db, name: db.select_graph(name))
    with pytest.raises(graphiti_client.AcceptanceProbeReadError):
        asyncio.run(graphiti_client.search_with_path("pokagon", "who approved the process map", 1))


def test_graph_read_passes_the_budget_to_both_command_forms(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", False)
    graphiti_client._graph_read(graph, "RETURN 1", None, 1234)
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", True)
    graphiti_client._graph_read(graph, "RETURN 1", None, 1234)
    assert graph.calls == [("RETURN 1", 1234), ("ro:RETURN 1", 1234)]


def test_timeout_detection_reads_the_database_error_text():
    assert graphiti_client._is_query_timeout(ResponseError("Query timed out"))
    assert graphiti_client._is_query_timeout(TimeoutError())
    assert not graphiti_client._is_query_timeout(ResponseError("Invalid input"))


def test_the_budgets_default_inside_the_backend_search_window():
    assert 0 < settings.search_bm25_timeout_ms <= 4000
    assert 0 < settings.search_vector_timeout_ms <= 5000
