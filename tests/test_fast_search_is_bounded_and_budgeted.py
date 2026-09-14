"""The fast search bounds its BM25 leg before the join and budgets both legs.

Measured 2026-09-14 on client_pokagon after the Phase 1 record backfill (25.9k
edges): "Who approved the Table Fill Inspection process map, and when?" took
68 s on the FAST path while the backend gives a search 8 s. The BM25 leg joined
every relationship matching "approved" and "process" back to its endpoints
before applying its limit; the hybrid path had the same defect and the same
fix. Each leg now carries a FalkorDB query TIMEOUT and the two legs run
concurrently: a BM25 leg past its budget makes the request vector-only, a
vector leg past its budget makes the request answer empty on the fast path
rather than fall back to the slower hybrid search. Only a query that really
ran out of time is a budget: a rejected TIMEOUT argument or any other error
keeps the fallback behaviour it always had.
"""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from app.config import settings
from app.services import graphiti_client

ROW = ["e1", "The Finance Team owns the Monthly Close process.", "OWNS", "s", "Finance Team", "t", "Monthly Close", "['ep']", None, None, None]
QUESTION = "who approved the process map"
BUDGET_MS = 40


class FakeGraph:
    """`fail` names a leg ("bm25" / "vector") and `how` the failure: a real
    TIMEOUT runs the budget out and then says so; a rejection fails at once."""

    def __init__(self, fail: str | None = None, how: str = "timeout", barrier: threading.Barrier | None = None):
        self.calls: list[tuple[str, int | None]] = []
        self.fail, self.how, self.barrier = fail, how, barrier

    def _run(self, q, timeout):
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        leg = "bm25" if "fulltext" in q else "vector" if "db.idx.vector" in q else None
        if leg == self.fail:
            if self.how == "timeout":
                time.sleep((timeout or BUDGET_MS) / 1000)
                raise ResponseError("Query timed out")
            if self.how == "rejected":
                raise ResponseError("The query TIMEOUT exceeds the TIMEOUT_MAX configuration parameter")
            if self.how == "instant-timed-out":
                raise ResponseError("Query timed out")
            if self.how == "connection-lost":
                time.sleep((timeout or BUDGET_MS) / 1000)
                raise RedisConnectionError("Error 60 connecting to falkordb:6379. Operation timed out.")
            raise ResponseError("Invalid input")
        return SimpleNamespace(result_set=[ROW])

    def query(self, q, params=None, timeout=None):
        self.calls.append((q, timeout))
        return self._run(q, timeout)

    def ro_query(self, q, params=None, timeout=None):
        self.calls.append(("ro:" + q, timeout))
        return self._run(q, timeout)

    def call(self, fragment: str) -> tuple[str, int | None]:
        (found,) = [c for c in self.calls if fragment in c[0]]
        return found


class FakeEmbedder:
    async def create(self, input_data):
        return [0.1, 0.2, 0.3]


@pytest.fixture
def fast(monkeypatch):
    def install(fail: str | None = None, how: str = "timeout", barrier=None, bm25_ms=BUDGET_MS, vector_ms=BUDGET_MS) -> FakeGraph:
        graph = FakeGraph(fail, how, barrier)
        monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: SimpleNamespace(select_graph=lambda name: graph))
        monkeypatch.setattr(graphiti_client, "_ensure_edge_vector_index", lambda g, n: None)
        monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: FakeEmbedder())
        monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", False)
        monkeypatch.setattr(settings, "search_bm25_timeout_ms", bm25_ms)
        monkeypatch.setattr(settings, "search_vector_timeout_ms", vector_ms)
        return graph
    return install


def _search(max_results=10):
    return asyncio.run(graphiti_client._search_fast("pokagon", QUESTION, max_results))


def _search_with_path(max_results=10):
    return asyncio.run(graphiti_client.search_with_path("pokagon", QUESTION, max_results))


def test_the_bm25_leg_is_bounded_before_the_join_and_both_legs_carry_their_budget(fast):
    graph = fast(bm25_ms=2500, vector_ms=2100)
    edges = _search()
    assert [e.uuid for e in edges] == ["e1"]
    vector = graph.call("db.idx.vector.queryRelationships")
    assert vector[1] == 2100
    bm25 = graph.call("db.idx.fulltext.queryRelationships")
    assert bm25[1] == 2500
    q = " ".join(bm25[0].split())
    bound = f"WITH rel, score ORDER BY score DESC LIMIT {10 * 2 * graphiti_client.FULLTEXT_OVERFETCH} MATCH (a:Entity)"
    assert bound in q, "the bound must precede the join"
    assert q.index("LIMIT") < q.index("MATCH (a:Entity)")
    assert "WHERE e.group_id = $group_id WITH a, e, b, score RETURN" in q, "score is carried into the final ordering, as in the proven query"
    assert q.endswith("ORDER BY score DESC LIMIT 20"), "the caller's pool is still the final limit"


def test_the_two_legs_run_concurrently_so_a_call_costs_the_larger_budget_not_the_sum(fast):
    # Each leg waits for the other inside the driver; sequential legs would
    # never meet and the barrier would break.
    graph = fast(barrier=threading.Barrier(2))
    edges = _search()
    assert [e.uuid for e in edges] == ["e1"]
    assert len(graph.calls) == 2


def test_a_bm25_leg_that_runs_out_of_time_makes_the_request_vector_only(fast, caplog):
    graph = fast(fail="bm25")
    with caplog.at_level(logging.WARNING):
        edges = _search()
    assert [e.uuid for e in edges] == ["e1"]
    assert len(graph.calls) == 2
    assert "fast BM25 leg exceeded" in caplog.text


def test_a_bm25_leg_that_fails_to_parse_is_vector_only_and_says_so_at_warning(fast, caplog):
    fast(fail="bm25", how="parse")
    with caplog.at_level(logging.WARNING):
        edges = _search()
    assert [e.uuid for e in edges] == ["e1"]
    assert "fast BM25 leg skipped" in caplog.text


def test_a_vector_leg_that_runs_out_of_time_answers_empty_on_the_fast_path_without_the_hybrid_fallback(fast, monkeypatch):
    fast(fail="vector")

    async def never(*_args, **_kwargs):
        raise AssertionError("the hybrid fallback must not run for a budgeted request")

    monkeypatch.setattr(graphiti_client, "get_client", never)
    edges, path = _search_with_path()
    assert edges == []
    assert path == "fast"


@pytest.mark.parametrize("how", ["rejected", "parse", "instant-timed-out", "connection-lost"])
def test_a_vector_leg_error_that_is_not_a_spent_budget_still_falls_back_to_hybrid(fast, monkeypatch, how):
    # A rejected TIMEOUT argument, a parse error, a "timed out" that came
    # back at once (the budget was not spent), or a lost connection whose
    # transport wording also says "timed out" all keep the prior behaviour.
    fast(fail="vector", how=how, vector_ms=5000 if how != "connection-lost" else BUDGET_MS)
    called: list[str] = []

    class FakeClient:
        async def search(self, query, num_results, group_ids):
            called.append(query)
            return []

    async def get_client(_slug):
        return FakeClient()

    monkeypatch.setattr(graphiti_client, "get_client", get_client)
    edges, path = _search_with_path()
    assert path == "hybrid_fallback"
    assert called == [QUESTION]


def test_a_spent_budget_in_probe_mode_is_still_a_probe_failure(fast, monkeypatch):
    fast(fail="vector")
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", True)
    monkeypatch.setattr(graphiti_client, "_select_existing_probe_graph", lambda db, name: db.select_graph(name))
    with pytest.raises(graphiti_client.AcceptanceProbeReadError):
        _search_with_path(1)


def test_graph_read_passes_the_budget_to_both_command_forms_and_nothing_extra_without_one(monkeypatch):
    class Recorder:
        def __init__(self):
            self.calls = []

        def query(self, *args, **kwargs):
            self.calls.append(("query", args, kwargs))

        def ro_query(self, *args, **kwargs):
            self.calls.append(("ro_query", args, kwargs))

    graph = Recorder()
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", False)
    graphiti_client._graph_read(graph, "RETURN 1", None, 1234)
    graphiti_client._graph_read(graph, "RETURN 1", None)
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", True)
    graphiti_client._graph_read(graph, "RETURN 1", None, 1234)
    assert graph.calls == [
        ("query", ("RETURN 1", None), {"timeout": 1234}),
        ("query", ("RETURN 1", None), {}),
        ("ro_query", ("RETURN 1", None), {"timeout": 1234}),
    ]


def test_timeout_detection_needs_both_the_text_and_the_elapsed_time():
    spent = graphiti_client._is_query_timeout
    assert spent(ResponseError("Query timed out"), 2500, 2500)
    assert spent(ResponseError("Query timed out"), 2300, 2500)
    assert not spent(ResponseError("Query timed out"), 5, 2500)
    assert not spent(ResponseError("The query TIMEOUT exceeds the TIMEOUT_MAX configuration parameter"), 5, 2500)
    assert not spent(TimeoutError("Timeout reading from socket"), 900000, 2500)
    assert not spent(RedisConnectionError("Error 60 connecting to falkordb:6379. Operation timed out."), 3000, 2500)
    assert not spent(RedisConnectionError("Error timed out connecting to falkordb:6379."), 3000, 2500)
    assert not spent(ResponseError("Invalid input"), 2500, 2500)


def test_the_budgets_leave_the_preview_path_two_calls_inside_the_backend_window():
    # Concurrent legs: a call costs at most max(budgets); the preview path
    # makes two calls, each preceded by a query embedding.
    per_call = max(settings.search_bm25_timeout_ms, settings.search_vector_timeout_ms)
    assert 0 < per_call
    assert 2 * per_call <= 5000
