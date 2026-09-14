"""The profiler runs exactly the Cypher the fast search runs, and returns plans only.

The fast path's latency grew linearly with the candidate pool on an idle graph
(2026-09-14, about 70 ms per vector-index candidate at 25.9k edges). The slow
log ranks queries but shows no operators; this route returns the plan
FalkorDB executed per leg with its timings. One query builder feeds both the
search and the profiler, so what is measured is what runs.
"""

import asyncio
from types import SimpleNamespace

import pytest
from falkordb.execution_plan import ExecutionPlan
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import admin
from app.services import graphiti_client
from tests.test_fast_search_is_bounded_and_budgeted import ROW, FakeEmbedder


# The lines FalkorDB returns for GRAPH.PROFILE, as the driver hands them to
# ExecutionPlan: operator, arguments, then the records and timing segment.
RAW_PLAN = [
    "Results",
    "    Project | Records produced: 20, Execution time: 1.200000 ms",
    "        Conditional Traverse | (a)-[e:RELATES_TO]->(b) | Records produced: 20, Execution time: 900.100000 ms",
    "            ProcedureCall | db.idx.vector.queryRelationships | Records produced: 20, Execution time: 3.100000 ms",
]


class ProfilingGraph:
    def __init__(self):
        self.profiled: list[tuple[str, dict]] = []
        self.queried: list[str] = []

    def profile(self, q, params=None):
        self.profiled.append((q, params or {}))
        return ExecutionPlan(list(RAW_PLAN))

    def query(self, q, params=None, timeout=None):
        self.queried.append(q)
        return SimpleNamespace(result_set=[ROW])


def _install(monkeypatch, graph):
    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: SimpleNamespace(select_graph=lambda name: graph))
    monkeypatch.setattr(graphiti_client, "_ensure_edge_vector_index", lambda g, n: None)
    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", False)


def test_the_profiler_runs_exactly_the_queries_the_search_runs(monkeypatch):
    graph = ProfilingGraph()
    _install(monkeypatch, graph)
    asyncio.run(graphiti_client._search_fast("pokagon", "who approved the process map", 10))
    out = asyncio.run(graphiti_client.profile_fast_search("pokagon", "who approved the process map", 10))
    searched = sorted(graph.queried)
    profiled = sorted(q for q, _ in graph.profiled)
    assert profiled == searched, "the profiler must measure the Cypher the search issues"
    assert out["pool"] == 20
    assert out["graph_name"] == "client_pokagon"
    assert {p["group_id"] for _, p in graph.profiled} == {"client_pokagon"}
    assert out["vector"]["query"] == graphiti_client.fast_search_leg_queries(20)[0]
    assert out["bm25"]["query"] == graphiti_client.fast_search_leg_queries(20)[1]


def test_a_plan_the_driver_cannot_parse_is_read_raw_from_the_server(monkeypatch):
    """ExecutionPlan parses in __init__ and raises on an integer-formatted time
    or four spaces inside an argument; the route then reads GRAPH.PROFILE raw."""
    raw = ["Results", "    Project | Records produced: 20, Execution time: 0 ms"]

    class Unparseable(ProfilingGraph):
        _name = "client_pokagon"

        def profile(self, q, params=None):
            self.profiled.append((q, params or {}))
            return ExecutionPlan(list(raw))  # raises inside the driver

        def _build_params_header(self, params):
            return "CYPHER q=[] "

        def execute_command(self, *args):
            assert args[0] == "GRAPH.PROFILE" and args[1] == "client_pokagon"
            return [line.encode() for line in raw]

    _install(monkeypatch, Unparseable())
    out = asyncio.run(graphiti_client.profile_fast_search("pokagon", "who approved", 1))
    assert out["vector"]["plan"] == raw


def test_a_server_refusal_is_not_retried_raw(monkeypatch):
    from redis.exceptions import ResponseError

    class Refusing(ProfilingGraph):
        def profile(self, q, params=None):
            raise ResponseError("Unknown procedure")

        def execute_command(self, *args):
            raise AssertionError("a refused query must not be re-sent")

    _install(monkeypatch, Refusing())
    with pytest.raises(ResponseError):
        asyncio.run(graphiti_client.profile_fast_search("pokagon", "who approved", 1))


def test_the_leg_queries_are_the_ones_the_search_issued_before_the_builder_existed():
    """Pinned literals, so a drift in the shared builder moves the search and
    the profiler together and this test still notices."""
    vector, bm25 = graphiti_client.fast_search_leg_queries(20)
    assert vector == (
        "CALL db.idx.vector.queryRelationships('RELATES_TO', 'fact_embedding', 20, vecf32($q)) "
        "YIELD relationship AS rel, score "
        "MATCH (a:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(b:Entity) WHERE e.group_id = $group_id "
        "RETURN e.uuid AS uuid, e.fact AS fact, e.name AS name, a.uuid AS src, a.name AS src_name, "
        "b.uuid AS tgt, b.name AS tgt_name, e.episodes AS episodes, e.valid_at AS va, e.invalid_at AS ia, "
        "e.expired_at AS ea"
    )
    assert bm25 == (
        "CALL db.idx.fulltext.queryRelationships('RELATES_TO', $query) YIELD relationship AS rel, score "
        "WITH rel, score ORDER BY score DESC LIMIT 200 "
        "MATCH (a:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(b:Entity) WHERE e.group_id = $group_id "
        "WITH a, e, b, score RETURN e.uuid AS uuid, e.fact AS fact, e.name AS name, a.uuid AS src, "
        "a.name AS src_name, b.uuid AS tgt, b.name AS tgt_name, e.episodes AS episodes, e.valid_at AS va, "
        "e.invalid_at AS ia, e.expired_at AS ea ORDER BY score DESC LIMIT 20"
    )


def test_the_profiler_returns_plan_lines_and_timings_only(monkeypatch):
    graph = ProfilingGraph()
    _install(monkeypatch, graph)
    out = asyncio.run(graphiti_client.profile_fast_search("pokagon", "who approved the process map", 5))
    assert out["vector"]["plan"] == RAW_PLAN, "the server's lines, records and timings intact"
    assert any("Execution time: 900.100000 ms" in line for line in out["vector"]["plan"]), "the per-operation timing is the point of the route"
    assert isinstance(out["vector"]["ms"], (float, int))
    assert set(out) == {"graph_name", "pool", "embed_ms", "vector", "bm25"}
    assert set(out["vector"]) == {"ms", "query", "plan"}
    assert "Monthly Close" not in str(out), "no fact text leaves the profiler"


def test_the_result_count_is_capped_where_the_search_budget_would_be(monkeypatch):
    graph = ProfilingGraph()
    _install(monkeypatch, graph)
    with pytest.raises(ValueError):
        asyncio.run(graphiti_client.profile_fast_search("pokagon", "q", graphiti_client.PROFILE_MAX_RESULTS + 1))
    assert graph.profiled == []
    res = _admin_client().post("/admin/profile-fast-search", json={"client_slug": "pokagon", "query": "q", "max_results": 50})
    assert res.status_code == 422


@pytest.mark.parametrize("slow_leg", ["db.idx.vector.queryRelationships", "db.idx.fulltext.queryRelationships"])
def test_a_leg_past_the_profile_bound_is_reported_and_the_other_leg_still_profiled(monkeypatch, slow_leg):
    """Each leg is bounded on its own: only the named leg is slow here, so an
    unbounded leg would return a real plan instead of the exceeded marker."""
    import time as _time

    class Slow(ProfilingGraph):
        def profile(self, q, params=None):
            if slow_leg in q:
                _time.sleep(0.3)
            return super().profile(q, params)

    _install(monkeypatch, Slow())
    monkeypatch.setattr(graphiti_client, "PROFILE_LEG_TIMEOUT_SECONDS", 0.05)
    out = asyncio.run(graphiti_client.profile_fast_search("pokagon", "who approved the process map", 1))
    slow, other = ("vector", "bm25") if "vector" in slow_leg else ("bm25", "vector")
    assert out[slow]["plan"] == ["(exceeded the 0.05 s profile bound)"]
    assert out[slow]["ms"] == 50
    assert out[other]["plan"] == RAW_PLAN, "the other leg is still profiled"


def test_a_scratch_graph_is_profiled_under_its_own_name(monkeypatch):
    graph = ProfilingGraph()
    _install(monkeypatch, graph)
    out = asyncio.run(graphiti_client.profile_fast_search("pokagon", "who approved", 1, scratch_graph="scratch_pokagon_records"))
    assert out["graph_name"] == "scratch_pokagon_records"
    assert {p["group_id"] for _, p in graph.profiled} == {"scratch_pokagon_records"}


def _admin_client() -> TestClient:
    # The admin scope is enforced where app.main mounts the router
    # (dependencies=[Depends(require_scope("admin"))]) and is covered by the
    # auth suite; the route tests mount the router bare, as the rehearsal
    # tests do.
    bare = FastAPI()
    bare.include_router(admin.router, prefix="/admin")
    return TestClient(bare)


def test_the_route_lives_under_the_admin_router_and_returns_the_profile(monkeypatch):
    graph = ProfilingGraph()
    _install(monkeypatch, graph)
    assert any(getattr(r, "path", "") == "/profile-fast-search" for r in admin.router.routes)
    res = _admin_client().post(
        "/admin/profile-fast-search",
        json={"client_slug": "pokagon", "query": "who approved the process map", "max_results": 10},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["pool"] == 20
    assert data["vector"]["plan"][0] == "Results"
    assert data["bm25"]["ms"] >= 0


def test_the_route_rejects_a_non_scratch_graph_name(monkeypatch):
    graph = ProfilingGraph()
    _install(monkeypatch, graph)
    res = _admin_client().post(
        "/admin/profile-fast-search",
        json={"client_slug": "pokagon", "query": "q", "max_results": 1, "scratch_graph": "client_thrive"},
    )
    assert res.status_code == 422
    assert graph.profiled == []


def test_a_database_failure_is_a_clean_502(monkeypatch):
    class Broken(ProfilingGraph):
        def profile(self, q, params=None):
            raise RuntimeError("Unknown procedure")

    _install(monkeypatch, Broken())
    res = _admin_client().post(
        "/admin/profile-fast-search",
        json={"client_slug": "pokagon", "query": "q", "max_results": 1},
    )
    assert res.status_code == 502
    assert "Unknown procedure" not in res.text
