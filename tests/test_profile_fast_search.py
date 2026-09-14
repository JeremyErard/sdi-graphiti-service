"""The profiler runs exactly the Cypher the fast search runs, and returns plans only.

The fast path's latency grew linearly with the candidate pool on an idle graph
(2026-09-14, about 70 ms per vector-index candidate at 25.9k edges). The slow
log ranks queries but shows no operators; this route returns the plan
FalkorDB executed per leg with its timings. One query builder feeds both the
search and the profiler, so what is measured is what runs.
"""

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import admin
from app.services import graphiti_client
from tests.test_fast_search_is_bounded_and_budgeted import ROW, FakeEmbedder


class ProfilingGraph:
    def __init__(self):
        self.profiled: list[tuple[str, dict]] = []
        self.queried: list[str] = []

    def profile(self, q, params=None):
        self.profiled.append((q, params or {}))
        return "Results\n    Project | Records produced: 20, Execution time: 1.2 ms\n        Conditional Traverse | (a)-[e:RELATES_TO]->(b) | Records produced: 20, Execution time: 900.0 ms\n            ProcedureCall | db.idx.vector.queryRelationships | Records produced: 20, Execution time: 3.1 ms\n"

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
    assert out["vector"]["query"] == graphiti_client.fast_search_leg_queries(20)[0]
    assert out["bm25"]["query"] == graphiti_client.fast_search_leg_queries(20)[1]
    assert {p["group_id"] for _, p in graph.profiled} == {"client_pokagon"}


def test_the_profiler_returns_plan_lines_and_timings_only(monkeypatch):
    graph = ProfilingGraph()
    _install(monkeypatch, graph)
    out = asyncio.run(graphiti_client.profile_fast_search("pokagon", "who approved the process map", 5))
    assert out["vector"]["plan"][0] == "Results"
    assert any("ProcedureCall | db.idx.vector.queryRelationships" in line for line in out["vector"]["plan"])
    assert isinstance(out["vector"]["ms"], float) or isinstance(out["vector"]["ms"], int)
    assert set(out) == {"graph_name", "pool", "embed_ms", "vector", "bm25"}
    assert set(out["vector"]) == {"ms", "query", "plan"}
    assert "Monthly Close" not in str(out), "no fact text leaves the profiler"


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
