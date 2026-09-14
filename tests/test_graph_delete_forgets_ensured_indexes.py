"""A deleted graph must be re-ensured when it comes back under the same name.

Live, 2026-09-14 17:15Z: a scratch graph was deleted and recreated in one
process; the "index already ensured" marks survived, the vector indexes were
never recreated, and FalkorDB rejected every dedup index query with
"Attempted to access undefined attribute" until the process restarted. The
dedup fell back to the scan silently. Every graph delete now forgets the
marks on both the ingest and the search path.
"""

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import admin
from app.services import graphiti_client, indexed_falkor

GRAPH = "scratch_pokagon_records"


def _mark_all():
    indexed_falkor._node_vindex_ensured.add(GRAPH)
    indexed_falkor._edge_vindex_ensured_via.add(GRAPH)
    graphiti_client._edge_vindex_ensured.add(GRAPH)


def _marks():
    return (
        GRAPH in indexed_falkor._node_vindex_ensured,
        GRAPH in indexed_falkor._edge_vindex_ensured_via,
        GRAPH in graphiti_client._edge_vindex_ensured,
    )


def test_forget_graph_indexes_clears_every_mark():
    _mark_all()
    graphiti_client.forget_graph_indexes(GRAPH)
    assert _marks() == (False, False, False)
    graphiti_client.forget_graph_indexes(GRAPH)  # idempotent on an unknown graph


def test_the_delete_route_forgets_the_marks(monkeypatch):
    _mark_all()

    class _Graph:
        def delete(self):
            return None

    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: SimpleNamespace(select_graph=lambda name: _Graph()))
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    res = TestClient(app).post(
        "/admin/delete-graph",
        json={"client_slug": "pokagon", "confirm": "I understand this wipes all data", "scratch_graph": GRAPH},
    )
    assert res.status_code == 200, res.text
    assert _marks() == (False, False, False)


def test_evict_graph_forgets_the_marks():
    _mark_all()
    asyncio.run(graphiti_client.evict_graph(GRAPH))
    assert _marks() == (False, False, False)


def test_the_ensure_runs_again_after_a_forget():
    class _Executor:
        def __init__(self):
            self.creates = 0

        async def execute_query(self, cypher, **params):
            if "CREATE VECTOR INDEX" in cypher:
                self.creates += 1
            return [], None, None

    ex = _Executor()
    indexed_falkor._edge_vindex_ensured_via.discard(GRAPH)
    asyncio.run(indexed_falkor.ensure_edge_vector_index_via(ex, GRAPH, 8))
    asyncio.run(indexed_falkor.ensure_edge_vector_index_via(ex, GRAPH, 8))
    assert ex.creates == 1, "once per process while the graph lives"
    indexed_falkor.forget_graph(GRAPH)
    asyncio.run(indexed_falkor.ensure_edge_vector_index_via(ex, GRAPH, 8))
    assert ex.creates == 2, "a recreated graph gets its index again"
