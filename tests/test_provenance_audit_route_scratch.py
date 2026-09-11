"""The audit route's new fields: apply, batch_engagement_id, scratch_graph.

Apply on a tenant graph is a 409 carrying the module's block code; on a
scratch copy the fields pass through to the audit. graph-stats and
delete-graph accept a scratch name and nothing else as an override.
"""

import json
import time
import uuid

import falkordb
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.auth import build_signature, require_scope
from app.config import settings
from app.routers import admin

ADMIN_SECRET = "admin-secret-that-is-at-least-32-characters"


def _headers(path: str, body: bytes, client_slug: str = "pokagon") -> dict:
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    signature = build_signature(secret=ADMIN_SECRET, timestamp=timestamp, nonce=nonce, method="POST", path=path, scope="admin", client_slug=client_slug, body=body)
    return {"content-type": "application/json", "x-sdi-kg-timestamp": timestamp, "x-sdi-kg-scope": "admin", "x-sdi-kg-client": client_slug, "x-sdi-kg-nonce": nonce, "x-sdi-kg-signature": signature}


def _encoded(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class _Result:
    def __init__(self, rows):
        self.result_set = rows


class _Graph:
    deleted: list[str] = []

    def __init__(self, name):
        self.name = name

    def query(self, query, params=None):
        if "MATCH (n)" in query:
            return _Result([[3]])
        if "MATCH ()-[r]->()" in query:
            return _Result([[4]])
        raise AssertionError(query)

    def delete(self):
        _Graph.deleted.append(self.name)


class _FalkorDB:
    selected: list[str] = []

    def __init__(self, **_kwargs):
        pass

    def list_graphs(self):
        return ["client_pokagon", "scratch_pokagon_provenance"]

    def select_graph(self, name):
        _FalkorDB.selected.append(name)
        return _Graph(name)


@pytest.fixture(autouse=True)
def _auth(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required")
    monkeypatch.setattr(settings, "graphiti_admin_secret", ADMIN_SECRET)
    monkeypatch.setattr(settings, "graphiti_auth_max_clock_skew_seconds", 300)
    monkeypatch.setattr(falkordb, "FalkorDB", _FalkorDB)
    _FalkorDB.selected = []
    _Graph.deleted = []

    async def consume(scope: str, nonce: str) -> bool:
        return True

    monkeypatch.setattr(auth, "_consume_nonce", consume)
    from app.services import graphiti_client

    monkeypatch.setattr(graphiti_client, "_falkor_db", None, raising=False)

    async def quiet(*_a, **_k):
        return None

    monkeypatch.setattr(graphiti_client, "_log_slow_queries", quiet)
    monkeypatch.setattr(graphiti_client, "_log_memory_usage", quiet)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin", dependencies=[Depends(require_scope("admin"))])
    return TestClient(app)


def test_the_scratch_fields_pass_through_to_the_audit(monkeypatch):
    calls = []

    def record(client_slug, **kwargs):
        calls.append((client_slug, kwargs))
        return {"mode": "apply", "counts": {}, "codes": {}}

    monkeypatch.setattr(admin, "run_provenance_audit", record)
    body = _encoded({"client_slug": "pokagon", "apply": True, "batch_engagement_id": "cmmnn76l60000lqzwaua2zed7", "scratch_graph": "scratch_pokagon_provenance"})
    response = _client().post("/admin/provenance-audit", content=body, headers=_headers("/admin/provenance-audit", body))
    assert response.status_code == 200
    assert calls == [("pokagon", {"apply": True, "batch_engagement_id": "cmmnn76l60000lqzwaua2zed7", "scratch_graph": "scratch_pokagon_provenance"})]


@pytest.mark.parametrize("bad", ["client_pokagon", "scratch-pokagon", "SCRATCH_x", "scratch_"])
def test_a_non_scratch_override_is_rejected_at_the_wire(monkeypatch, bad):
    monkeypatch.setattr(admin, "run_provenance_audit", lambda client_slug, **kwargs: {"mode": "audit", "counts": {}, "codes": {}})
    body = _encoded({"client_slug": "pokagon", "scratch_graph": bad})
    assert _client().post("/admin/provenance-audit", content=body, headers=_headers("/admin/provenance-audit", body)).status_code == 422
    body = _encoded({"client_slug": "pokagon", "scratch_graph": bad})
    assert _client().post("/admin/graph-stats", content=body, headers=_headers("/admin/graph-stats", body)).status_code == 422
    body = _encoded({"client_slug": "pokagon", "confirm": "I understand this wipes all data", "scratch_graph": bad})
    assert _client().post("/admin/delete-graph", content=body, headers=_headers("/admin/delete-graph", body)).status_code == 422


def test_graph_stats_reads_the_scratch_copy_when_asked():
    body = _encoded({"client_slug": "pokagon", "scratch_graph": "scratch_pokagon_provenance"})
    response = _client().post("/admin/graph-stats", content=body, headers=_headers("/admin/graph-stats", body))
    assert response.status_code == 200
    assert response.json() == {"graphs": [{"graph_name": "scratch_pokagon_provenance", "nodes": 3, "edges": 4}], "graph_count": 1}
    assert _FalkorDB.selected == ["scratch_pokagon_provenance"]


def test_delete_graph_removes_the_scratch_copy_and_never_the_tenant_graph():
    body = _encoded({"client_slug": "pokagon", "confirm": "I understand this wipes all data", "scratch_graph": "scratch_pokagon_provenance"})
    response = _client().post("/admin/delete-graph", content=body, headers=_headers("/admin/delete-graph", body))
    assert response.status_code == 200
    assert response.json()["graph_name"] == "scratch_pokagon_provenance"
    assert _Graph.deleted == ["scratch_pokagon_provenance"]
