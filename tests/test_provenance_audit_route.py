"""The dormant provenance audit gets a read-only admin surface.

The first live graph-stats run (2026-09-11) found 7,244 of the Pokagon graph's
7,757 facts written before the provenance contract. The planner that says how
many can be anchored deterministically was reachable only as a CLI inside the
container. This route exposes the audit: signed admin scope, one exact tenant,
no apply flag on the wire, bounded counts and codes only.
"""

import json
import time
import uuid

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.auth import build_signature, require_scope
from app.config import settings
from app.routers import admin
from app.services.provenance_ops import (
    AUDIT_GRAPH_NOT_FOUND_CODE,
    ProvenanceAuditReadError,
)

ADMIN_SECRET = "admin-secret-that-is-at-least-32-characters"
SUMMARY = {
    "mode": "audit",
    "counts": {
        "episodes_scanned": 7,
        "edges_scanned": 7757,
        "episode_anchor_updates_planned": 5,
        "edge_endpoint_updates_planned": 0,
        "edge_episode_list_updates_planned": 7244,
        "apply_attempted": 0,
        "apply_succeeded": 0,
        "apply_conflicts": 0,
    },
    "codes": {"EPISODE_ALREADY_ANCHORED": 2},
}


def _headers(body: bytes, client_slug: str = "pokagon") -> dict:
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    signature = build_signature(
        secret=ADMIN_SECRET, timestamp=timestamp, nonce=nonce, method="POST",
        path="/admin/provenance-audit", scope="admin", client_slug=client_slug, body=body,
    )
    return {
        "content-type": "application/json",
        "x-sdi-kg-timestamp": timestamp,
        "x-sdi-kg-scope": "admin",
        "x-sdi-kg-client": client_slug,
        "x-sdi-kg-nonce": nonce,
        "x-sdi-kg-signature": signature,
    }


def _encoded(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


@pytest.fixture(autouse=True)
def _auth(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required")
    monkeypatch.setattr(settings, "graphiti_admin_secret", ADMIN_SECRET)
    monkeypatch.setattr(settings, "graphiti_auth_max_clock_skew_seconds", 300)

    async def consume(scope: str, nonce: str) -> bool:
        return True

    monkeypatch.setattr(auth, "_consume_nonce", consume)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin", dependencies=[Depends(require_scope("admin"))])
    return TestClient(app)


def test_the_audit_is_post_only_signed_and_scoped_to_one_tenant(monkeypatch):
    calls = []

    def fake_audit(client_slug, **kwargs):
        calls.append((client_slug, kwargs))
        return SUMMARY

    monkeypatch.setattr(admin, "run_provenance_audit", fake_audit)
    client = _client()

    assert client.get("/admin/provenance-audit").status_code == 405
    assert client.post("/admin/provenance-audit", json={"client_slug": "pokagon"}).status_code == 401

    body = _encoded({"client_slug": "pokagon"})
    response = client.post("/admin/provenance-audit", content=body, headers=_headers(body))
    assert response.status_code == 200
    assert response.json() == SUMMARY
    assert calls == [("pokagon", {})]


def test_no_apply_flag_exists_on_the_wire(monkeypatch):
    monkeypatch.setattr(admin, "run_provenance_audit", lambda client_slug, **kwargs: SUMMARY)
    client = _client()

    body = _encoded({"client_slug": "pokagon", "apply": True})
    response = client.post("/admin/provenance-audit", content=body, headers=_headers(body))
    assert response.status_code == 422


def test_a_missing_graph_is_a_409_with_the_module_code(monkeypatch):
    def missing(client_slug, **kwargs):
        raise ProvenanceAuditReadError(AUDIT_GRAPH_NOT_FOUND_CODE)

    monkeypatch.setattr(admin, "run_provenance_audit", missing)
    client = _client()

    body = _encoded({"client_slug": "nobody"})
    response = client.post("/admin/provenance-audit", content=body, headers=_headers(body, "nobody"))
    assert response.status_code == 409
    assert response.json() == {"detail": AUDIT_GRAPH_NOT_FOUND_CODE}


def test_a_database_failure_is_a_generic_502_with_the_cause_logged(monkeypatch, caplog):
    def boom(client_slug, **kwargs):
        raise RuntimeError("FalkorDB does not currently support =~")

    monkeypatch.setattr(admin, "run_provenance_audit", boom)
    client = _client()

    body = _encoded({"client_slug": "pokagon"})
    import logging

    with caplog.at_level(logging.ERROR, logger="graphiti_service"):
        response = client.post("/admin/provenance-audit", content=body, headers=_headers(body))
    assert response.status_code == 502
    assert response.json() == {"detail": "provenance-audit failed"}
    assert "=~" not in response.text
    assert any("=~" in record.getMessage() for record in caplog.records)
