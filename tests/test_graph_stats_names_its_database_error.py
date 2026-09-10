"""When a provenance read fails inside the database, the log must say why.

On 2026-09-10 the first live ``include_provenance=true`` call against the
Pokagon graph returned 502 and logged only ``error_type=ResponseError``. That
is the surface the ingestion completeness audit depends on, and the cause was
unrecoverable from a healthy deployment. The database's message names the
failing clause and carries no fact text; it goes to the log, never the wire.
"""

import json
import logging
import time
import uuid

import falkordb
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from redis.exceptions import ResponseError

from app import auth
from app.auth import build_signature, require_scope
from app.config import settings
from app.routers import admin

ADMIN_SECRET = "admin-secret-that-is-at-least-32-characters"
DB_MESSAGE = "Type mismatch: expected String but was Integer"


class _Result:
    def __init__(self, rows):
        self.result_set = rows


class _Graph:
    def __init__(self, name):
        self.name = name

    def query(self, query, params=None):
        if "MATCH (n)" in query:
            return _Result([[11]])
        if "MATCH ()-[r]->()" in query:
            return _Result([[22]])
        raise AssertionError("provenance reads must use ro_query")

    def ro_query(self, query, params=None):
        raise ResponseError(DB_MESSAGE)


class _FalkorDB:
    def __init__(self, **_kwargs):
        pass

    def list_graphs(self):
        return ["client_pokagon"]

    def select_graph(self, name):
        return _Graph(name)


def _headers(body: bytes) -> dict:
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    signature = build_signature(
        secret=ADMIN_SECRET, timestamp=timestamp, nonce=nonce, method="POST",
        path="/admin/graph-stats", scope="admin", client_slug="pokagon", body=body,
    )
    return {
        "content-type": "application/json",
        "x-sdi-kg-timestamp": timestamp,
        "x-sdi-kg-scope": "admin",
        "x-sdi-kg-client": "pokagon",
        "x-sdi-kg-nonce": nonce,
        "x-sdi-kg-signature": signature,
    }


@pytest.fixture(autouse=True)
def _auth(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required")
    monkeypatch.setattr(settings, "graphiti_admin_secret", ADMIN_SECRET)
    monkeypatch.setattr(settings, "graphiti_auth_max_clock_skew_seconds", 300)
    monkeypatch.setattr(falkordb, "FalkorDB", _FalkorDB)

    async def consume(scope: str, nonce: str) -> bool:
        return True

    monkeypatch.setattr(auth, "_consume_nonce", consume)


def test_the_log_names_the_database_error_and_the_wire_does_not(monkeypatch, caplog):
    from app.services import graphiti_client

    monkeypatch.setattr(graphiti_client, "_falkor_db", None, raising=False)

    async def quiet(*_args, **_kwargs):
        return None

    monkeypatch.setattr(graphiti_client, "_log_slow_queries", quiet)
    monkeypatch.setattr(graphiti_client, "_log_memory_usage", quiet)

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin", dependencies=[Depends(require_scope("admin"))])
    body = json.dumps({"client_slug": "pokagon", "include_provenance": True}, separators=(",", ":")).encode()

    with caplog.at_level(logging.ERROR, logger="graphiti_service"):
        response = TestClient(app).post("/admin/graph-stats", content=body, headers=_headers(body))

    assert response.status_code == 502
    assert response.json() == {"detail": "graph-stats failed"}
    assert DB_MESSAGE not in response.text
    failures = [r.getMessage() for r in caplog.records if "graph-stats failed" in r.getMessage()]
    assert failures, "the failure was not logged"
    assert "error_type=ResponseError" in failures[0]
    assert DB_MESSAGE in failures[0]
