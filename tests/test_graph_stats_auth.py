"""Graph-stats must use the existing method/body-bound admin auth contract."""

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
from app.routers import admin, graph, ingest, search, structured
from app.services.provenance_stats import (
    PROVENANCE_STATS_EDGE_ROW_LIMIT_CODE,
    ProvenanceStatsReadError,
)
from scripts.graphiti_http import signed_headers as operator_signed_headers


ADMIN_SECRET = "admin-secret-that-is-at-least-32-characters"


class FakeQueryResult:
    def __init__(self, rows):
        self.result_set = rows


class FakeGraph:
    def __init__(self, name: str):
        self.name = name

    def query(self, query: str, params: dict | None = None):
        # The default node/edge totals stay on the shipped GRAPH.QUERY command.
        FakeFalkorDB.queries.append((self.name, query, params or {}))
        if "MATCH (n)" in query:
            return FakeQueryResult([[11]])
        if "MATCH ()-[r]->()" in query:
            return FakeQueryResult([[22]])
        raise AssertionError("graph-stats provenance reads must use ro_query")

    def ro_query(self, query: str, params: dict | None = None):
        FakeFalkorDB.queries.append((self.name, query, params or {}))
        if "MATCH (n) RETURN count(n)" in query or "MATCH ()-[r]->()" in query:
            raise AssertionError(
                "default graph-stats totals must stay on the shipped query command"
            )
        if "MATCH (episode:Episodic)" in query:
            assert params["group_id"] == self.name
            assert "disallowed_control_pattern" in params
            assert "nonblank_text_pattern" in params
            return FakeQueryResult([])
        if "edge:RELATES_TO" in query:
            assert params["group_id"] == self.name
            assert "disallowed_control_pattern" in params
            assert "nonblank_text_pattern" in params
            assert params["temporal_storage_limit"] == 128
            return FakeQueryResult([])
        raise AssertionError(f"unexpected graph-stats query: {query}")


class FakeFalkorDB:
    selected: list[str] = []
    queries: list[tuple[str, str, dict]] = []

    def __init__(self, **_kwargs):
        pass

    def list_graphs(self):
        return ["segment_tribal_gaming", "client_pokagon"]

    def select_graph(self, name: str):
        self.selected.append(name)
        return FakeGraph(name)


def encoded(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def admin_headers(*, body: bytes, client_slug: str, nonce: str | None = None):
    timestamp = str(int(time.time()))
    nonce = nonce or uuid.uuid4().hex
    path = "/admin/graph-stats"
    signature = build_signature(
        secret=ADMIN_SECRET,
        timestamp=timestamp,
        nonce=nonce,
        method="POST",
        path=path,
        scope="admin",
        client_slug=client_slug,
        body=body,
    )
    return {
        "content-type": "application/json",
        "x-sdi-kg-timestamp": timestamp,
        "x-sdi-kg-scope": "admin",
        "x-sdi-kg-client": client_slug,
        "x-sdi-kg-nonce": nonce,
        "x-sdi-kg-signature": signature,
    }


@pytest.fixture(autouse=True)
def required_auth(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required")
    monkeypatch.setattr(settings, "graphiti_admin_secret", ADMIN_SECRET)
    monkeypatch.setattr(settings, "graphiti_auth_max_clock_skew_seconds", 300)
    monkeypatch.setattr(falkordb, "FalkorDB", FakeFalkorDB)
    FakeFalkorDB.selected = []
    FakeFalkorDB.queries = []
    seen: set[tuple[str, str]] = set()

    async def consume(scope: str, nonce: str) -> bool:
        key = (scope, nonce)
        if key in seen:
            return False
        seen.add(key)
        return True

    monkeypatch.setattr(auth, "_consume_nonce", consume)


def client() -> TestClient:
    app = FastAPI()
    app.include_router(
        admin.router,
        prefix="/admin",
        dependencies=[Depends(require_scope("admin"))],
    )
    return TestClient(app)


def test_graph_stats_is_post_only_and_rejects_unsigned_access():
    test_client = client()
    assert test_client.get("/admin/graph-stats").status_code == 405
    assert test_client.post("/admin/graph-stats", json={}).status_code == 401


def test_every_protected_business_route_uses_the_supported_post_contract():
    offenders: list[str] = []
    for prefix, router in (
        ("/admin", admin.router),
        ("/graph", graph.router),
        ("/ingest", ingest.router),
        ("/ingest", structured.router),
        ("/search", search.router),
    ):
        for route in router.routes:
            methods = set(getattr(route, "methods", set()) or set())
            if methods != {"POST"}:
                offenders.append(
                    f"{prefix}{getattr(route, 'path', '?')}={sorted(methods)}"
                )
    assert offenders == [], (
        "Protected GET/other methods need method-aware signing in both clients "
        f"before they may be added: {offenders}"
    )


def test_platform_signed_body_counts_all_graphs_without_mutation():
    body = encoded({})
    response = client().post(
        "/admin/graph-stats",
        content=body,
        headers=admin_headers(body=body, client_slug="*"),
    )
    assert response.status_code == 200
    assert response.json() == {
        "graphs": [
            {"graph_name": "client_pokagon", "nodes": 11, "edges": 22},
            {"graph_name": "segment_tribal_gaming", "nodes": 11, "edges": 22},
        ],
        "graph_count": 2,
    }
    assert FakeFalkorDB.selected == ["client_pokagon", "segment_tribal_gaming"]
    assert not any(
        "Episodic" in query or "edge:RELATES_TO" in query
        for _name, query, _params in FakeFalkorDB.queries
    )


def test_provenance_aggregates_are_explicitly_opt_in_and_content_free():
    body = encoded({"client_slug": "pokagon", "include_provenance": True})

    response = client().post(
        "/admin/graph-stats",
        content=body,
        headers=admin_headers(body=body, client_slug="pokagon"),
    )

    assert response.status_code == 200
    assert response.json() == {
        "graphs": [
            {
                "graph_name": "client_pokagon",
                "nodes": 11,
                "edges": 22,
                "provenance": {
                    "facts_total": 0,
                    "malformed_response_events": 0,
                    "by_structural_status": [
                        {"structural_status": "chained", "count": 0},
                        {"structural_status": "pre_chain", "count": 0},
                        {"structural_status": "malformed", "count": 0},
                    ],
                    "by_episode_type": [],
                    "by_engagement": [],
                },
            }
        ],
        "graph_count": 1,
    }
    provenance_queries = [
        query
        for _name, query, _params in FakeFalkorDB.queries
        if "Episodic" in query or "edge:RELATES_TO" in query
    ]
    assert len(provenance_queries) == 2
    serialized = response.text.lower()
    for forbidden in ("fact", "source_description", "episode_name", "content"):
        # Structural field names such as facts_total are allowed; graph values are not.
        if forbidden == "fact":
            continue
        assert forbidden not in serialized


def test_provenance_opt_in_requires_one_exact_client_before_graph_access():
    body = encoded({"include_provenance": True})
    response = client().post(
        "/admin/graph-stats",
        content=body,
        headers=admin_headers(body=body, client_slug="*"),
    )

    assert response.status_code == 422
    assert FakeFalkorDB.selected == []


def test_missing_exact_graph_fails_with_fixed_code_without_selection():
    body = encoded({"client_slug": "missing"})
    response = client().post(
        "/admin/graph-stats",
        content=body,
        headers=admin_headers(body=body, client_slug="missing"),
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "GRAPH_STATS_GRAPH_NOT_FOUND"}
    assert FakeFalkorDB.selected == []


def test_provenance_bound_failure_returns_only_fixed_safe_code(monkeypatch):
    def fail_stats(*_args, **_kwargs):
        raise ProvenanceStatsReadError(PROVENANCE_STATS_EDGE_ROW_LIMIT_CODE)

    monkeypatch.setattr(admin, "provenance_stats_for_graph", fail_stats)
    body = encoded({"client_slug": "pokagon", "include_provenance": True})
    response = client().post(
        "/admin/graph-stats",
        content=body,
        headers=admin_headers(body=body, client_slug="pokagon"),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": PROVENANCE_STATS_EDGE_ROW_LIMIT_CODE}


def test_supported_operator_helper_signs_graph_stats_post(monkeypatch):
    monkeypatch.setenv("GRAPHITI_AUTH_MODE", "required")
    monkeypatch.setenv("GRAPHITI_ADMIN_SECRET", ADMIN_SECRET)
    body = encoded({})
    response = client().post(
        "/admin/graph-stats",
        content=body,
        headers=operator_signed_headers(
            "/admin/graph-stats",
            body,
            "*",
            nonce="00112233445566778899aabbccddeeff",
        ),
    )
    assert response.status_code == 200
    assert response.json()["graph_count"] == 2


def test_tenant_signed_body_scopes_the_count_and_ignores_query_tampering():
    body = encoded({"client_slug": "pokagon"})
    response = client().post(
        "/admin/graph-stats?client_slug=test-provision",
        content=body,
        headers=admin_headers(body=body, client_slug="pokagon"),
    )
    assert response.status_code == 200
    assert response.json() == {
        "graphs": [{"graph_name": "client_pokagon", "nodes": 11, "edges": 22}],
        "graph_count": 1,
    }
    assert FakeFalkorDB.selected == ["client_pokagon"]


def test_header_tenant_cannot_differ_from_signed_body_tenant():
    body = encoded({"client_slug": "test-provision"})
    response = client().post(
        "/admin/graph-stats",
        content=body,
        headers=admin_headers(body=body, client_slug="pokagon"),
    )
    assert response.status_code == 403
    assert FakeFalkorDB.selected == []


def test_body_tampering_and_replay_are_rejected():
    original = encoded({"client_slug": "pokagon"})
    signed = admin_headers(
        body=original,
        client_slug="pokagon",
        nonce="00112233445566778899aabbccddeeff",
    )
    tampered = encoded({"client_slug": "pokagon", "ignored": "tampered"})
    assert client().post(
        "/admin/graph-stats", content=tampered, headers=signed
    ).status_code == 401

    test_client = client()
    first = test_client.post("/admin/graph-stats", content=original, headers=signed)
    replay = test_client.post("/admin/graph-stats", content=original, headers=signed)
    assert first.status_code == 200
    assert replay.status_code == 409


def test_invalid_client_slug_is_rejected_before_graph_access():
    body = encoded({"client_slug": "../pokagon"})
    response = client().post(
        "/admin/graph-stats",
        content=body,
        headers=admin_headers(body=body, client_slug="../pokagon"),
    )
    assert response.status_code == 422
    assert FakeFalkorDB.selected == []
