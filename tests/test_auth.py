"""Contract tests for the Graphiti service authentication perimeter."""

import json
import time
import uuid

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app import auth
from app.auth import build_signature, require_scope, validate_auth_configuration
from app.config import settings

SEARCH_SECRET = "search-secret-that-is-at-least-32-characters"
INGEST_SECRET = "ingest-secret-that-is-at-least-32-characters"
ADMIN_SECRET = "admin-secret-that-is-at-least-32-characters"
SIDE_EFFECTS = {"ingest": 0, "admin": 0, "graph": 0}


class TenantRequest(BaseModel):
    client_slug: str
    value: str = "ok"


app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search/context", dependencies=[Depends(require_scope("search"))])
async def search_echo(req: TenantRequest):
    return req.model_dump()


@app.post("/ingest/episode", dependencies=[Depends(require_scope("ingest"))])
async def ingest_echo(req: TenantRequest):
    SIDE_EFFECTS["ingest"] += 1
    return req.model_dump()


@app.post("/admin/save", dependencies=[Depends(require_scope("admin"))])
async def admin_echo():
    SIDE_EFFECTS["admin"] += 1
    return {"status": "saved"}


@app.post("/graph/data", dependencies=[Depends(require_scope("search"))])
async def graph_echo(req: TenantRequest):
    SIDE_EFFECTS["graph"] += 1
    return req.model_dump()


client = TestClient(app)


@pytest.fixture(autouse=True)
def required_auth(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required")
    monkeypatch.setattr(settings, "graphiti_search_secret", SEARCH_SECRET)
    monkeypatch.setattr(settings, "graphiti_ingest_secret", INGEST_SECRET)
    monkeypatch.setattr(settings, "graphiti_admin_secret", ADMIN_SECRET)
    monkeypatch.setattr(settings, "graphiti_auth_max_clock_skew_seconds", 300)
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", False)
    for key in SIDE_EFFECTS:
        SIDE_EFFECTS[key] = 0
    seen: set[tuple[str, str]] = set()

    async def consume(scope: str, nonce: str) -> bool:
        key = (scope, nonce)
        if key in seen:
            return False
        seen.add(key)
        return True

    monkeypatch.setattr(auth, "_consume_nonce", consume)


def encoded(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def headers(
    *,
    path: str,
    scope: str,
    client_slug: str,
    body: bytes,
    secret: str,
    timestamp: str | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    timestamp = timestamp or str(int(time.time()))
    nonce = nonce or uuid.uuid4().hex
    signature = build_signature(
        secret=secret,
        timestamp=timestamp,
        nonce=nonce,
        method="POST",
        path=path,
        scope=scope,
        client_slug=client_slug,
        body=body,
    )
    return {
        "content-type": "application/json",
        "x-sdi-kg-timestamp": timestamp,
        "x-sdi-kg-scope": scope,
        "x-sdi-kg-client": client_slug,
        "x-sdi-kg-nonce": nonce,
        "x-sdi-kg-signature": signature,
    }


def test_health_is_public():
    assert client.get("/health").status_code == 200


def test_cross_language_signature_vector_is_stable():
    body = b'{"client_slug":"pokagon","value":"preserved"}'
    assert build_signature(
        secret=SEARCH_SECRET,
        timestamp="1750000000",
        nonce="00112233445566778899aabbccddeeff",
        method="POST",
        path="/search/context",
        scope="search",
        client_slug="pokagon",
        body=body,
    ) == "bf3424f70a5827bacb4e3ba526541317dfbd549540628c98f70e46f4f523016d"


def test_required_mode_rejects_unsigned_request():
    response = client.post("/search/context", json={"client_slug": "pokagon"})
    assert response.status_code == 401


def test_valid_signature_reaches_route_and_preserves_body():
    path = "/search/context"
    body = encoded({"client_slug": "pokagon", "value": "preserved"})
    response = client.post(
        path,
        content=body,
        headers=headers(
            path=path,
            scope="search",
            client_slug="pokagon",
            body=body,
            secret=SEARCH_SECRET,
        ),
    )
    assert response.status_code == 200
    assert response.json() == {"client_slug": "pokagon", "value": "preserved"}


def test_probe_process_blocks_ingest_admin_and_graph_before_side_effects(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", True)
    requests = (
        (
            "/ingest/episode",
            "ingest",
            "pokagon",
            INGEST_SECRET,
            encoded({"client_slug": "pokagon"}),
        ),
        ("/admin/save", "admin", "*", ADMIN_SECRET, b""),
        (
            "/graph/data",
            "search",
            "pokagon",
            SEARCH_SECRET,
            encoded({"client_slug": "pokagon"}),
        ),
    )

    for path, scope, client_slug, secret, body in requests:
        response = client.post(
            path,
            content=body,
            headers=headers(
                path=path,
                scope=scope,
                client_slug=client_slug,
                body=body,
                secret=secret,
            ),
        )
        assert response.status_code == 403
        assert response.json() == {
            "detail": "Acceptance probe process permits only the search probe endpoint"
        }

    assert SIDE_EFFECTS == {"ingest": 0, "admin": 0, "graph": 0}
    assert client.get("/health").status_code == 200


def test_scope_cannot_be_reused_for_ingest():
    path = "/ingest/episode"
    body = encoded({"client_slug": "pokagon"})
    response = client.post(
        path,
        content=body,
        headers=headers(
            path=path,
            scope="search",
            client_slug="pokagon",
            body=body,
            secret=SEARCH_SECRET,
        ),
    )
    assert response.status_code == 403


def test_signed_tenant_cannot_be_retargeted_in_body():
    path = "/search/context"
    body = encoded({"client_slug": "mrla"})
    response = client.post(
        path,
        content=body,
        headers=headers(
            path=path,
            scope="search",
            client_slug="pokagon",
            body=body,
            secret=SEARCH_SECRET,
        ),
    )
    assert response.status_code == 403


def test_signature_is_bound_to_exact_path_and_body():
    original_path = "/search/context"
    body = encoded({"client_slug": "pokagon"})
    signed = headers(
        path=original_path,
        scope="search",
        client_slug="pokagon",
        body=body,
        secret=SEARCH_SECRET,
    )
    tampered = encoded({"client_slug": "pokagon", "value": "changed"})
    response = client.post(original_path, content=tampered, headers=signed)
    assert response.status_code == 401


def test_identical_signed_request_is_rejected_as_a_replay():
    path = "/ingest/episode"
    body = encoded({"client_slug": "pokagon"})
    signed = headers(
        path=path,
        scope="ingest",
        client_slug="pokagon",
        body=body,
        secret=INGEST_SECRET,
        nonce="00112233445566778899aabbccddeeff",
    )
    assert client.post(path, content=body, headers=signed).status_code == 200
    replay = client.post(path, content=body, headers=signed)
    assert replay.status_code == 409
    assert replay.json()["detail"] == "Replayed graph service request"


def test_replay_store_outage_fails_closed(monkeypatch):
    async def unavailable(_scope: str, _nonce: str) -> bool:
        raise ConnectionError("replay store unavailable")

    monkeypatch.setattr(auth, "_consume_nonce", unavailable)
    path = "/ingest/episode"
    body = encoded({"client_slug": "pokagon"})
    response = client.post(
        path,
        content=body,
        headers=headers(
            path=path,
            scope="ingest",
            client_slug="pokagon",
            body=body,
            secret=INGEST_SECRET,
        ),
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Graph service replay protection unavailable"


def test_expired_signature_is_rejected():
    path = "/search/context"
    body = encoded({"client_slug": "pokagon"})
    old = str(int(time.time()) - 301)
    response = client.post(
        path,
        content=body,
        headers=headers(
            path=path,
            scope="search",
            client_slug="pokagon",
            body=body,
            secret=SEARCH_SECRET,
            timestamp=old,
        ),
    )
    assert response.status_code == 401


def test_admin_without_tenant_requires_platform_claim():
    path = "/admin/save"
    body = b""
    tenant_response = client.post(
        path,
        content=body,
        headers=headers(
            path=path,
            scope="admin",
            client_slug="pokagon",
            body=body,
            secret=ADMIN_SECRET,
        ),
    )
    assert tenant_response.status_code == 403

    platform_response = client.post(
        path,
        content=body,
        headers=headers(
            path=path,
            scope="admin",
            client_slug="*",
            body=body,
            secret=ADMIN_SECRET,
        ),
    )
    assert platform_response.status_code == 200


def test_optional_mode_allows_unsigned_rollout_traffic_but_rejects_bad_signatures(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "optional")
    body = encoded({"client_slug": "pokagon"})
    assert client.post("/search/context", content=body, headers={"content-type": "application/json"}).status_code == 200

    bad = headers(
        path="/search/context",
        scope="search",
        client_slug="pokagon",
        body=body,
        secret="wrong-secret-that-is-still-at-least-32-characters",
    )
    assert client.post("/search/context", content=body, headers=bad).status_code == 401


def test_off_mode_preserves_pre_rollout_contract(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "off")
    response = client.post("/ingest/episode", json={"client_slug": "pokagon"})
    assert response.status_code == 200


def test_required_configuration_rejects_reused_and_placeholder_credentials(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required")
    monkeypatch.setattr(settings, "graphiti_search_secret", "replace-with-public-placeholder-secret")
    with pytest.raises(RuntimeError, match="weak/placeholders"):
        validate_auth_configuration()

    monkeypatch.setattr(settings, "graphiti_search_secret", SEARCH_SECRET)
    monkeypatch.setattr(settings, "graphiti_ingest_secret", SEARCH_SECRET)
    with pytest.raises(RuntimeError, match="must be distinct"):
        validate_auth_configuration()


def test_optional_configuration_rejects_any_configured_placeholder(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "optional")
    monkeypatch.setattr(settings, "graphiti_search_secret", "replace-before-use-search-secret-with-random-value")
    monkeypatch.setattr(settings, "graphiti_ingest_secret", "")
    monkeypatch.setattr(settings, "graphiti_admin_secret", "")
    with pytest.raises(RuntimeError, match="weak/placeholders"):
        validate_auth_configuration()
