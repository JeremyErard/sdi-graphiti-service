"""HMAC authentication for Engage -> Graphiti service requests.

Every protected request is signed over its exact body plus the HTTP method,
path, scope, tenant claim, and a short-lived timestamp. This prevents a valid
search request from being replayed as an ingest/admin request or retargeted to
another client by changing ``client_slug``.

Rollout is deliberately staged through GRAPHITI_AUTH_MODE:

* off: preserve the legacy contract (development / pre-rollout only)
* optional: accept unsigned legacy requests, verify signed requests
* required: fail closed for every non-health route
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, NoReturn

from fastapi import HTTPException, Request

from app.config import settings

AuthScope = Literal["search", "ingest", "admin"]

AUTH_TIMESTAMP_HEADER = "x-sdi-kg-timestamp"
AUTH_SCOPE_HEADER = "x-sdi-kg-scope"
AUTH_CLIENT_HEADER = "x-sdi-kg-client"
AUTH_NONCE_HEADER = "x-sdi-kg-nonce"
AUTH_SIGNATURE_HEADER = "x-sdi-kg-signature"

logger = logging.getLogger("graphiti_service.auth")


@dataclass(frozen=True)
class GraphPrincipal:
    scope: AuthScope
    client_slug: str


def _deny(
    request: Request,
    expected_scope: AuthScope,
    status_code: int,
    detail: str,
) -> NoReturn:
    peer = request.client.host if request.client else "unknown"
    logger.warning(
        "[graphiti-auth] denied status=%s scope=%s client=%s path=%s peer=%s reason=%s",
        status_code,
        expected_scope,
        request.headers.get(AUTH_CLIENT_HEADER, "missing"),
        request.url.path,
        peer,
        detail,
    )
    raise HTTPException(status_code=status_code, detail=detail)


def _secret_for_scope(scope: AuthScope) -> str:
    return {
        "search": settings.graphiti_search_secret,
        "ingest": settings.graphiti_ingest_secret,
        "admin": settings.graphiti_admin_secret,
    }[scope]


def _canonical_message(
    *,
    timestamp: str,
    nonce: str,
    method: str,
    path: str,
    scope: AuthScope,
    client_slug: str,
    body: bytes,
) -> bytes:
    body_hash = hashlib.sha256(body).hexdigest()
    return (
        f"v2\n{timestamp}\n{nonce}\n{method.upper()}\n{path}\n{scope}\n"
        f"{client_slug}\n{body_hash}"
    ).encode("utf-8")


def build_signature(
    *,
    secret: str,
    timestamp: str,
    nonce: str,
    method: str,
    path: str,
    scope: AuthScope,
    client_slug: str,
    body: bytes,
) -> str:
    """Build the v2 request signature. Exported for contract tests/CLI tools."""
    return hmac.new(
        secret.encode("utf-8"),
        _canonical_message(
            timestamp=timestamp,
            nonce=nonce,
            method=method,
            path=path,
            scope=scope,
            client_slug=client_slug,
            body=body,
        ),
        hashlib.sha256,
    ).hexdigest()


def validate_auth_configuration() -> None:
    """Refuse to start with missing, public-placeholder, or reused secrets."""
    if settings.graphiti_auth_mode == "off":
        return
    secrets = {scope: _secret_for_scope(scope) for scope in ("search", "ingest", "admin")}
    if settings.graphiti_auth_mode == "required":
        missing = [scope for scope, secret in secrets.items() if not secret]
    else:
        missing = []
    if missing:
        raise RuntimeError(
            "GRAPHITI_AUTH_MODE=required needs secrets for: "
            + ", ".join(missing)
        )
    configured = {scope: secret for scope, secret in secrets.items() if secret}
    weak = [
        scope
        for scope, secret in configured.items()
        if len(secret) < 32
        or any(marker in secret.lower() for marker in ("replace-with", "replace-before", "changeme", "example-secret"))
    ]
    if weak:
        raise RuntimeError("Graphiti auth credentials are weak/placeholders for: " + ", ".join(weak))
    if len(set(configured.values())) != len(configured):
        raise RuntimeError("Graphiti search, ingest, and admin credentials must be distinct")


_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


async def _consume_nonce(scope: AuthScope, nonce: str) -> bool:
    """Atomically reserve a signed request nonce in FalkorDB's Redis layer."""
    import redis.asyncio as redis

    nonce_digest = hashlib.sha256(f"{scope}\0{nonce}".encode("utf-8")).hexdigest()
    key = f"sdi:graphiti:auth:nonce:{nonce_digest}"
    ttl = max(60, settings.graphiti_auth_max_clock_skew_seconds * 2)
    client = redis.Redis(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    try:
        return bool(await client.set(key, "1", ex=ttl, nx=True))
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


async def verify_request(request: Request, expected_scope: AuthScope) -> GraphPrincipal:
    mode = settings.graphiti_auth_mode
    if mode == "off":
        return GraphPrincipal(scope=expected_scope, client_slug="legacy")

    timestamp = request.headers.get(AUTH_TIMESTAMP_HEADER)
    claimed_scope = request.headers.get(AUTH_SCOPE_HEADER)
    claimed_client = request.headers.get(AUTH_CLIENT_HEADER)
    nonce = request.headers.get(AUTH_NONCE_HEADER)
    signature = request.headers.get(AUTH_SIGNATURE_HEADER)
    supplied = [timestamp, claimed_scope, claimed_client, nonce, signature]

    if not any(supplied):
        if mode == "optional":
            logger.warning(
                "[graphiti-auth] accepting unsigned legacy %s request in optional mode",
                expected_scope,
            )
            return GraphPrincipal(scope=expected_scope, client_slug="legacy")
        _deny(request, expected_scope, 401, "Graph service authentication required")

    if not all(supplied):
        _deny(request, expected_scope, 401, "Incomplete graph service authentication")
    if claimed_scope != expected_scope:
        _deny(request, expected_scope, 403, "Graph service scope denied")
    if not _NONCE_RE.fullmatch(nonce):
        _deny(request, expected_scope, 401, "Invalid graph service nonce")

    try:
        timestamp_number = int(timestamp)
    except (TypeError, ValueError):
        _deny(request, expected_scope, 401, "Invalid graph service timestamp")
    if abs(int(time.time()) - timestamp_number) > settings.graphiti_auth_max_clock_skew_seconds:
        _deny(request, expected_scope, 401, "Expired graph service request")

    body = await request.body()
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        _deny(request, expected_scope, 400, "Invalid JSON request body")

    body_client = payload.get("client_slug") if isinstance(payload, dict) else None
    if body_client is None:
        if claimed_client != "*":
            _deny(request, expected_scope, 403, "Platform-scoped request required")
    elif not isinstance(body_client, str) or body_client != claimed_client:
        _deny(request, expected_scope, 403, "Graph tenant claim does not match request")

    secret = _secret_for_scope(expected_scope)
    if len(secret) < 32:
        logger.error("[graphiti-auth] %s credential is not configured", expected_scope)
        _deny(request, expected_scope, 503, "Graph service authentication is not configured")

    expected = build_signature(
        secret=secret,
        timestamp=timestamp,
        nonce=nonce,
        method=request.method,
        path=request.url.path,
        scope=expected_scope,
        client_slug=claimed_client,
        body=body,
    )
    if not hmac.compare_digest(signature, expected):
        _deny(request, expected_scope, 401, "Invalid graph service signature")

    try:
        nonce_is_new = await _consume_nonce(expected_scope, nonce)
    except Exception:
        logger.exception("[graphiti-auth] replay-protection store unavailable")
        _deny(request, expected_scope, 503, "Graph service replay protection unavailable")
    if not nonce_is_new:
        _deny(request, expected_scope, 409, "Replayed graph service request")

    return GraphPrincipal(scope=expected_scope, client_slug=claimed_client)


def require_scope(
    expected_scope: AuthScope,
) -> Callable[[Request], Awaitable[GraphPrincipal]]:
    async def dependency(request: Request) -> GraphPrincipal:
        return await verify_request(request, expected_scope)

    return dependency
