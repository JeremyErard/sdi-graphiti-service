"""Shared HMAC headers for Graphiti operator scripts."""

import hashlib
import hmac
import os
import secrets
import time


def _scope_for_path(path: str) -> str:
    if path.startswith("/search/") or path.startswith("/graph/"):
        return "search"
    if path.startswith("/ingest/"):
        return "ingest"
    if path.startswith("/admin/"):
        return "admin"
    raise ValueError(f"No Graphiti auth scope is defined for path: {path}")


def signed_headers(
    path: str,
    body: bytes,
    client_slug: str,
    *,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Return content/auth headers using the same v2 contract as Engage."""
    headers = {"content-type": "application/json"}
    mode = os.getenv("GRAPHITI_AUTH_MODE", "off")
    if mode == "off":
        return headers

    scope = _scope_for_path(path)
    env_name = f"GRAPHITI_{scope.upper()}_SECRET"
    secret = os.getenv(env_name, "")
    if not secret:
        if mode == "optional":
            return headers
        raise SystemExit(f"{env_name} is required when GRAPHITI_AUTH_MODE=required")
    if len(secret) < 32:
        raise SystemExit(f"{env_name} must be at least 32 characters")

    timestamp_text = str(timestamp if timestamp is not None else int(time.time()))
    nonce_text = nonce or secrets.token_hex(16)
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = (
        f"v2\n{timestamp_text}\n{nonce_text}\nPOST\n{path}\n{scope}\n"
        f"{client_slug}\n{body_hash}"
    ).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
    headers.update(
        {
            "X-SDI-KG-Timestamp": timestamp_text,
            "X-SDI-KG-Scope": scope,
            "X-SDI-KG-Client": client_slug,
            "X-SDI-KG-Nonce": nonce_text,
            "X-SDI-KG-Signature": signature,
        }
    )
    return headers
