#!/usr/bin/env python3
"""Read-only, manifest-pinned Graphiti P1 probe harness.

The only network operation this module can issue is a signed POST to the fixed
``/search/context`` path. It has no ingest, admin, model, fallback-control, or
tenant-discovery capability.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any, Callable, Literal, Mapping
import urllib.error
import urllib.parse
import urllib.request
import uuid as uuidlib

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


SEARCH_PATH = "/search/context"
MANIFEST_CONTRACT_VERSION = "graphiti_p1_probe_manifest_v1"
RESPONSE_CONTRACT_VERSION = "graphiti_search_context_v3"
MAX_RESPONSE_BYTES = 2_000_000


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(RejectRedirects())


def _open_without_redirect(request: urllib.request.Request, *, timeout: int):
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


class ProbeFailure(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _bounded_text(value: str, label: str, maximum: int) -> str:
    if (
        not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


class ExpectedIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: uuidlib.UUID
    episode_uuid: uuidlib.UUID
    source_type: str
    source_id: str

    @field_validator("source_type", "source_id")
    @classmethod
    def validate_source_identity(cls, value: str) -> str:
        return _bounded_text(value, "source identity", 240)


class PinnedProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    query: str
    max_results: int = Field(ge=1, le=50)
    minimum_results: int = Field(ge=1, le=50)
    expected: list[ExpectedIdentity] = Field(min_length=1, max_length=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return _bounded_text(value, "query", 2_000)

    @model_validator(mode="after")
    def validate_non_vacuity(self):
        if self.minimum_results > self.max_results:
            raise ValueError("minimum_results cannot exceed max_results")
        distinct_fact_ids = {str(item.fact_id) for item in self.expected}
        if len(distinct_fact_ids) > self.max_results:
            raise ValueError("expected fact identities exceed max_results")
        identities = [
            (str(item.fact_id), str(item.episode_uuid), item.source_type, item.source_id)
            for item in self.expected
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("expected identities must be unique")
        return self


class ProbeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[MANIFEST_CONTRACT_VERSION]
    service_url: str
    client_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,127}$")
    engagement_id: str
    auth_scope: Literal["search"]
    auth_secret_env: Literal["GRAPHITI_SEARCH_SECRET"]
    probes: list[PinnedProbe] = Field(min_length=1, max_length=20)

    @field_validator("service_url")
    @classmethod
    def validate_service_url(cls, value: str) -> str:
        value = _bounded_text(value, "service_url", 2_048)
        parsed = urllib.parse.urlsplit(value)
        try:
            parsed.port
        except ValueError:
            raise ValueError("service_url port is invalid") from None
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or value.endswith("/")
        ):
            raise ValueError("service_url must be an exact origin without a trailing slash")
        return value

    @field_validator("engagement_id")
    @classmethod
    def validate_engagement_id(cls, value: str) -> str:
        return _bounded_text(value, "engagement_id", 240)

    @model_validator(mode="after")
    def validate_probe_ids(self):
        probe_ids = [probe.probe_id for probe in self.probes]
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("probe_id values must be unique")
        return self


def load_manifest(path: str | Path) -> ProbeManifest:
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 1_000_000:
            raise ProbeFailure("MANIFEST_TOO_LARGE")
        return ProbeManifest.model_validate_json(raw)
    except ProbeFailure:
        raise
    except (OSError, ValidationError, ValueError, UnicodeError):
        raise ProbeFailure("MANIFEST_INVALID") from None


def _signed_headers(
    *,
    secret: str,
    body: bytes,
    client_slug: str,
    timestamp: int,
    nonce: str,
) -> dict[str, str]:
    timestamp_text = str(timestamp)
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = (
        f"v2\n{timestamp_text}\n{nonce}\nPOST\n{SEARCH_PATH}\nsearch\n"
        f"{client_slug}\n{body_hash}"
    ).encode("utf-8")
    signature = hmac.new(
        secret.encode("utf-8"),
        canonical,
        hashlib.sha256,
    ).hexdigest()
    return {
        "content-type": "application/json",
        "X-SDI-KG-Timestamp": timestamp_text,
        "X-SDI-KG-Scope": "search",
        "X-SDI-KG-Client": client_slug,
        "X-SDI-KG-Nonce": nonce,
        "X-SDI-KG-Signature": signature,
    }


def _read_response(response: Any) -> dict[str, Any]:
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ProbeFailure("RESPONSE_TOO_LARGE")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ProbeFailure("RESPONSE_INVALID_JSON") from None
    if not isinstance(decoded, dict):
        raise ProbeFailure("RESPONSE_INVALID_SHAPE")
    return decoded


def _canonical_uuid_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        canonical = str(uuidlib.UUID(value))
    except (ValueError, TypeError, AttributeError):
        return None
    return canonical if canonical == value else None


def _nonempty_wire_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_probe_response(
    manifest: ProbeManifest,
    probe: PinnedProbe,
    response: dict[str, Any],
) -> None:
    if response.get("contract_version") != RESPONSE_CONTRACT_VERSION:
        raise ProbeFailure("RESPONSE_CONTRACT_MISMATCH")
    if response.get("segment_insights") != []:
        raise ProbeFailure("SEGMENT_CHANNEL_NONEMPTY")
    summary = response.get("provenance_summary")
    if (
        not isinstance(summary, dict)
        or summary.get("contract_version") != "graphiti_provenance_summary_v1"
    ):
        raise ProbeFailure("SUMMARY_CONTRACT_MISMATCH")
    facts = response.get("facts")
    if not isinstance(facts, list):
        raise ProbeFailure("FACTS_INVALID_SHAPE")
    if len(facts) < probe.minimum_results:
        raise ProbeFailure("PROBE_VACUOUS")
    if len(facts) > probe.max_results:
        raise ProbeFailure("FACTS_EXCEED_PINNED_LIMIT")

    facts_by_id: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if not isinstance(fact, dict):
            raise ProbeFailure("FACT_INVALID_SHAPE")
        if fact.get("chain_status") != "chained":
            raise ProbeFailure("FACT_NOT_CHAINED")
        fact_id = _canonical_uuid_text(fact.get("fact_id"))
        if fact_id is None or fact_id in facts_by_id:
            raise ProbeFailure("FACT_ID_INVALID")
        if _canonical_uuid_text(fact.get("subject")) is None or _canonical_uuid_text(
            fact.get("object")
        ) is None:
            raise ProbeFailure("FACT_ENDPOINT_ID_INVALID")
        if not all(
            _nonempty_wire_text(fact.get(field))
            for field in ("subject_name", "predicate", "object_name", "fact")
        ):
            raise ProbeFailure("FACT_TEXT_SHAPE_INVALID")
        episodes = fact.get("episodes")
        if (
            not isinstance(episodes, list)
            or not episodes
            or any(_canonical_uuid_text(item) is None for item in episodes)
        ):
            raise ProbeFailure("FACT_EPISODES_INVALID")
        sources = fact.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ProbeFailure("FACT_SOURCES_EMPTY")
        if any(
            not isinstance(source, dict)
            or source.get("engagement_id") != manifest.engagement_id
            or _canonical_uuid_text(source.get("episode_uuid")) is None
            or source.get("episode_uuid") not in episodes
            or not all(
                _nonempty_wire_text(source.get(field))
                for field in (
                    "episode_name",
                    "source_description",
                    "source_type",
                    "source_id",
                    "episode_type",
                    "anchor_mode",
                    "producer_contract_version",
                )
            )
            for source in sources
        ):
            raise ProbeFailure("FACT_SOURCE_ENGAGEMENT_MISMATCH")
        facts_by_id[fact_id] = fact

    for expected in probe.expected:
        fact = facts_by_id.get(str(expected.fact_id))
        if fact is None:
            raise ProbeFailure("EXPECTED_FACT_MISSING")
        episode_id = str(expected.episode_uuid)
        episodes = fact.get("episodes")
        if not isinstance(episodes, list) or episode_id not in episodes:
            raise ProbeFailure("EXPECTED_EPISODE_MISSING")
        sources = fact["sources"]
        if not any(
            source.get("episode_uuid") == episode_id
            and source.get("source_type") == expected.source_type
            and source.get("source_id") == expected.source_id
            for source in sources
        ):
            raise ProbeFailure("EXPECTED_SOURCE_MISSING")


def run_probe_manifest(
    manifest: ProbeManifest,
    *,
    service_url: str,
    auth_secret_env: str,
    environ: Mapping[str, str] | None = None,
    opener: Callable[..., Any] = _open_without_redirect,
    timestamp_factory: Callable[[], int] = lambda: int(time.time()),
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(16),
) -> dict[str, Any]:
    """Run pinned read-only searches and return counts/codes only."""

    if service_url != manifest.service_url:
        raise ProbeFailure("SERVICE_URL_MISMATCH")
    if auth_secret_env != manifest.auth_secret_env:
        raise ProbeFailure("AUTH_INPUT_MISMATCH")
    environment = os.environ if environ is None else environ
    if environment.get("GRAPHITI_AUTH_MODE") != "required":
        raise ProbeFailure("AUTH_MODE_NOT_REQUIRED")
    secret = environment.get(auth_secret_env, "")
    if len(secret) < 32:
        raise ProbeFailure("AUTH_SECRET_MISSING")

    passed = 0
    expected_total = 0
    for probe in manifest.probes:
        body = json.dumps(
            {
                "client_slug": manifest.client_slug,
                "engagement_id": manifest.engagement_id,
                "query": probe.query,
                "max_results": probe.max_results,
                "include_segment": False,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            manifest.service_url + SEARCH_PATH,
            data=body,
            headers=_signed_headers(
                secret=secret,
                body=body,
                client_slug=manifest.client_slug,
                timestamp=timestamp_factory(),
                nonce=nonce_factory(),
            ),
            method="POST",
        )
        try:
            with opener(request, timeout=60) as raw_response:
                response = _read_response(raw_response)
        except ProbeFailure:
            raise
        except urllib.error.HTTPError as error:
            code = (
                "HTTP_REDIRECT_REJECTED"
                if 300 <= error.code < 400
                else f"HTTP_STATUS_{error.code}"
            )
            raise ProbeFailure(code) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ProbeFailure("HTTP_TRANSPORT_FAILED") from None
        _validate_probe_response(manifest, probe, response)
        passed += 1
        expected_total += len(probe.expected)

    return {
        "counts": {
            "probes_total": len(manifest.probes),
            "probes_passed": passed,
            "expected_identities": expected_total,
        },
        "codes": {"PROBE_PASS": passed},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pinned read-only P1 probes",
        allow_abbrev=False,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--auth-secret-env", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        result = run_probe_manifest(
            manifest,
            service_url=args.service_url,
            auth_secret_env=args.auth_secret_env,
        )
    except ProbeFailure as error:
        print(json.dumps({"counts": {}, "codes": {error.code: 1}}, sort_keys=True))
        return 2
    except Exception as error:
        code = f"PROBE_FAILED_{type(error).__name__.upper()}"
        print(json.dumps({"counts": {}, "codes": {code: 1}}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
