"""Pinned read-only probe harness contract tests."""

from copy import deepcopy
import json
from pathlib import Path
import urllib.error

from pydantic import ValidationError
import pytest

from app.auth import build_signature
from scripts import provenance_probe


FACT_ID = "80000000-0000-4000-8000-000000000001"
EPISODE_ID = "81000000-0000-4000-8000-000000000001"
SUBJECT_ID = "82000000-0000-4000-8000-000000000001"
OBJECT_ID = "82000000-0000-4000-8000-000000000002"
SECRET = "search-secret-that-is-at-least-32-characters"


def _manifest_payload():
    return {
        "contract_version": "graphiti_p1_probe_manifest_v1",
        "service_url": "https://graphiti.example.invalid",
        "client_slug": "pokagon",
        "engagement_id": "engagement-123",
        "auth_scope": "search",
        "auth_secret_env": "GRAPHITI_SEARCH_SECRET",
        "probes": [
            {
                "probe_id": "document_owner",
                "query": "Who owns monthly close?",
                "max_results": 5,
                "minimum_results": 1,
                "expected": [
                    {
                        "fact_id": FACT_ID,
                        "episode_uuid": EPISODE_ID,
                        "source_type": "document",
                        "source_id": "doc-456",
                    }
                ],
            }
        ],
    }


def _response(*, facts=None):
    if facts is None:
        facts = [
            {
                "fact_id": FACT_ID,
                "subject": SUBJECT_ID,
                "subject_name": "Finance Team",
                "predicate": "owns",
                "object": OBJECT_ID,
                "object_name": "Monthly Close",
                "fact": "SECRET_FACT_TEXT",
                "episodes": [EPISODE_ID],
                "sources": [
                    {
                        "episode_uuid": EPISODE_ID,
                        "episode_name": "SECRET_EPISODE_NAME",
                        "source_description": "SECRET_SOURCE_DESCRIPTION",
                        "source_type": "document",
                        "source_id": "doc-456",
                        "engagement_id": "engagement-123",
                        "episode_type": "document_analysis",
                        "anchor_mode": "typed_source",
                        "producer_contract_version": "structured_provenance_v2",
                    }
                ],
                "chain_status": "chained",
            }
        ]
    return {
        "contract_version": "graphiti_search_context_v3",
        "facts": facts,
        "segment_insights": [],
        "provenance_summary": {
            "contract_version": "graphiti_provenance_summary_v1",
            "retrieval_path": "fast",
        },
    }


class _HTTPResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        return self.payload[:limit]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["probes"][0].update(query=""),
        lambda payload: payload["probes"][0].update(minimum_results=0),
        lambda payload: payload["probes"][0].update(expected=[]),
        lambda payload: payload["probes"][0].update(force_fallback=True),
        lambda payload: payload.update(service_url="https://user@example.invalid"),
        lambda payload: payload.update(service_url="http://graphiti.example.invalid"),
    ],
)
def test_manifest_rejects_vacuous_or_unratified_inputs(mutate):
    payload = _manifest_payload()
    mutate(payload)

    with pytest.raises(ValidationError):
        provenance_probe.ProbeManifest.model_validate(payload)


def test_manifest_file_is_bounded_and_strict(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    manifest = provenance_probe.load_manifest(manifest_path)

    assert manifest.contract_version == "graphiti_p1_probe_manifest_v1"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(provenance_probe.ProbeFailure) as failure:
        provenance_probe.load_manifest(invalid)
    assert failure.value.code == "MANIFEST_INVALID"


def test_probe_posts_only_signed_search_and_returns_counts_without_content():
    manifest = provenance_probe.ProbeManifest.model_validate(_manifest_payload())
    requests = []

    def opener(request, *, timeout):
        requests.append((request, timeout))
        return _HTTPResponse(_response())

    result = provenance_probe.run_probe_manifest(
        manifest,
        service_url="https://graphiti.example.invalid",
        auth_secret_env="GRAPHITI_SEARCH_SECRET",
        environ={
            "GRAPHITI_AUTH_MODE": "required",
            "GRAPHITI_SEARCH_SECRET": SECRET,
        },
        opener=opener,
        timestamp_factory=lambda: 1_750_000_000,
        nonce_factory=lambda: "00112233445566778899aabbccddeeff",
    )

    assert result == {
        "counts": {
            "probes_total": 1,
            "probes_passed": 1,
            "expected_identities": 1,
        },
        "codes": {"PROBE_PASS": 1},
    }
    serialized = json.dumps(result)
    for forbidden in (
        "SECRET_FACT_TEXT",
        "SECRET_EPISODE_NAME",
        "SECRET_SOURCE_DESCRIPTION",
        "Who owns monthly close?",
        FACT_ID,
        EPISODE_ID,
        "doc-456",
    ):
        assert forbidden not in serialized

    assert len(requests) == 1
    request, timeout = requests[0]
    assert timeout == 60
    assert request.method == "POST"
    assert request.full_url == "https://graphiti.example.invalid/search/context"
    body = json.loads(request.data)
    assert body == {
        "client_slug": "pokagon",
        "engagement_id": "engagement-123",
        "query": "Who owns monthly close?",
        "max_results": 5,
        "include_segment": False,
    }
    assert "fallback" not in json.dumps(body).lower()
    assert request.headers["X-sdi-kg-scope"] == "search"
    assert request.headers["X-sdi-kg-client"] == "pokagon"
    assert request.headers["X-sdi-kg-signature"] == build_signature(
        secret=SECRET,
        timestamp="1750000000",
        nonce="00112233445566778899aabbccddeeff",
        method="POST",
        path="/search/context",
        scope="search",
        client_slug="pokagon",
        body=request.data,
    )


@pytest.mark.parametrize(
    ("service_url", "auth_env", "environment", "code"),
    [
        (
            "https://other.example.invalid",
            "GRAPHITI_SEARCH_SECRET",
            {"GRAPHITI_AUTH_MODE": "required", "GRAPHITI_SEARCH_SECRET": SECRET},
            "SERVICE_URL_MISMATCH",
        ),
        (
            "https://graphiti.example.invalid",
            "OTHER_SECRET",
            {"GRAPHITI_AUTH_MODE": "required", "GRAPHITI_SEARCH_SECRET": SECRET},
            "AUTH_INPUT_MISMATCH",
        ),
        (
            "https://graphiti.example.invalid",
            "GRAPHITI_SEARCH_SECRET",
            {"GRAPHITI_AUTH_MODE": "optional", "GRAPHITI_SEARCH_SECRET": SECRET},
            "AUTH_MODE_NOT_REQUIRED",
        ),
        (
            "https://graphiti.example.invalid",
            "GRAPHITI_SEARCH_SECRET",
            {"GRAPHITI_AUTH_MODE": "required"},
            "AUTH_SECRET_MISSING",
        ),
    ],
)
def test_exact_url_and_auth_inputs_fail_before_network(
    service_url, auth_env, environment, code
):
    manifest = provenance_probe.ProbeManifest.model_validate(_manifest_payload())
    called = False

    def forbidden_opener(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(provenance_probe.ProbeFailure) as failure:
        provenance_probe.run_probe_manifest(
            manifest,
            service_url=service_url,
            auth_secret_env=auth_env,
            environ=environment,
            opener=forbidden_opener,
        )
    assert failure.value.code == code
    assert called is False


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (_response(facts=[]), "PROBE_VACUOUS"),
        (
            _response(
                facts=[
                    {
                        **_response()["facts"][0],
                        "fact_id": "80000000-0000-4000-8000-000000000099",
                    }
                ]
            ),
            "EXPECTED_FACT_MISSING",
        ),
    ],
)
def test_probe_non_vacuity_and_pinned_identity_fail_closed(payload, code):
    manifest = provenance_probe.ProbeManifest.model_validate(_manifest_payload())

    with pytest.raises(provenance_probe.ProbeFailure) as failure:
        provenance_probe.run_probe_manifest(
            manifest,
            service_url=manifest.service_url,
            auth_secret_env=manifest.auth_secret_env,
            environ={
                "GRAPHITI_AUTH_MODE": "required",
                "GRAPHITI_SEARCH_SECRET": SECRET,
            },
            opener=lambda *_args, **_kwargs: _HTTPResponse(payload),
        )
    assert failure.value.code == code


def test_probe_source_contains_no_write_or_fallback_endpoint_capability():
    source = Path(provenance_probe.__file__).read_text(encoding="utf-8")

    assert 'SEARCH_PATH = "/search/context"' in source
    assert '"/ingest/' not in source
    assert '"/admin/' not in source
    assert "force_fallback" not in source


def test_redirects_are_rejected_without_following_or_leaking_response_body():
    handler = provenance_probe.RejectRedirects()
    assert (
        handler.redirect_request(
            None,
            None,
            302,
            "Found",
            {},
            "https://other.example.invalid/search/context",
        )
        is None
    )
    manifest = provenance_probe.ProbeManifest.model_validate(_manifest_payload())
    calls = 0

    def redirecting_opener(request, *, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url,
            302,
            "SECRET_REDIRECT_BODY",
            {},
            None,
        )

    with pytest.raises(provenance_probe.ProbeFailure) as failure:
        provenance_probe.run_probe_manifest(
            manifest,
            service_url=manifest.service_url,
            auth_secret_env=manifest.auth_secret_env,
            environ={
                "GRAPHITI_AUTH_MODE": "required",
                "GRAPHITI_SEARCH_SECRET": SECRET,
            },
            opener=redirecting_opener,
        )
    assert failure.value.code == "HTTP_REDIRECT_REJECTED"
    assert "SECRET_REDIRECT_BODY" not in str(failure.value)
    assert calls == 1


def test_cli_requires_manifest_service_url_and_auth_input():
    parser = provenance_probe.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args(
        [
            "--manifest",
            "manifest.json",
            "--service-url",
            "https://graphiti.example.invalid",
            "--auth-secret-env",
            "GRAPHITI_SEARCH_SECRET",
        ]
    )
    assert parsed.manifest == "manifest.json"
