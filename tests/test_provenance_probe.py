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
                "retrieval_path": "fast",
                "expected": [
                    {
                        "fact_id": FACT_ID,
                        "episode_uuid": EPISODE_ID,
                        "source_type": "document",
                        "source_id": "doc-456",
                        "episode_type": "document_analysis",
                        "anchor_mode": "typed_source",
                        "producer_contract_version": "structured_provenance_v2",
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
                        "valid_at": None,
                    }
                ],
                "chain_status": "chained",
                "valid_from": None,
                "valid_to": None,
                "expired_at": None,
            }
        ]
    return {
        "contract_version": "graphiti_search_context_v3",
        "facts": facts,
        "segment_insights": [],
        "graph_name": "client_pokagon",
        "search_time_ms": 12.5,
        "provenance_summary": {
            "contract_version": "graphiti_provenance_summary_v1",
            "candidates": len(facts),
            "service_forwarded": len(facts),
            "malformed_item_suppressed": 0,
            "expired_suppressed": 0,
            "pre_chain_suppressed": 0,
            "cross_engagement_suppressed": 0,
            "malformed_response_events": 0,
            "retrieval_path": "fast",
            "requested_results": 5,
            "overfetch_limit": 15,
            "starved_at_service": False,
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
        lambda payload: payload["probes"][0].update(retrieval_path="hybrid_fallback"),
        lambda payload: payload["probes"][0]["expected"][0].update(
            fact_id=FACT_ID.replace("-", "")
        ),
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
        "acceptance_probe": True,
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


def test_local_auth_mode_value_is_not_treated_as_remote_auth_evidence():
    manifest = provenance_probe.ProbeManifest.model_validate(_manifest_payload())

    result = provenance_probe.run_probe_manifest(
        manifest,
        service_url=manifest.service_url,
        auth_secret_env=manifest.auth_secret_env,
        environ={
            "GRAPHITI_AUTH_MODE": "optional",
            "GRAPHITI_SEARCH_SECRET": SECRET,
        },
        opener=lambda *_args, **_kwargs: _HTTPResponse(_response()),
    )

    assert result["codes"] == {"PROBE_PASS": 1}


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


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda payload: payload.update(unratified=True),
            "RESPONSE_TOP_LEVEL_SHAPE_INVALID",
        ),
        (
            lambda payload: payload.update(graph_name="client_other"),
            "RESPONSE_GRAPH_MISMATCH",
        ),
        (
            lambda payload: payload["facts"][0].pop("valid_from"),
            "FACT_INVALID_SHAPE",
        ),
        (
            lambda payload: payload["facts"][0]["sources"][0].update(
                unratified=True
            ),
            "FACT_SOURCE_SHAPE_INVALID",
        ),
        (
            lambda payload: payload["facts"][0]["episodes"].append(
                "81000000-0000-4000-8000-000000000099"
            ),
            "FACT_SOURCE_EPISODE_SET_MISMATCH",
        ),
        (
            lambda payload: payload["provenance_summary"].update(candidates=2),
            "SUMMARY_ACCOUNTING_INVALID",
        ),
        (
            lambda payload: payload["provenance_summary"].update(
                starved_at_service=True
            ),
            "SUMMARY_STARVATION_INVALID",
        ),
        (
            lambda payload: payload["provenance_summary"].update(
                requested_results=4
            ),
            "SUMMARY_REQUEST_MISMATCH",
        ),
        (
            lambda payload: payload["provenance_summary"].update(
                retrieval_path="hybrid_fallback"
            ),
            "SUMMARY_RETRIEVAL_PATH_MISMATCH",
        ),
        (
            lambda payload: payload["facts"][0]["sources"][0].update(
                anchor_mode="engagement"
            ),
            "EXPECTED_SOURCE_MISSING",
        ),
    ],
)
def test_probe_rejects_adversarial_v3_shape_and_algebra(mutate, code):
    manifest = provenance_probe.ProbeManifest.model_validate(_manifest_payload())
    payload = _response()
    mutate(payload)

    with pytest.raises(provenance_probe.ProbeFailure) as failure:
        provenance_probe._validate_probe_response(
            manifest,
            manifest.probes[0],
            payload,
        )

    assert failure.value.code == code


def _source_for_episode(episode_id: str) -> dict:
    source = deepcopy(_response()["facts"][0]["sources"][0])
    source["episode_uuid"] = episode_id
    return source


def test_probe_enforces_64_sources_per_fact_without_truncation():
    manifest = provenance_probe.ProbeManifest.model_validate(_manifest_payload())
    payload = _response()
    episode_ids = [
        EPISODE_ID,
        *[
            f"81000000-0000-4000-8001-{index:012d}"
            for index in range(1, 65)
        ],
    ]
    payload["facts"][0]["episodes"] = episode_ids
    payload["facts"][0]["sources"] = [
        _source_for_episode(episode_id) for episode_id in episode_ids
    ]

    with pytest.raises(provenance_probe.ProbeFailure) as failure:
        provenance_probe._validate_probe_response(
            manifest,
            manifest.probes[0],
            payload,
        )

    assert failure.value.code == "FACT_EPISODE_LIMIT_EXCEEDED"


def test_probe_enforces_500_sources_per_response_without_truncation():
    manifest_payload = _manifest_payload()
    manifest_payload["probes"][0]["max_results"] = 50
    manifest = provenance_probe.ProbeManifest.model_validate(manifest_payload)
    facts = []
    source_index = 0
    for fact_index in range(9):
        fact = deepcopy(_response()["facts"][0])
        fact["fact_id"] = f"80000000-0000-4000-8001-{fact_index + 1:012d}"
        episode_ids = []
        for _ in range(56):
            source_index += 1
            episode_ids.append(
                f"81000000-0000-4000-8002-{source_index:012d}"
            )
        if fact_index == 0:
            episode_ids[0] = EPISODE_ID
            fact["fact_id"] = FACT_ID
        fact["episodes"] = episode_ids
        fact["sources"] = [
            _source_for_episode(episode_id) for episode_id in episode_ids
        ]
        facts.append(fact)
    payload = _response(facts=facts)
    payload["provenance_summary"].update(
        requested_results=50,
        overfetch_limit=150,
    )

    with pytest.raises(provenance_probe.ProbeFailure) as failure:
        provenance_probe._validate_probe_response(
            manifest,
            manifest.probes[0],
            payload,
        )

    assert failure.value.code == "RESPONSE_SOURCE_LIMIT_EXCEEDED"


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
