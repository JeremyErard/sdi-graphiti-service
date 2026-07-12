"""P1 provenance filtering, accounting, and retrieval-path contract tests."""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import uuid

import falkordb
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from app.config import Settings
from app.models.search import (
    FactResult,
    FactSource,
    LegacyFactResult,
    LegacySearchContextResponse,
    ProvenanceShadow,
    ProvenanceSummary,
    SearchContextRequest,
    SearchContextResponse,
    ShadowSearchContextResponse,
)
from app.routers import search as search_router
from app.services import graphiti_client


FACT_IDS = [f"40000000-0000-4000-8000-{index:012d}" for index in range(1, 9)]
SUBJECT_ID = "50000000-0000-4000-8000-000000000001"
OBJECT_ID = "50000000-0000-4000-8000-000000000002"
EPISODE_ID = "60000000-0000-4000-8000-000000000001"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "graphiti_search_context_v3.json"
SHADOW_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "graphiti_search_context_shadow_v1.json"
)


@pytest.fixture(autouse=True)
def _enforced_provenance_mode(monkeypatch):
    monkeypatch.setattr(search_router.settings, "graphiti_provenance_mode", "enforce")


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(search_router.router, prefix="/search")
    return TestClient(app)


def _raw(fact_id: str):
    return SimpleNamespace(uuid=fact_id)


def _legacy_raw(fact_id: str, *, fact: str | None = None):
    return SimpleNamespace(
        uuid=fact_id,
        source_node_uuid=SUBJECT_ID,
        target_node_uuid=OBJECT_ID,
        name="owns",
        fact=fact or f"Compatibility fact {fact_id}",
        valid_at=None,
        invalid_at=None,
        expired_at=None,
    )


def _source(**overrides) -> graphiti_client.ResolvedEpisodeAnchor:
    values = {
        "episode_uuid": EPISODE_ID,
        "episode_name": "document_analysis: document/doc-456",
        "source_description": "Operating-model source document",
        "source_type": "document",
        "source_id": "doc-456",
        "engagement_id": "engagement-123",
        "episode_type": "document_analysis",
        "anchor_mode": "typed_source",
        "producer_contract_version": "structured_provenance_v2",
        "provenance_write_state": "complete",
        "valid_at": None,
    }
    values.update(overrides)
    return graphiti_client.ResolvedEpisodeAnchor(**values)


def _edge(
    fact_id: str,
    *,
    sources: tuple[graphiti_client.ResolvedEpisodeAnchor, ...] | None = None,
    expired_at: datetime | None = None,
    malformed: bool = False,
) -> graphiti_client.ResolvedSearchEdge:
    resolved_sources = (_source(),) if sources is None else sources
    return graphiti_client.ResolvedSearchEdge(
        fact_id=fact_id,
        subject_uuid=SUBJECT_ID,
        subject_name="Finance Team",
        predicate="owns",
        object_uuid=OBJECT_ID,
        object_name="Monthly Close",
        fact="The Finance Team owns the Monthly Close process.",
        episode_uuids=tuple(source.episode_uuid for source in resolved_sources),
        sources=resolved_sources,
        valid_at=None,
        invalid_at=None,
        expired_at=expired_at,
        malformed=malformed,
    )


def _patch_search(
    monkeypatch,
    *,
    raw_edges,
    resolved: dict[str, graphiti_client.ResolvedSearchEdge],
    malformed_response_events: int = 0,
    path: graphiti_client.RetrievalPath = "fast",
):
    calls: dict[str, dict] = {}

    async def fake_search_with_path(**kwargs):
        calls["search"] = kwargs
        return raw_edges, path

    async def fake_resolve_search_provenance(**kwargs):
        calls["resolve"] = kwargs
        return resolved, malformed_response_events

    monkeypatch.setattr(
        graphiti_client, "search_with_path", fake_search_with_path
    )
    monkeypatch.setattr(
        graphiti_client,
        "resolve_search_provenance",
        fake_resolve_search_provenance,
    )
    monkeypatch.setattr(search_router, "time", SimpleNamespace(time=lambda: 100.0))
    return calls


def _request(
    *,
    max_results: int = 1,
    include_segment=None,
    acceptance_probe=None,
) -> dict:
    payload = {
        "client_slug": "pokagon",
        "engagement_id": "engagement-123",
        "query": "Who owns monthly close?",
        "max_results": max_results,
    }
    if include_segment is not None:
        payload["include_segment"] = include_segment
    if acceptance_probe is not None:
        payload["acceptance_probe"] = acceptance_probe
    return payload


def test_search_exact_response_retains_endpoint_ids_names_and_source_anchors(
    monkeypatch,
):
    fact_id = FACT_IDS[0]
    calls = _patch_search(
        monkeypatch,
        raw_edges=[_raw(fact_id)],
        resolved={fact_id: _edge(fact_id)},
    )

    response = _client().post("/search/context", json=_request())

    assert response.status_code == 200
    assert response.json() == {
        "contract_version": "graphiti_search_context_v3",
        "facts": [
            {
                "fact_id": fact_id,
                "subject": SUBJECT_ID,
                "subject_name": "Finance Team",
                "predicate": "owns",
                "object": OBJECT_ID,
                "object_name": "Monthly Close",
                "fact": "The Finance Team owns the Monthly Close process.",
                "episodes": [EPISODE_ID],
                "sources": [
                    {
                        "episode_uuid": EPISODE_ID,
                        "episode_name": "document_analysis: document/doc-456",
                        "source_description": "Operating-model source document",
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
        ],
        "segment_insights": [],
        "graph_name": "client_pokagon",
        "search_time_ms": 0.0,
        "provenance_summary": {
            "contract_version": "graphiti_provenance_summary_v1",
            "candidates": 1,
            "service_forwarded": 1,
            "malformed_item_suppressed": 0,
            "expired_suppressed": 0,
            "pre_chain_suppressed": 0,
            "cross_engagement_suppressed": 0,
            "malformed_response_events": 0,
            "retrieval_path": "fast",
            "requested_results": 1,
            "overfetch_limit": 3,
            "starved_at_service": False,
        },
    }
    assert calls["search"]["max_results"] == 3
    assert calls["resolve"]["edges"][0].uuid == fact_id
    fact = response.json()["facts"][0]
    assert "reference_status" not in fact
    assert "content_grounding_status" not in fact


def test_canonical_v3_fixture_is_serialized_by_the_pydantic_contract():
    response = SearchContextResponse(
        facts=[
            FactResult(
                fact_id=FACT_IDS[0],
                subject=SUBJECT_ID,
                subject_name="Finance Team",
                predicate="owns",
                object=OBJECT_ID,
                object_name="Monthly Close",
                fact="The Finance Team owns the Monthly Close process.",
                episodes=[EPISODE_ID],
                sources=[
                    FactSource(
                        episode_uuid=EPISODE_ID,
                        episode_name="document_analysis: document/doc-456",
                        source_description="Operating-model source document",
                        source_type="document",
                        source_id="doc-456",
                        engagement_id="engagement-123",
                        episode_type="document_analysis",
                        anchor_mode="typed_source",
                        producer_contract_version="structured_provenance_v2",
                    )
                ],
            )
        ],
        graph_name="client_pokagon",
        search_time_ms=0.0,
        provenance_summary=ProvenanceSummary(
            candidates=1,
            service_forwarded=1,
            retrieval_path="fast",
            requested_results=1,
            overfetch_limit=3,
        ),
    )
    serialized = json.dumps(
        response.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
    ) + "\n"
    fixture = FIXTURE_PATH.read_text(encoding="utf-8")

    assert fixture == serialized
    assert SearchContextResponse.model_validate_json(fixture) == response
    assert hashlib.sha256(fixture.encode("utf-8")).hexdigest() == (
        "41bb8316a1dec2fa3b11eacda5378c42eab1bf97589790cda7f48167f54c6414"
    )


def test_canonical_shadow_fixture_preserves_legacy_top_level_and_v3_preview():
    response = ShadowSearchContextResponse(
        facts=[
            LegacyFactResult(
                subject=SUBJECT_ID,
                predicate="owns",
                object=OBJECT_ID,
                fact="The Finance Team owns the Monthly Close process.",
            )
        ],
        graph_name="client_pokagon",
        search_time_ms=0.0,
        provenance_shadow=ProvenanceShadow(
            facts=[
                FactResult(
                    fact_id=FACT_IDS[0],
                    subject=SUBJECT_ID,
                    subject_name="Finance Team",
                    predicate="owns",
                    object=OBJECT_ID,
                    object_name="Monthly Close",
                    fact="The Finance Team owns the Monthly Close process.",
                    episodes=[EPISODE_ID],
                    sources=[
                        FactSource(
                            episode_uuid=EPISODE_ID,
                            episode_name="document_analysis: document/doc-456",
                            source_description="Operating-model source document",
                            source_type="document",
                            source_id="doc-456",
                            engagement_id="engagement-123",
                            episode_type="document_analysis",
                            anchor_mode="typed_source",
                            producer_contract_version="structured_provenance_v2",
                        )
                    ],
                )
            ],
            provenance_summary=ProvenanceSummary(
                candidates=1,
                service_forwarded=1,
                retrieval_path="fast",
                requested_results=1,
                overfetch_limit=3,
            ),
        ),
    )
    serialized = json.dumps(
        response.model_dump(mode="json"), indent=2, sort_keys=True
    ) + "\n"
    fixture = SHADOW_FIXTURE_PATH.read_text(encoding="utf-8")

    assert fixture == serialized
    assert ShadowSearchContextResponse.model_validate_json(fixture) == response
    assert hashlib.sha256(fixture.encode("utf-8")).hexdigest() == (
        "8936dae2746e057b35c17953f7fdca205e9e94ce9f14de86252bd44ba18290cc"
    )


def test_rollout_configuration_defaults_safe_and_rejects_unknown_modes():
    defaults = Settings(_env_file=None)
    assert defaults.graphiti_provenance_mode == "legacy"
    assert defaults.graphiti_structured_v2_write_mode == "off"
    assert defaults.graphiti_acceptance_probe_mode is False
    with pytest.raises(ValidationError):
        Settings(_env_file=None, graphiti_provenance_mode="unknown")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, graphiti_structured_v2_write_mode="enabled")
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            graphiti_provenance_mode="shadow",
            graphiti_structured_v2_write_mode="staged",
        )
    staged = Settings(
        _env_file=None,
        graphiti_provenance_mode="enforce",
        graphiti_structured_v2_write_mode="staged",
    )
    assert staged.graphiti_structured_v2_write_mode == "staged"
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            graphiti_acceptance_probe_mode=True,
            graphiti_provenance_mode="shadow",
            graphiti_auth_mode="required",
        )
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            graphiti_acceptance_probe_mode=True,
            graphiti_provenance_mode="enforce",
            graphiti_auth_mode="optional",
        )
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            graphiti_acceptance_probe_mode=True,
            graphiti_provenance_mode="enforce",
            graphiti_auth_mode="required",
            voyage_api_key="   ",
        )
    probe = Settings(
        _env_file=None,
        graphiti_acceptance_probe_mode=True,
        graphiti_provenance_mode="enforce",
        graphiti_auth_mode="required",
        voyage_api_key="probe-voyage-key",
    )
    assert probe.graphiti_acceptance_probe_mode is True


def test_legacy_mode_makes_one_k_sized_search_and_preserves_old_wire_cap(
    monkeypatch,
):
    monkeypatch.setattr(search_router.settings, "graphiti_provenance_mode", "legacy")
    calls: list[int] = []
    raw_edges = [_legacy_raw(fact_id) for fact_id in FACT_IDS]
    raw_edges.extend(
        _legacy_raw(f"40000000-0000-4000-8001-{index:012d}")
        for index in range(1, 9)
    )

    async def fake_search_with_path(**kwargs):
        calls.append(kwargs["max_results"])
        return raw_edges, "fast"

    async def forbidden_resolver(**_kwargs):
        raise AssertionError("legacy mode must not run provenance resolution")

    monkeypatch.setattr(graphiti_client, "search_with_path", fake_search_with_path)
    monkeypatch.setattr(
        graphiti_client, "resolve_search_provenance", forbidden_resolver
    )
    monkeypatch.setattr(search_router, "time", SimpleNamespace(time=lambda: 100.0))

    response = _client().post("/search/context", json=_request(max_results=50))

    assert response.status_code == 200
    body = response.json()
    assert calls == [50]
    assert list(body) == ["facts", "segment_insights", "graph_name", "search_time_ms"]
    assert len(body["facts"]) == 15
    assert [fact["fact"] for fact in body["facts"]] == [
        edge.fact for edge in raw_edges[:15]
    ]
    assert set(body["facts"][0]) == {
        "subject",
        "predicate",
        "object",
        "fact",
        "valid_from",
        "valid_to",
        "expired_at",
    }


def test_shadow_preserves_legacy_output_and_adds_non_enforcing_v3_preview(
    monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="graphiti_service")
    monkeypatch.setattr(search_router.settings, "graphiti_provenance_mode", "shadow")
    sentinel = "RAW-COMPATIBILITY-SENTINEL"
    compatibility_edges = [_legacy_raw(FACT_IDS[0], fact=sentinel)]
    preview_edges = [_raw(FACT_IDS[0]), _raw(FACT_IDS[1])]
    calls: list[int] = []

    async def fake_search_with_path(**kwargs):
        calls.append(kwargs["max_results"])
        return (
            (compatibility_edges, "fast")
            if len(calls) == 1
            else (preview_edges, "hybrid_fallback")
        )

    async def fake_resolve(**_kwargs):
        return {
            FACT_IDS[0]: _edge(FACT_IDS[0]),
            FACT_IDS[1]: _edge(FACT_IDS[1], sources=()),
        }, 0

    monkeypatch.setattr(graphiti_client, "search_with_path", fake_search_with_path)
    monkeypatch.setattr(graphiti_client, "resolve_search_provenance", fake_resolve)
    monkeypatch.setattr(search_router, "time", SimpleNamespace(time=lambda: 100.0))

    response = _client().post("/search/context", json=_request(max_results=1))

    assert response.status_code == 200
    body = response.json()
    assert calls == [1, 3]
    assert body["facts"][0]["fact"] == sentinel
    assert body["provenance_shadow"]["contract_version"] == (
        "graphiti_provenance_shadow_v1"
    )
    assert body["provenance_shadow"]["enforcement_applied"] is False
    assert [fact["fact_id"] for fact in body["provenance_shadow"]["facts"]] == [
        FACT_IDS[0]
    ]
    summary = body["provenance_shadow"]["provenance_summary"]
    assert summary["candidates"] == 1
    assert summary["service_forwarded"] == 1
    assert summary["retrieval_path"] == "hybrid_fallback"
    # An old consumer can validate only the established top-level contract and
    # ignore the additive nested preview.
    old_wire = LegacySearchContextResponse.model_validate(body)
    assert old_wire.facts[0].fact == sentinel
    assert sentinel not in caplog.text


def test_shadow_preview_failure_does_not_change_compatibility_output(monkeypatch):
    monkeypatch.setattr(search_router.settings, "graphiti_provenance_mode", "shadow")
    compatibility_edges = [
        _legacy_raw(
            f"40000000-0000-4000-8005-{index:012d}",
            fact=f"legacy survives {index}",
        )
        for index in range(1, 17)
    ]
    calls = 0

    async def fake_search_with_path(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return compatibility_edges, "fast"
        raise RuntimeError("preview unavailable")

    monkeypatch.setattr(graphiti_client, "search_with_path", fake_search_with_path)
    monkeypatch.setattr(search_router, "time", SimpleNamespace(time=lambda: 100.0))

    response = _client().post("/search/context", json=_request(max_results=50))

    assert response.status_code == 200
    body = response.json()
    assert calls == 2
    assert [fact["fact"] for fact in body["facts"]] == [
        f"legacy survives {index}" for index in range(1, 16)
    ]
    assert body["provenance_shadow"]["facts"] == []
    assert body["provenance_shadow"]["provenance_summary"][
        "malformed_response_events"
    ] == 1
    assert (
        body["provenance_shadow"]["provenance_summary"]["retrieval_path"]
        == "fast"
    )


def test_shadow_resolution_failure_is_attributed_to_the_preview_path(monkeypatch):
    monkeypatch.setattr(
        search_router.settings,
        "graphiti_provenance_mode",
        "shadow",
    )
    calls = 0

    async def fake_search_with_path(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_legacy_raw(FACT_IDS[0], fact="legacy survives")], "fast"
        return [_raw(FACT_IDS[0])], "hybrid_fallback"

    async def fail_preview_resolution(**_kwargs):
        raise RuntimeError("preview resolution unavailable")

    monkeypatch.setattr(graphiti_client, "search_with_path", fake_search_with_path)
    monkeypatch.setattr(
        graphiti_client,
        "resolve_search_provenance",
        fail_preview_resolution,
    )
    monkeypatch.setattr(search_router, "time", SimpleNamespace(time=lambda: 100.0))

    response = _client().post("/search/context", json=_request(max_results=1))

    assert response.status_code == 200
    body = response.json()
    assert body["facts"][0]["fact"] == "legacy survives"
    assert body["provenance_shadow"]["facts"] == []
    summary = body["provenance_shadow"]["provenance_summary"]
    assert summary["malformed_response_events"] == 1
    assert summary["retrieval_path"] == "hybrid_fallback"


def test_enforce_mode_makes_only_the_bounded_overfetch_call(monkeypatch):
    calls = _patch_search(
        monkeypatch,
        raw_edges=[_raw(FACT_IDS[0])],
        resolved={FACT_IDS[0]: _edge(FACT_IDS[0])},
    )

    response = _client().post("/search/context", json=_request(max_results=2))

    assert response.status_code == 200
    assert calls["search"]["max_results"] == 6


def test_acceptance_probe_boolean_is_a_bidirectional_process_fence(monkeypatch):
    called = False

    async def forbidden_search(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("mismatched probe fence must precede graph access")

    monkeypatch.setattr(graphiti_client, "search_with_path", forbidden_search)
    response = _client().post(
        "/search/context",
        json=_request(acceptance_probe=True),
    )
    assert response.status_code == 409
    assert response.json() == {
        "detail": "Acceptance probe request/process mode mismatch"
    }
    assert called is False

    monkeypatch.setattr(search_router.settings, "graphiti_acceptance_probe_mode", True)
    response = _client().post("/search/context", json=_request())
    assert response.status_code == 409
    assert called is False

    response = _client().post(
        "/search/context",
        json={**_request(), "acceptance_probe": "true"},
    )
    assert response.status_code == 422
    assert called is False

    _patch_search(
        monkeypatch,
        raw_edges=[_raw(FACT_IDS[0])],
        resolved={FACT_IDS[0]: _edge(FACT_IDS[0])},
    )
    response = _client().post(
        "/search/context",
        json=_request(acceptance_probe=True),
    )
    assert response.status_code == 200
    assert response.json()["provenance_summary"]["retrieval_path"] == "fast"


def test_segment_defaults_off_and_true_is_rejected_before_graph_access(monkeypatch):
    assert SearchContextRequest(
        client_slug="pokagon",
        engagement_id="engagement-123",
        query="test",
    ).include_segment is False
    called = False

    async def forbidden_search(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("segment rejection must happen before search")

    monkeypatch.setattr(graphiti_client, "search_with_path", forbidden_search)

    response = _client().post(
        "/search/context", json=_request(include_segment=True)
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Segment context requires a governed pattern contract"
    }
    assert called is False


@pytest.mark.parametrize("mode", ["legacy", "shadow", "enforce"])
def test_segment_true_is_rejected_before_graph_access_in_every_mode(
    monkeypatch, mode
):
    monkeypatch.setattr(search_router.settings, "graphiti_provenance_mode", mode)
    called = False

    async def forbidden_search(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("segment rejection must precede all search modes")

    monkeypatch.setattr(graphiti_client, "search_with_path", forbidden_search)
    response = _client().post(
        "/search/context", json=_request(include_segment=True)
    )

    assert response.status_code == 409
    assert called is False


@pytest.mark.parametrize(
    ("edge", "bucket"),
    [
        (
            _edge(
                FACT_IDS[0],
                sources=(
                    _source(
                        anchor_mode="pre_chain",
                        producer_contract_version="legacy_structured_v1",
                    ),
                ),
            ),
            "pre_chain_suppressed",
        ),
        (
            _edge(
                FACT_IDS[0],
                sources=(
                    _source(
                        anchor_mode="typed_source",
                        producer_contract_version="legacy_structured_v1",
                    ),
                ),
            ),
            "pre_chain_suppressed",
        ),
        (
            _edge(FACT_IDS[0], sources=(_source(source_id=None),)),
            "malformed_item_suppressed",
        ),
        (
            _edge(
                FACT_IDS[0],
                sources=(_source(anchor_mode="engagement"),),
            ),
            "malformed_item_suppressed",
        ),
        (
            _edge(FACT_IDS[0], sources=(_source(malformed=True),)),
            "malformed_item_suppressed",
        ),
        (
            _edge(
                FACT_IDS[0],
                expired_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
            ),
            "expired_suppressed",
        ),
        (
            _edge(
                FACT_IDS[0],
                sources=(_source(engagement_id="engagement-other"),),
            ),
            "cross_engagement_suppressed",
        ),
    ],
)
def test_search_fail_closed_terminal_buckets(monkeypatch, edge, bucket):
    fact_id = FACT_IDS[0]
    _patch_search(
        monkeypatch,
        raw_edges=[_raw(fact_id)],
        resolved={fact_id: edge},
    )

    response = _client().post("/search/context", json=_request())

    assert response.status_code == 200
    body = response.json()
    assert body["facts"] == []
    summary = body["provenance_summary"]
    assert summary["candidates"] == 1
    assert summary["service_forwarded"] == 0
    assert summary[bucket] == 1
    terminal_total = sum(
        summary[name]
        for name in (
            "service_forwarded",
            "malformed_item_suppressed",
            "expired_suppressed",
            "pre_chain_suppressed",
            "cross_engagement_suppressed",
        )
    )
    assert terminal_total == summary["candidates"]
    assert summary["starved_at_service"] is True


def test_mixed_overfetch_pool_has_exact_ordered_prefix_algebra(monkeypatch):
    raw_edges = [_raw(fact_id) for fact_id in FACT_IDS[:6]]
    resolved = {
        FACT_IDS[0]: _edge(FACT_IDS[0], sources=()),
        FACT_IDS[1]: _edge(FACT_IDS[1]),
        FACT_IDS[2]: _edge(
            FACT_IDS[2],
            expired_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
        ),
        FACT_IDS[3]: _edge(
            FACT_IDS[3], sources=(_source(engagement_id="engagement-other"),)
        ),
        FACT_IDS[4]: _edge(FACT_IDS[4], malformed=True),
        FACT_IDS[5]: _edge(
            FACT_IDS[5], sources=(_source(episode_uuid=FACT_IDS[7]),)
        ),
    }
    calls = _patch_search(
        monkeypatch,
        raw_edges=raw_edges,
        resolved=resolved,
        path="hybrid_fallback",
    )

    response = _client().post(
        "/search/context", json=_request(max_results=2)
    )

    assert response.status_code == 200
    body = response.json()
    assert [fact["fact_id"] for fact in body["facts"]] == [
        FACT_IDS[1],
        FACT_IDS[5],
    ]
    assert body["provenance_summary"] == {
        "contract_version": "graphiti_provenance_summary_v1",
        "candidates": 6,
        "service_forwarded": 2,
        "malformed_item_suppressed": 1,
        "expired_suppressed": 1,
        "pre_chain_suppressed": 1,
        "cross_engagement_suppressed": 1,
        "malformed_response_events": 0,
        "retrieval_path": "hybrid_fallback",
        "requested_results": 2,
        "overfetch_limit": 6,
        "starved_at_service": False,
    }
    assert calls["search"]["max_results"] == 6


def test_duplicate_hits_are_deduped_and_unexamined_tail_is_excluded(monkeypatch):
    pre_chain_id, eligible_id, tail_id = FACT_IDS[:3]
    raw_edges = [
        _raw(pre_chain_id),
        _raw(pre_chain_id),
        _raw(eligible_id),
        _raw(tail_id),
    ]
    resolved = {
        pre_chain_id: _edge(pre_chain_id, sources=()),
        eligible_id: _edge(eligible_id),
        tail_id: _edge(tail_id, malformed=True),
    }
    _patch_search(monkeypatch, raw_edges=raw_edges, resolved=resolved)

    response = _client().post("/search/context", json=_request(max_results=1))

    assert response.status_code == 200
    body = response.json()
    assert [fact["fact_id"] for fact in body["facts"]] == [eligible_id]
    assert body["provenance_summary"]["candidates"] == 2
    assert body["provenance_summary"]["pre_chain_suppressed"] == 1
    assert body["provenance_summary"]["malformed_item_suppressed"] == 0


def test_independent_valid_source_wins_over_a_corrupt_duplicate_anchor(monkeypatch):
    fact_id = FACT_IDS[0]
    valid_episode_id = FACT_IDS[7]
    edge = _edge(
        fact_id,
        sources=(
            _source(malformed=True),
            _source(episode_uuid=valid_episode_id),
        ),
    )
    _patch_search(
        monkeypatch,
        raw_edges=[_raw(fact_id)],
        resolved={fact_id: edge},
    )

    response = _client().post("/search/context", json=_request())

    assert response.status_code == 200
    body = response.json()
    assert body["provenance_summary"]["service_forwarded"] == 1
    assert body["provenance_summary"]["malformed_item_suppressed"] == 0
    assert body["facts"][0]["episodes"] == [valid_episode_id]
    assert [
        source["episode_uuid"] for source in body["facts"][0]["sources"]
    ] == [valid_episode_id]


def test_wholly_malformed_pool_is_an_event_not_filter_starvation(monkeypatch):
    async def fake_search_with_path(**_kwargs):
        return None, "fast"

    async def forbidden_resolver(**_kwargs):
        raise AssertionError("unsupported producer shape must not reach resolver")

    monkeypatch.setattr(graphiti_client, "search_with_path", fake_search_with_path)
    monkeypatch.setattr(
        graphiti_client, "resolve_search_provenance", forbidden_resolver
    )
    monkeypatch.setattr(search_router, "time", SimpleNamespace(time=lambda: 100.0))

    response = _client().post(
        "/search/context", json=_request(max_results=2)
    )

    assert response.status_code == 200
    summary = response.json()["provenance_summary"]
    assert summary["candidates"] == 0
    assert summary["malformed_response_events"] == 1
    assert summary["starved_at_service"] is False


def test_naturally_short_exact_response_is_not_filter_starvation(monkeypatch):
    fact_id = FACT_IDS[0]
    _patch_search(
        monkeypatch,
        raw_edges=[_raw(fact_id)],
        resolved={fact_id: _edge(fact_id)},
    )

    response = _client().post(
        "/search/context", json=_request(max_results=2)
    )

    assert response.status_code == 200
    summary = response.json()["provenance_summary"]
    assert summary["service_forwarded"] == 1
    assert summary["candidates"] == 1
    assert summary["starved_at_service"] is False


def test_staged_structured_anchor_is_pre_chain_until_finalized(monkeypatch):
    fact_id = FACT_IDS[0]
    staged = _edge(
        fact_id,
        sources=(_source(provenance_write_state="staging"),),
    )
    _patch_search(
        monkeypatch,
        raw_edges=[_raw(fact_id)],
        resolved={fact_id: staged},
    )

    response = _client().post("/search/context", json=_request())

    assert response.status_code == 200
    body = response.json()
    assert body["facts"] == []
    assert body["provenance_summary"]["pre_chain_suppressed"] == 1


def test_engage_episode_anchor_does_not_require_structured_write_state(monkeypatch):
    fact_id = FACT_IDS[0]
    source = _source(
        producer_contract_version="engage_episode_v2",
        provenance_write_state=None,
    )
    _patch_search(
        monkeypatch,
        raw_edges=[_raw(fact_id)],
        resolved={fact_id: _edge(fact_id, sources=(source,))},
    )

    response = _client().post("/search/context", json=_request())

    assert response.status_code == 200
    assert response.json()["provenance_summary"]["service_forwarded"] == 1


def _many_sources(start: int, count: int):
    return tuple(
        _source(
            episode_uuid=f"60000000-0000-4000-8002-{index:012d}",
        )
        for index in range(start, start + count)
    )


def test_more_than_64_sources_suppresses_the_fact_without_truncating_authority(
    monkeypatch,
):
    fact_id = FACT_IDS[0]
    _patch_search(
        monkeypatch,
        raw_edges=[_raw(fact_id)],
        resolved={fact_id: _edge(fact_id, sources=_many_sources(1, 65))},
    )

    response = _client().post("/search/context", json=_request())

    assert response.status_code == 200
    body = response.json()
    assert body["facts"] == []
    assert body["provenance_summary"]["malformed_item_suppressed"] == 1
    assert body["provenance_summary"]["starved_at_service"] is True


def test_response_never_exceeds_500_aggregate_source_anchors(monkeypatch):
    fact_ids = [
        f"40000000-0000-4000-8003-{index:012d}" for index in range(1, 9)
    ]
    resolved = {
        fact_id: _edge(
            fact_id,
            sources=_many_sources((index * 64) + 1, 64),
        )
        for index, fact_id in enumerate(fact_ids)
    }
    _patch_search(
        monkeypatch,
        raw_edges=[_raw(fact_id) for fact_id in fact_ids],
        resolved=resolved,
    )

    response = _client().post(
        "/search/context", json=_request(max_results=8)
    )

    assert response.status_code == 200
    body = response.json()
    assert sum(len(fact["sources"]) for fact in body["facts"]) == 448
    assert body["provenance_summary"]["service_forwarded"] == 7
    assert body["provenance_summary"]["malformed_item_suppressed"] == 1
    assert body["provenance_summary"]["starved_at_service"] is True


def test_router_caps_an_oversized_producer_pool_and_reports_the_violation(
    monkeypatch,
):
    raw_edges = [_raw(fact_id) for fact_id in FACT_IDS[:4]]
    resolved = {fact_id: _edge(fact_id) for fact_id in FACT_IDS[:3]}
    _patch_search(monkeypatch, raw_edges=raw_edges, resolved=resolved)

    response = _client().post(
        "/search/context", json=_request(max_results=1)
    )

    assert response.status_code == 200
    summary = response.json()["provenance_summary"]
    assert summary["overfetch_limit"] == 3
    assert summary["candidates"] == 1
    assert summary["malformed_response_events"] == 1


def test_search_failure_returns_fixed_client_safe_detail(monkeypatch):
    async def fail_search(**_kwargs):
        raise RuntimeError("tenant graph query and content")

    monkeypatch.setattr(graphiti_client, "search_with_path", fail_search)

    response = _client().post("/search/context", json=_request())

    assert response.status_code == 500
    assert response.json() == {"detail": "Search failed"}
    assert "tenant graph query" not in response.text


class _QueryResult:
    def __init__(self, rows):
        self.result_set = rows


class _ResolutionGraph:
    def __init__(self, edge_rows, source_rows):
        self.edge_rows = edge_rows
        self.source_rows = source_rows
        self.calls: list[tuple[str, dict]] = []

    def query(self, query: str, params: dict | None = None):
        self.calls.append((query, params or {}))
        if "MATCH (subject:Entity)" in query:
            return _QueryResult(self.edge_rows)
        if "MATCH (episode:Episodic)" in query:
            return _QueryResult(self.source_rows)
        raise AssertionError(f"unexpected query: {query}")


class _ResolutionDB:
    def __init__(self, graph):
        self.graph = graph
        self.selected: list[str] = []

    def select_graph(self, graph_name):
        self.selected.append(graph_name)
        return self.graph


def _resolution_rows(fact_id=FACT_IDS[0]):
    edge_row = [
        fact_id,
        SUBJECT_ID,
        "Finance Team",
        "owns",
        OBJECT_ID,
        "Monthly Close",
        "The Finance Team owns the Monthly Close process.",
        [EPISODE_ID],
        None,
        None,
        None,
    ]
    source_row = [
        EPISODE_ID,
        "document_analysis: document/doc-456",
        "Operating-model source document",
        "document",
        "doc-456",
        "engagement-123",
        "document_analysis",
        "typed_source",
        "structured_provenance_v2",
        None,
        "complete",
    ]
    return edge_row, source_row


def test_batched_resolution_unifies_fallback_with_endpoint_names_and_anchors(
    monkeypatch,
):
    edge_row, source_row = _resolution_rows()
    graph = _ResolutionGraph([edge_row], [source_row])
    db = _ResolutionDB(graph)
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **_kwargs: db)

    resolved, events = asyncio.run(
        graphiti_client.resolve_search_provenance(
            client_slug="pokagon",
            edges=[SimpleNamespace(uuid=FACT_IDS[0])],
        )
    )

    assert events == 0
    assert db.selected == ["client_pokagon"]
    assert resolved[FACT_IDS[0]] == _edge(FACT_IDS[0])
    assert len(graph.calls) == 2
    assert graph.calls[0][1] == {
        "edge_uuids": [FACT_IDS[0]],
        "group_id": "client_pokagon",
    }
    assert graph.calls[1][1] == {
        "episode_uuids": [EPISODE_ID],
        "group_id": "client_pokagon",
    }


def test_duplicate_authoritative_fact_rows_are_suppressed_once(monkeypatch):
    edge_row, source_row = _resolution_rows()
    graph = _ResolutionGraph([edge_row, list(edge_row)], [source_row])
    db = _ResolutionDB(graph)
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **_kwargs: db)

    resolved, events = asyncio.run(
        graphiti_client.resolve_search_provenance(
            client_slug="pokagon",
            edges=[_raw(FACT_IDS[0]), _raw(FACT_IDS[0])],
        )
    )

    assert events == 0
    assert list(resolved) == [FACT_IDS[0]]
    assert resolved[FACT_IDS[0]].malformed is True
    assert resolved[FACT_IDS[0]].sources[0].malformed is False


@pytest.mark.parametrize("reverse_rows", [False, True])
def test_duplicate_episode_rows_cannot_select_a_favorable_engagement(
    monkeypatch, reverse_rows
):
    edge_row, source_row = _resolution_rows()
    conflicting = list(source_row)
    conflicting[5] = "engagement-other"
    source_rows = [source_row, conflicting]
    if reverse_rows:
        source_rows.reverse()
    graph = _ResolutionGraph([edge_row], source_rows)
    db = _ResolutionDB(graph)
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **_kwargs: db)

    resolved, events = asyncio.run(
        graphiti_client.resolve_search_provenance(
            client_slug="pokagon",
            edges=[_raw(FACT_IDS[0])],
        )
    )

    assert events == 0
    assert resolved[FACT_IDS[0]].malformed is False
    assert resolved[FACT_IDS[0]].sources[0].malformed is True


def test_missing_or_invalid_raw_fact_ids_are_response_events_not_candidates(
    monkeypatch,
):
    graph = _ResolutionGraph([], [])
    db = _ResolutionDB(graph)
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **_kwargs: db)

    resolved, events = asyncio.run(
        graphiti_client.resolve_search_provenance(
            client_slug="pokagon",
            edges=[SimpleNamespace(), _raw("not-a-uuid")],
        )
    )

    assert resolved == {}
    assert events == 2
    assert db.selected == []


def test_fast_search_path_is_reported_without_fallback(monkeypatch):
    edge = _raw(FACT_IDS[0])

    async def fake_fast(_client_slug, _query, _max_results):
        return [edge]

    async def forbidden_client(_client_slug):
        raise AssertionError("fallback must not run for a non-empty fast result")

    monkeypatch.setattr(graphiti_client, "_search_fast", fake_fast)
    monkeypatch.setattr(graphiti_client, "get_client", forbidden_client)

    edges, path = asyncio.run(
        graphiti_client.search_with_path("pokagon", "monthly close", 3)
    )

    assert edges == [edge]
    assert path == "fast"


def test_fast_query_joins_endpoints_and_projects_names_and_episodes(monkeypatch):
    edge_row, _source_row = _resolution_rows()
    fast_row = [
        edge_row[0],
        edge_row[6],
        edge_row[3],
        edge_row[1],
        edge_row[2],
        edge_row[4],
        edge_row[5],
        edge_row[7],
        edge_row[8],
        edge_row[9],
        edge_row[10],
    ]

    class _Embedder:
        async def create(self, *, input_data):
            assert input_data == ["monthly close"]
            return [0.1, 0.2]

    class _Graph:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        def query(self, query, params=None):
            self.calls.append((query, params or {}))
            return _QueryResult([fast_row])

    graph = _Graph()
    db = _ResolutionDB(graph)
    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: _Embedder())
    monkeypatch.setattr(
        graphiti_client, "_ensure_edge_vector_index", lambda *_args: None
    )
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **_kwargs: db)

    edges = asyncio.run(
        graphiti_client._search_fast("pokagon", "monthly close", 1)
    )

    assert len(edges) == 1
    assert edges[0].source_node_uuid == SUBJECT_ID
    assert edges[0].source_node_name == "Finance Team"
    assert edges[0].target_node_uuid == OBJECT_ID
    assert edges[0].target_node_name == "Monthly Close"
    assert edges[0].episodes == [EPISODE_ID]
    assert len(graph.calls) == 2
    for query, params in graph.calls:
        assert "YIELD relationship AS rel, score" in query
        assert "MATCH (a:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(b:Entity)" in query
        assert "WHERE e.group_id = $group_id" in query
        assert "a.uuid AS src, a.name AS src_name" in query
        assert "b.uuid AS tgt, b.name AS tgt_name" in query
        assert "e.episodes AS episodes" in query
        assert params["group_id"] == "client_pokagon"


@pytest.mark.parametrize("fast_failure", [None, RuntimeError("index unavailable")])
def test_empty_or_failed_fast_search_uses_hybrid_fallback(monkeypatch, fast_failure):
    fallback_edge = _raw(FACT_IDS[0])
    captured: dict = {}

    async def fake_fast(_client_slug, _query, _max_results):
        if fast_failure is not None:
            raise fast_failure
        return []

    class _Client:
        async def search(self, **kwargs):
            captured.update(kwargs)
            return [fallback_edge]

    async def fake_client(_client_slug):
        return _Client()

    monkeypatch.setattr(graphiti_client, "_search_fast", fake_fast)
    monkeypatch.setattr(graphiti_client, "get_client", fake_client)

    edges, path = asyncio.run(
        graphiti_client.search_with_path("pokagon", "monthly close", 3)
    )

    assert edges == [fallback_edge]
    assert path == "hybrid_fallback"
    assert captured == {
        "query": "monthly close",
        "num_results": 3,
        "group_ids": ["client_pokagon"],
    }


def test_malformed_fallback_shape_reaches_router_telemetry_boundary(monkeypatch):
    async def empty_fast(_client_slug, _query, _max_results):
        return []

    class _Client:
        async def search(self, **_kwargs):
            return None

    async def fake_client(_client_slug):
        return _Client()

    monkeypatch.setattr(graphiti_client, "_search_fast", empty_fast)
    monkeypatch.setattr(graphiti_client, "get_client", fake_client)

    edges, path = asyncio.run(
        graphiti_client.search_with_path("pokagon", "monthly close", 3)
    )

    assert edges is None
    assert path == "hybrid_fallback"


def test_probe_fast_search_and_resolution_use_only_existing_graph_ro_queries(
    monkeypatch,
):
    edge_row, source_row = _resolution_rows()
    fast_row = [
        edge_row[0],
        edge_row[6],
        edge_row[3],
        edge_row[1],
        edge_row[2],
        edge_row[4],
        edge_row[5],
        edge_row[7],
        edge_row[8],
        edge_row[9],
        edge_row[10],
    ]

    class _Embedder:
        async def create(self, *, input_data):
            assert input_data == ["monthly close"]
            return [0.1, 0.2]

    class _ProbeGraph:
        def __init__(self):
            self.reads: list[tuple[str, dict]] = []

        def query(self, *_args, **_kwargs):
            raise AssertionError("probe graph access must never use query")

        def ro_query(self, query, params=None):
            self.reads.append((query, params or {}))
            if "db.idx.vector.queryRelationships" in query:
                return _QueryResult([fast_row])
            if "db.idx.fulltext.queryRelationships" in query:
                return _QueryResult([fast_row])
            if "MATCH (subject:Entity)" in query:
                return _QueryResult([edge_row])
            if "MATCH (episode:Episodic)" in query:
                return _QueryResult([source_row])
            raise AssertionError(f"unexpected read-only query: {query}")

    class _ProbeDB:
        def __init__(self, graph):
            self.graph = graph
            self.selected: list[str] = []

        def list_graphs(self):
            return ["client_pokagon"]

        def select_graph(self, graph_name):
            self.selected.append(graph_name)
            return self.graph

    graph = _ProbeGraph()
    db = _ProbeDB(graph)

    async def forbidden_client(*_args, **_kwargs):
        raise AssertionError("probe mode must never initialize Graphiti")

    def forbidden_index(*_args, **_kwargs):
        raise AssertionError("probe mode must never ensure or build an index")

    def forbidden_llm(*_args, **_kwargs):
        raise AssertionError("probe mode must never initialize a generative client")

    monkeypatch.setattr(graphiti_client.settings, "graphiti_acceptance_probe_mode", True)
    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: _Embedder())
    monkeypatch.setattr(graphiti_client, "_ensure_edge_vector_index", forbidden_index)
    monkeypatch.setattr(graphiti_client, "_create_llm_client", forbidden_llm)
    monkeypatch.setattr(graphiti_client, "get_client", forbidden_client)
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **_kwargs: db)

    edges, path = asyncio.run(
        graphiti_client.search_with_path("pokagon", "monthly close", 3)
    )
    resolved, events = asyncio.run(
        graphiti_client.resolve_search_provenance(
            "pokagon",
            edges,
        )
    )

    assert path == "fast"
    assert events == 0
    assert resolved[FACT_IDS[0]] == _edge(FACT_IDS[0])
    assert db.selected == ["client_pokagon", "client_pokagon"]
    assert len(graph.reads) == 4


@pytest.mark.parametrize("fast_failure", [None, RuntimeError("unavailable")])
def test_probe_fast_empty_or_failure_never_falls_back_or_gets_client(
    monkeypatch, fast_failure
):
    async def fake_fast(*_args, **_kwargs):
        if fast_failure is not None:
            raise fast_failure
        return []

    async def forbidden_client(*_args, **_kwargs):
        raise AssertionError("probe mode must never enter Graphiti fallback")

    monkeypatch.setattr(graphiti_client.settings, "graphiti_acceptance_probe_mode", True)
    monkeypatch.setattr(graphiti_client, "_search_fast", fake_fast)
    monkeypatch.setattr(graphiti_client, "get_client", forbidden_client)

    with pytest.raises(graphiti_client.AcceptanceProbeReadError):
        asyncio.run(
            graphiti_client.search_with_path("pokagon", "monthly close", 3)
        )


def test_probe_missing_graph_fails_before_select_embedding_or_index(monkeypatch):
    class _MissingDB:
        def __init__(self):
            self.selected = []

        def list_graphs(self):
            return ["client_other"]

        def select_graph(self, graph_name):
            self.selected.append(graph_name)
            raise AssertionError("missing graph must never be selected")

    db = _MissingDB()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("missing graph preflight must fail first")

    monkeypatch.setattr(graphiti_client.settings, "graphiti_acceptance_probe_mode", True)
    monkeypatch.setattr(graphiti_client, "_create_embedder", forbidden)
    monkeypatch.setattr(graphiti_client, "_ensure_edge_vector_index", forbidden)
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **_kwargs: db)

    with pytest.raises(graphiti_client.AcceptanceProbeReadError):
        asyncio.run(graphiti_client._search_fast("pokagon", "monthly close", 3))

    assert db.selected == []


def test_fast_row_projection_retains_endpoint_uuid_name_and_episode_list():
    edge = graphiti_client._row_to_edge(
        [
            FACT_IDS[0],
            "Finance owns close.",
            "owns",
            SUBJECT_ID,
            "Finance Team",
            OBJECT_ID,
            "Monthly Close",
            [EPISODE_ID],
            None,
            None,
            None,
        ]
    )

    assert vars(edge) == {
        "uuid": FACT_IDS[0],
        "fact": "Finance owns close.",
        "name": "owns",
        "source_node_uuid": SUBJECT_ID,
        "source_node_name": "Finance Team",
        "target_node_uuid": OBJECT_ID,
        "target_node_name": "Monthly Close",
        "episodes": [EPISODE_ID],
        "valid_at": None,
        "invalid_at": None,
        "expired_at": None,
    }


def test_episode_reference_parser_rejects_oversized_storage_without_truncation():
    too_many = [
        f"60000000-0000-4000-8004-{index:012d}" for index in range(1, 66)
    ]
    assert graphiti_client._episode_uuid_list(too_many) == ((), False)
    assert graphiti_client._episode_uuid_list("[" + ("x" * 100_001) + "]") == (
        (),
        False,
    )
    allowed = too_many[:64]
    parsed, valid = graphiti_client._episode_uuid_list(allowed)
    assert valid is True
    assert parsed == tuple(allowed)
    assert graphiti_client._episode_uuid_list([allowed[0], allowed[0]]) == (
        (),
        False,
    )


def test_episode_reference_character_limit_precedes_utf8_encoding():
    class _HugeString(str):
        def encode(self, *_args, **_kwargs):
            raise AssertionError("oversized string must not be encoded")

    assert graphiti_client._episode_uuid_list(
        _HugeString("x" * 100_001)
    ) == ((), False)
