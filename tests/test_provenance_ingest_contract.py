"""P1 source-anchor contracts for structured and episode ingestion."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
import uuid

import falkordb
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routers import ingest as ingest_router
from app.routers import structured as structured_router
from app.services import graphiti_client


EPISODE_ID = "10000000-0000-4000-8000-000000000001"
FACT_ID = "20000000-0000-4000-8000-000000000001"
SUBJECT_ID = "30000000-0000-4000-8000-000000000001"
OBJECT_ID = "30000000-0000-4000-8000-000000000002"


class _QueryResult:
    def __init__(self, rows=None):
        self.result_set = rows or []


class _RecordingGraph:
    def __init__(self, *, failure: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.failure = failure

    def query(self, query: str, params: dict | None = None):
        if self.failure is not None:
            raise self.failure
        self.calls.append((query, params or {}))
        return _QueryResult()


class _RecordingDB:
    def __init__(self, graph: _RecordingGraph):
        self.graph = graph
        self.selected: list[str] = []

    def select_graph(self, graph_name: str):
        self.selected.append(graph_name)
        return self.graph


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
        return value if tz is not None else value.replace(tzinfo=None)


def _structured_client() -> TestClient:
    app = FastAPI()
    app.include_router(structured_router.router, prefix="/ingest")
    return TestClient(app)


def _v2_payload() -> dict:
    return {
        "contract_version": "structured_provenance_v2",
        "client_slug": "pokagon",
        "engagement_id": "engagement-123",
        "episode_uuid": EPISODE_ID,
        "episode_name": "document_analysis: document/doc-456",
        "episode_type": "document_analysis",
        "source_id": "doc-456",
        "source_type": "document",
        "source_description": "Operating-model source document",
        "anchor_mode": "typed_source",
        "producer_contract_version": "structured_provenance_v2",
        "reference_time": "2026-07-10T09:00:00Z",
        "entities": [
            {
                "name": "Finance Team",
                "type": "team",
                "description": "Finance operations",
            },
            {
                "name": "Monthly Close",
                "type": "process",
                "description": "Month-end close",
            },
        ],
        "relationships": [
            {
                "fact_id": FACT_ID,
                "source": "Finance Team",
                "target": "Monthly Close",
                "relation": "owns",
                "fact": "The Finance Team owns the Monthly Close process.",
            }
        ],
    }


def _install_fake_db(monkeypatch, graph: _RecordingGraph) -> _RecordingDB:
    db = _RecordingDB(graph)
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **_kwargs: db)
    return db


def test_structured_v2_writes_strict_anchors_and_stable_fact_id(monkeypatch):
    graph = _RecordingGraph()
    db = _install_fake_db(monkeypatch, graph)
    generated_ids = iter((uuid.UUID(SUBJECT_ID), uuid.UUID(OBJECT_ID)))
    monkeypatch.setattr(structured_router.uuidlib, "uuid4", lambda: next(generated_ids))
    monkeypatch.setattr(structured_router, "datetime", _FrozenDateTime)

    response = _structured_client().post("/ingest/structured/v2", json=_v2_payload())

    assert response.status_code == 200
    assert response.json() == {
        "graph_name": "client_pokagon",
        "episode_uuid": EPISODE_ID,
        "entities_created": 2,
        "entities_merged": 0,
        "relationships_created": 1,
        "relationships_skipped": 0,
        "elapsed_ms": 0,
        "contract_version": "structured_provenance_v2",
        "chain_status": "chained",
        "fact_ids": [FACT_ID],
    }
    assert db.selected == ["client_pokagon"]

    episode_query, episode_params = next(
        call for call in graph.calls if "CREATE (ep:Episodic" in call[0]
    )
    assert "producer_contract_version: $producer_contract_version" in episode_query
    assert episode_params == {
        "uuid": EPISODE_ID,
        "name": "document_analysis: document/doc-456",
        "content": "",
        "source_description": "Operating-model source document",
        "source_id": "doc-456",
        "source_type": "document",
        "engagement_id": "engagement-123",
        "episode_type": "document_analysis",
        "anchor_mode": "typed_source",
        "producer_contract_version": "structured_provenance_v2",
        "valid_at": "2026-07-10T09:00:00+00:00",
        "created_at": "2026-07-11T12:00:00+00:00",
        "group_id": "client_pokagon",
    }

    edge_query, edge_params = next(
        call for call in graph.calls if "CREATE (s)-[r:RELATES_TO" in call[0]
    )
    assert "source_uuid: $src" in edge_query
    assert "target_uuid: $tgt" in edge_query
    assert edge_params == {
        "src": SUBJECT_ID,
        "tgt": OBJECT_ID,
        "edge_uuid": FACT_ID,
        "name": "owns",
        "fact": "The Finance Team owns the Monthly Close process.",
        "episodes": [EPISODE_ID],
        "engagement_id": "engagement-123",
        "source_id": "doc-456",
        "source_type": "document",
        "episode_type": "document_analysis",
        "anchor_mode": "typed_source",
        "producer_contract_version": "structured_provenance_v2",
        "created_at": "2026-07-11T12:00:00+00:00",
        "group_id": "client_pokagon",
    }


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("missing_source_id", None),
        ("unsupported_anchor_mode", "exact"),
        ("unversioned_producer", "document_analysis_v1"),
        ("extra_field", "not-allowed"),
        ("mismatched_engagement_anchor", "engagement"),
    ],
)
def test_structured_v2_rejects_malformed_or_invented_anchors(
    monkeypatch, mutation, value
):
    graph = _RecordingGraph()
    db = _install_fake_db(monkeypatch, graph)
    payload = deepcopy(_v2_payload())
    if mutation == "missing_source_id":
        payload.pop("source_id")
    elif mutation == "unsupported_anchor_mode":
        payload["anchor_mode"] = value
    elif mutation == "unversioned_producer":
        payload["producer_contract_version"] = value
    elif mutation == "mismatched_engagement_anchor":
        payload["anchor_mode"] = value
    else:
        payload["projection_operation_id"] = value

    response = _structured_client().post("/ingest/structured/v2", json=payload)

    assert response.status_code == 422
    assert db.selected == []
    assert graph.calls == []


def test_structured_v2_rejects_duplicate_facts_and_undeclared_endpoints(monkeypatch):
    graph = _RecordingGraph()
    db = _install_fake_db(monkeypatch, graph)
    payload = _v2_payload()
    payload["relationships"].append(deepcopy(payload["relationships"][0]))
    payload["relationships"][1]["target"] = "Undeclared Team"

    response = _structured_client().post("/ingest/structured/v2", json=payload)

    assert response.status_code == 422
    assert db.selected == []


@pytest.mark.parametrize("collision", ["episode", "fact"])
def test_structured_v2_identity_collision_fails_before_any_write(
    monkeypatch, collision
):
    class _CollisionGraph(_RecordingGraph):
        def query(self, query: str, params: dict | None = None):
            self.calls.append((query, params or {}))
            if collision == "episode" and "MATCH (ep:Episodic {uuid:" in query:
                return _QueryResult([[EPISODE_ID]])
            if collision == "fact" and "MATCH ()-[edge:RELATES_TO]" in query:
                return _QueryResult([[FACT_ID]])
            return _QueryResult()

    graph = _CollisionGraph()
    _install_fake_db(monkeypatch, graph)

    response = _structured_client().post("/ingest/structured/v2", json=_v2_payload())

    assert response.status_code == 409
    assert response.json() == {"detail": "Structured ingest identity conflict"}
    assert not any("CREATE" in query for query, _params in graph.calls)


def test_legacy_structured_wire_response_is_preserved_and_written_pre_chain(
    monkeypatch,
):
    graph = _RecordingGraph()
    _install_fake_db(monkeypatch, graph)
    generated_ids = iter(
        (
            uuid.UUID(EPISODE_ID),
            uuid.UUID(SUBJECT_ID),
            uuid.UUID(OBJECT_ID),
            uuid.UUID(FACT_ID),
        )
    )
    monkeypatch.setattr(structured_router.uuidlib, "uuid4", lambda: next(generated_ids))
    monkeypatch.setattr(structured_router, "datetime", _FrozenDateTime)
    payload = {
        "client_slug": "pokagon",
        "episode_name": "legacy structured episode",
        "source_description": "legacy source label",
        "reference_time": "2026-07-10T09:00:00Z",
        "entities": [
            {"name": "Finance Team", "type": "Team", "description": ""},
            {"name": "Monthly Close", "type": "Process", "description": ""},
        ],
        "relationships": [
            {
                "source": "Finance Team",
                "target": "Monthly Close",
                "relation": "owns",
                "fact": "Finance owns close.",
            }
        ],
    }

    response = _structured_client().post("/ingest/structured", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "graph_name": "client_pokagon",
        "episode_uuid": EPISODE_ID,
        "entities_created": 2,
        "entities_merged": 0,
        "relationships_created": 1,
        "relationships_skipped": 0,
        "elapsed_ms": 0,
    }
    episode_query, episode_params = next(
        call for call in graph.calls if "CREATE (ep:Episodic" in call[0]
    )
    assert "anchor_mode: 'pre_chain'" in episode_query
    assert episode_params["producer_contract_version"] == "legacy_structured_v1"
    edge_query, edge_params = next(
        call for call in graph.calls if "CREATE (s)-[r:RELATES_TO" in call[0]
    )
    assert "anchor_mode: 'pre_chain'" in edge_query
    assert edge_params["producer_contract_version"] == "legacy_structured_v1"
    assert edge_params["src"] == SUBJECT_ID
    assert edge_params["tgt"] == OBJECT_ID


def test_structured_failure_returns_fixed_client_safe_detail(monkeypatch):
    _install_fake_db(monkeypatch, _RecordingGraph(failure=RuntimeError("tenant data")))

    response = _structured_client().post("/ingest/structured/v2", json=_v2_payload())

    assert response.status_code == 500
    assert response.json() == {"detail": "Structured ingest failed"}
    assert "tenant data" not in response.text


def test_episode_ingest_persists_legacy_episode_anchor_signature(monkeypatch):
    captured: dict = {}

    async def fake_add_episode(**kwargs):
        captured.update(kwargs)
        return {
            "episode_id": EPISODE_ID,
            "entities_extracted": 2,
            "facts_created": 1,
            "elapsed_ms": 3,
        }

    monkeypatch.setattr(graphiti_client, "add_episode", fake_add_episode)
    app = FastAPI()
    app.include_router(ingest_router.router, prefix="/ingest")
    response = TestClient(app).post(
        "/ingest/episode",
        json={
            "client_slug": "pokagon",
            "engagement_id": "engagement-123",
            "episode_type": "document_analysis",
            "content": "Finance owns the monthly close.",
            "source_id": "doc-456",
            "source_type": "document",
            "timestamp": "2026-07-10T09:00:00Z",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "episode_id": EPISODE_ID,
        "entities_extracted": 2,
        "facts_created": 1,
        "graph_name": "client_pokagon",
    }
    assert captured["engagement_id"] == "engagement-123"
    assert captured["source_id"] == "doc-456"
    assert captured["source_type"] == "document"
    assert captured["episode_type"] == "document_analysis"
    assert captured["anchor_mode"] == "legacy_episode_v0"
    assert captured["producer_contract_version"] == "legacy_episode_v0"


@pytest.mark.parametrize("anchor_mode", ["typed_source", "engagement"])
def test_episode_ingest_preserves_supplied_v2_producer_signature(
    monkeypatch, anchor_mode
):
    captured: dict = {}

    async def fake_add_episode(**kwargs):
        captured.update(kwargs)
        return {"episode_id": EPISODE_ID}

    monkeypatch.setattr(graphiti_client, "add_episode", fake_add_episode)
    app = FastAPI()
    app.include_router(ingest_router.router, prefix="/ingest")
    response = TestClient(app).post(
        "/ingest/episode",
        json={
            "client_slug": "pokagon",
            "engagement_id": "engagement-123",
            "episode_type": "document_analysis",
            "content": "Finance owns the monthly close.",
            "source_id": (
                "engagement-123" if anchor_mode == "engagement" else "doc-456"
            ),
            "source_type": (
                "engagement" if anchor_mode == "engagement" else "document"
            ),
            "anchor_mode": anchor_mode,
            "producer_contract_version": "engage_episode_v2",
        },
    )

    assert response.status_code == 200
    assert captured["anchor_mode"] == anchor_mode
    assert captured["producer_contract_version"] == "engage_episode_v2"


@pytest.mark.parametrize(
    "extra",
    [
        {"anchor_mode": "typed_source"},
        {"producer_contract_version": "engage_episode_v2"},
        {
            "anchor_mode": "typed_source",
            "producer_contract_version": "unknown_episode_v1",
        },
        {
            "anchor_mode": "exact",
            "producer_contract_version": "engage_episode_v2",
        },
        {"producer_contract_verison": "engage_episode_v2"},
        {
            "anchor_mode": "engagement",
            "producer_contract_version": "engage_episode_v2",
            "source_id": "not-the-engagement",
        },
        {
            "anchor_mode": "typed_source",
            "producer_contract_version": "engage_episode_v2",
            "source_type": "engagement",
            "source_id": "engagement-123",
        },
    ],
)
def test_episode_ingest_rejects_partial_or_unknown_v2_signature(monkeypatch, extra):
    called = False

    async def forbidden_add_episode(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(graphiti_client, "add_episode", forbidden_add_episode)
    payload = {
        "client_slug": "pokagon",
        "engagement_id": "engagement-123",
        "episode_type": "document_analysis",
        "content": "Finance owns the monthly close.",
        "source_id": "doc-456",
        "source_type": "document",
        **extra,
    }
    app = FastAPI()
    app.include_router(ingest_router.router, prefix="/ingest")

    response = TestClient(app).post("/ingest/episode", json=payload)

    assert response.status_code == 422
    assert called is False


def test_episode_ingest_failure_returns_fixed_client_safe_detail(monkeypatch):
    async def fail_add_episode(**_kwargs):
        raise RuntimeError("tenant content")

    monkeypatch.setattr(graphiti_client, "add_episode", fail_add_episode)
    app = FastAPI()
    app.include_router(ingest_router.router, prefix="/ingest")
    response = TestClient(app).post(
        "/ingest/episode",
        json={
            "client_slug": "pokagon",
            "engagement_id": "engagement-123",
            "episode_type": "document_analysis",
            "content": "Finance owns the monthly close.",
            "source_id": "doc-456",
            "source_type": "document",
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Ingestion failed"}
    assert "tenant content" not in response.text


def test_unanchored_bootstrap_add_episode_remains_compatible(monkeypatch):
    captured: dict = {}

    class _Client:
        async def add_episode(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                episode=SimpleNamespace(uuid=EPISODE_ID),
                nodes=[],
                edges=[],
            )

    async def fake_get_client(_client_slug):
        return _Client()

    monkeypatch.setattr(graphiti_client, "get_client", fake_get_client)

    result = asyncio.run(
        graphiti_client.add_episode(
            client_slug="pokagon",
            engagement_id="engagement-123",
            name="bootstrap: operating model",
            content="Operating-model content",
            source_description="Bootstrap document: general",
            reference_time=datetime(2026, 7, 11, tzinfo=timezone.utc),
        )
    )

    assert result["episode_id"] == EPISODE_ID
    assert captured["name"] == "bootstrap: operating model"


def test_add_episode_persists_supplied_v2_anchors_on_the_episode(monkeypatch):
    class _Client:
        async def add_episode(self, **_kwargs):
            return SimpleNamespace(
                episode=SimpleNamespace(uuid=EPISODE_ID),
                nodes=[],
                edges=[],
            )

    class _AnchorGraph(_RecordingGraph):
        def query(self, query: str, params: dict | None = None):
            self.calls.append((query, params or {}))
            return _QueryResult([[EPISODE_ID]])

    async def fake_get_client(_client_slug):
        return _Client()

    graph = _AnchorGraph()
    db = _install_fake_db(monkeypatch, graph)
    monkeypatch.setattr(graphiti_client, "get_client", fake_get_client)

    result = asyncio.run(
        graphiti_client.add_episode(
            client_slug="pokagon",
            engagement_id="engagement-123",
            name="document_analysis: document/doc-456",
            content="Finance owns the monthly close.",
            source_description="Document analysis source",
            reference_time=datetime(2026, 7, 11, tzinfo=timezone.utc),
            source_id="doc-456",
            source_type="document",
            episode_type="document_analysis",
            anchor_mode="typed_source",
            producer_contract_version="engage_episode_v2",
        )
    )

    assert result["episode_id"] == EPISODE_ID
    assert db.selected == ["client_pokagon"]
    query, params = graph.calls[0]
    assert "SET ep.source_id = $source_id" in query
    assert params == {
        "episode_uuid": EPISODE_ID,
        "group_id": "client_pokagon",
        "source_id": "doc-456",
        "source_type": "document",
        "engagement_id": "engagement-123",
        "episode_type": "document_analysis",
        "anchor_mode": "typed_source",
        "producer_contract_version": "engage_episode_v2",
    }
