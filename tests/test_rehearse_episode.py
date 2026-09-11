"""A producer's episode can be rehearsed on a scratch graph, never the tenant graph by accident.

Ruled 2026-09-11: rehearse the client-record producer on a scratch copy before
its records touch the Pokagon graph. The admin route takes the exact episode
the backend would queue plus a scratch graph name, and only a scratch name.
"""

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.provenance_contract import EPISODE_PROVENANCE_CONTRACT_VERSION
from app.routers import admin, ingest
from app.services import graphiti_client

EPISODE = {
    "client_slug": "pokagon",
    "engagement_id": "eng-phase-1",
    "episode_type": "approval_history",
    "content": "Approval record: Table Fill Inspection. Map version 2 approved 2026-05-02 by Beverly Robb (Director of Compliance).",
    "source_id": "wf-1",
    "source_type": "process",
    "anchor_mode": "typed_source",
    "producer_contract_version": EPISODE_PROVENANCE_CONTRACT_VERSION,
    "metadata": {"workflowId": "wf-1", "part": 1, "partCount": 1},
}


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    app.include_router(ingest.router, prefix="/ingest")
    return TestClient(app)


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    async def fake_add_episode(**kwargs):
        calls.append(kwargs)
        return {"episode_id": "ep-1", "entities_extracted": 3, "facts_created": 4, "elapsed_ms": 12.0}

    monkeypatch.setattr(graphiti_client, "add_episode", fake_add_episode)
    return calls


def test_rehearsal_writes_the_same_episode_to_the_scratch_graph(captured):
    body = {**EPISODE, "scratch_graph": "scratch_pokagon_records"}
    response = _client().post("/admin/rehearse-episode", json=body)
    assert response.status_code == 200
    assert response.json()["graph_name"] == "scratch_pokagon_records"
    assert captured[0]["graph_name_override"] == "scratch_pokagon_records"
    assert captured[0]["client_slug"] == "pokagon"
    assert captured[0]["episode_type"] == "approval_history"
    assert captured[0]["anchor_mode"] == "typed_source"
    assert captured[0]["source_id"] == "wf-1"


def test_only_a_scratch_name_is_accepted():
    for name in ("client_pokagon", "pokagon", "scratch_Pokagon", "scratch_", "scratch_" + "x" * 61):
        response = _client().post("/admin/rehearse-episode", json={**EPISODE, "scratch_graph": name})
        assert response.status_code == 422, name


def test_the_tenant_ingest_route_never_passes_an_override(captured):
    response = _client().post("/ingest/episode", json=EPISODE)
    assert response.status_code == 200
    assert response.json()["graph_name"] == "client_pokagon"
    assert captured[0]["graph_name_override"] is None


@pytest.mark.anyio
async def test_add_episode_targets_the_override_graph_and_its_group_id(monkeypatch):
    used: dict = {}

    class FakeGraphiti:
        def __init__(self, name):
            self.name = name

        async def add_episode(self, **kwargs):
            used["graph"] = self.name
            used["group_id"] = kwargs["group_id"]

            class Result:
                nodes = []
                edges = []
                episode = type("E", (), {"uuid": "ep-9"})()

            return Result()

    async def fake_get_client_for_graph(graph_name):
        return FakeGraphiti(graph_name)

    async def fake_get_client(client_slug):
        return FakeGraphiti("client_" + client_slug)

    async def quiet(*_args, **_kwargs):
        return None

    monkeypatch.setattr(graphiti_client, "get_client_for_graph", fake_get_client_for_graph)
    monkeypatch.setattr(graphiti_client, "get_client", fake_get_client)
    monkeypatch.setattr(graphiti_client, "_log_slow_queries", quiet)
    monkeypatch.setattr(graphiti_client, "entity_type_models", lambda: None)

    # Client selection and group_id are decided before any anchor write, so the
    # unanchored form exercises exactly the override without a driver.
    await graphiti_client.add_episode(
        "pokagon", "eng-1", "approval_history: process/wf-1", "content", "Engagement eng-1 — approval_history from process",
        datetime.now(timezone.utc), graph_name_override="scratch_pokagon_records",
    )
    assert used == {"graph": "scratch_pokagon_records", "group_id": "scratch_pokagon_records"}
    await graphiti_client.add_episode(
        "pokagon", "eng-1", "approval_history: process/wf-1", "content", "Engagement eng-1 — approval_history from process",
        datetime.now(timezone.utc),
    )
    assert used == {"graph": "client_pokagon", "group_id": "client_pokagon"}
