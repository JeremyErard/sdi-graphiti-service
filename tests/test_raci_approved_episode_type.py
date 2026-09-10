"""The service accepts the RACI-approved episode the backend produces.

Phase 2 (2026-09) turned each SOP into a RACI chart the client reviews, corrects
and approves. Every one of those charts and review threads was invisible to the
graph: `EpisodeType` is a closed enum, and an unknown value is a 422 before any
work starts. This pins the accepted shape: the exact typed-source v2 request the
backend's `raci_approved` producer sends.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pydantic
import pytest

from app.models.episode import EpisodeAnchorMode, EpisodeType, IngestEpisodeRequest
from app.provenance_contract import EPISODE_PROVENANCE_CONTRACT_VERSION
from app.routers import ingest
from app.services import ingest_jobs

RACI_EPISODE = {
    "client_slug": "pokagon",
    "engagement_id": "eng-1",
    "episode_type": "raci_approved",
    "content": (
        "RACI chart: Table Fill Inspection (approved version 3).\n"
        "Step 2 Verify the fill count: Responsible Table Games Supervisor; "
        "Accountable Table Games Shift Manager.\n"
        "Review thread: Wolgamott (Director) requested a change on step 2: "
        "the Shift Manager, not the Supervisor, signs the fill slip."
    ),
    "source_id": "draft-raci-1",
    "source_type": "raci",
    "anchor_mode": "typed_source",
    "producer_contract_version": EPISODE_PROVENANCE_CONTRACT_VERSION,
    "metadata": {"workflowId": "wf-1", "versionNumber": 3, "changeRequestCount": 1},
}


def client() -> TestClient:
    app = FastAPI()
    app.include_router(ingest.router, prefix="/ingest")
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_registry():
    ingest_jobs.reset()
    yield
    ingest_jobs.reset()


def test_raci_approved_is_an_episode_type_anchored_on_its_chart():
    req = IngestEpisodeRequest(**RACI_EPISODE)

    assert req.episode_type is EpisodeType.RACI_APPROVED
    assert req.episode_type.value == "raci_approved"
    assert req.anchor_mode is EpisodeAnchorMode.TYPED_SOURCE
    assert req.source_id == "draft-raci-1"


def test_a_misspelt_type_is_still_rejected_so_the_enum_stays_closed():
    with pytest.raises(pydantic.ValidationError) as caught:
        IngestEpisodeRequest(**{**RACI_EPISODE, "episode_type": "raci_approval"})

    assert caught.value.errors()[0]["loc"] == ("episode_type",)


def test_the_async_ingest_endpoint_accepts_the_backend_raci_request(monkeypatch):
    seen: list[IngestEpisodeRequest] = []

    async def record(req):
        seen.append(req)
        return {"episode_id": "ep-raci", "entities_extracted": 4, "facts_created": 6, "graph_name": "client_pokagon"}

    monkeypatch.setattr(ingest, "_perform_ingest", record)
    c = client()
    r = c.post("/ingest/episode/async", json=RACI_EPISODE)

    assert r.status_code == 202
    body = c.post("/ingest/jobs/status", json={"job_id": r.json()["job_id"], "client_slug": "pokagon"}).json()
    assert body["status"] == "succeeded"
    assert body["result"]["facts_created"] == 6
    assert seen[0].episode_type is EpisodeType.RACI_APPROVED
    assert seen[0].metadata["changeRequestCount"] == 1
