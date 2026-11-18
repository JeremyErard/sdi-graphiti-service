"""The service accepts the courseware-approved episode the backend produces.

`EpisodeType` is a closed enum, and an unknown value is a 422 before any work
starts. This pins the accepted shape: a typed-source v2 request with the
`courseware_approved` type, end to end through `/ingest/episode/async`.
All values here are invented.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pydantic
import pytest

from app.models.episode import EpisodeAnchorMode, EpisodeType, IngestEpisodeRequest
from app.provenance_contract import EPISODE_PROVENANCE_CONTRACT_VERSION
from app.routers import ingest
from app.services import ingest_jobs

COURSEWARE_EPISODE = {
    "client_slug": "acme-test",
    "engagement_id": "eng-1",
    "episode_type": "courseware_approved",
    "content": (
        "Courseware: Safe Lifting Basics (approved version 2).\n"
        "Module 1 teaches the two-person lift for loads over 50 pounds.\n"
        "Review thread: the reviewer asked for the lift limit to appear on the first screen."
    ),
    "source_id": "draft-course-1",
    "source_type": "courseware",
    "anchor_mode": "typed_source",
    "producer_contract_version": EPISODE_PROVENANCE_CONTRACT_VERSION,
    "metadata": {"workflowId": "wf-1", "versionNumber": 2},
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


def test_courseware_approved_is_an_episode_type_anchored_on_its_source():
    req = IngestEpisodeRequest(**COURSEWARE_EPISODE)

    assert req.episode_type is EpisodeType.COURSEWARE_APPROVED
    assert req.episode_type.value == "courseware_approved"
    assert req.anchor_mode is EpisodeAnchorMode.TYPED_SOURCE
    assert req.source_id == "draft-course-1"


def test_a_misspelt_type_is_still_rejected_so_the_enum_stays_closed():
    with pytest.raises(pydantic.ValidationError) as caught:
        IngestEpisodeRequest(**{**COURSEWARE_EPISODE, "episode_type": "courseware_approval"})

    assert caught.value.errors()[0]["loc"] == ("episode_type",)


def test_the_async_ingest_endpoint_accepts_the_backend_courseware_request(monkeypatch):
    seen: list[IngestEpisodeRequest] = []

    async def record(req):
        seen.append(req)
        return {"episode_id": "ep-course", "entities_extracted": 3, "facts_created": 5, "graph_name": "client_acme-test"}

    monkeypatch.setattr(ingest, "_perform_ingest", record)
    c = client()
    r = c.post("/ingest/episode/async", json=COURSEWARE_EPISODE)

    assert r.status_code == 202
    body = c.post("/ingest/jobs/status", json={"job_id": r.json()["job_id"], "client_slug": "acme-test"}).json()
    assert body["status"] == "succeeded"
    assert body["result"]["facts_created"] == 5
    assert seen[0].episode_type is EpisodeType.COURSEWARE_APPROVED
