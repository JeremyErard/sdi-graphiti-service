"""Asynchronous ingestion: start a job, poll it, and never leak the episode.

Extraction is 25-50 sequential model calls and regularly outlives the edge
proxy's request ceiling. On 2026-08-27 every synchronous ingest died at ~301s,
the backend seeing `520 <!DOCTYPE html>` then `fetch failed`, while this
service was working normally throughout — a successful Voyage embedding
mid-flight, no error logged, no completion. Nothing we control raises that
ceiling, so the request had to stop waiting for the work.
"""

import asyncio

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.routers import ingest
from app.services import ingest_jobs

EPISODE = {
    "client_slug": "pokagon",
    "engagement_id": "eng-1",
    "episode_type": "document_analysis",
    "content": "Finance owns the monthly close and the tenant content marker.",
    "source_id": "doc-456",
    "source_type": "document",
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


def test_start_returns_a_handle_immediately_without_doing_the_work(monkeypatch):
    started = asyncio.Event()

    async def slow_ingest(req):
        started.set()
        await asyncio.sleep(30)  # would blow the proxy ceiling
        return {"episode_id": "never"}

    monkeypatch.setattr(ingest, "_perform_ingest", slow_ingest)
    r = client().post("/ingest/episode/async", json=EPISODE)

    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "running"
    assert body["job_id"]


def test_a_completed_job_reports_its_result(monkeypatch):
    async def fast(req):
        return {"episode_id": "ep-1", "entities_extracted": 3, "facts_created": 5, "graph_name": "g"}

    monkeypatch.setattr(ingest, "_perform_ingest", fast)
    c = client()
    job_id = c.post("/ingest/episode/async", json=EPISODE).json()["job_id"]
    body = c.post("/ingest/jobs/status", json={"job_id": job_id, "client_slug": "pokagon"}).json()

    assert body["status"] == "succeeded"
    assert body["result"]["entities_extracted"] == 3
    assert body["result"]["facts_created"] == 5


def test_a_failed_job_reports_why(monkeypatch):
    # The real 2026-08-27 cause, which reached the queue only as
    # "Graphiti ingestion returned null".
    async def boom(req):
        raise TypeError("AsyncMessages.create() got an unexpected keyword argument 'temperature'")

    monkeypatch.setattr(ingest, "_perform_ingest", boom)
    c = client()
    job_id = c.post("/ingest/episode/async", json=EPISODE).json()["job_id"]
    body = c.post("/ingest/jobs/status", json={"job_id": job_id, "client_slug": "pokagon"}).json()

    assert body["status"] == "failed"
    assert body["error_type"] == "TypeError"
    assert "temperature" in body["error_message"]


def test_a_failure_never_echoes_the_episode_content(monkeypatch):
    # The sync endpoint's pinned contract is that a failure discloses no tenant
    # content. Reporting a cause must not quietly weaken that.
    async def leaky(req):
        raise ValueError(f"failed while processing: {EPISODE['content']}")

    monkeypatch.setattr(ingest, "_perform_ingest", leaky)
    c = client()
    job_id = c.post("/ingest/episode/async", json=EPISODE).json()["job_id"]
    r = c.post("/ingest/jobs/status", json={"job_id": job_id, "client_slug": "pokagon"})

    assert r.json()["status"] == "failed"
    assert r.json()["error_type"] == "ValueError"
    assert "tenant content marker" not in r.text
    assert "monthly close" not in r.text


def test_another_tenant_cannot_read_this_job(monkeypatch):
    async def fast(req):
        return {"episode_id": "ep-1", "entities_extracted": 1, "facts_created": 1, "graph_name": "g"}

    monkeypatch.setattr(ingest, "_perform_ingest", fast)
    c = client()
    job_id = c.post("/ingest/episode/async", json=EPISODE).json()["job_id"]
    # Indistinguishable from "no such job" on purpose.
    r = c.post("/ingest/jobs/status", json={"job_id": job_id, "client_slug": "someone-else"})
    assert r.status_code == 404


def test_an_unknown_job_is_404_so_a_poller_cannot_read_it_as_success():
    r = client().post("/ingest/jobs/status", json={"job_id": "does-not-exist", "client_slug": "pokagon"})
    assert r.status_code == 404


def test_the_registry_never_evicts_a_running_job(monkeypatch):
    monkeypatch.setattr(ingest_jobs, "MAX_JOBS", 3)
    live = ingest_jobs.create("pokagon")
    for _ in range(10):
        done = ingest_jobs.create("pokagon")
        ingest_jobs.mark_succeeded(done.job_id, {"episode_id": "x"})

    assert ingest_jobs.get(live.job_id, "pokagon") is not None
    assert ingest_jobs.get(live.job_id, "pokagon").status == "running"


def test_finished_jobs_are_pruned_after_their_ttl(monkeypatch):
    job = ingest_jobs.create("pokagon")
    ingest_jobs.mark_succeeded(job.job_id, {"episode_id": "x"})
    assert ingest_jobs.get(job.job_id, "pokagon") is not None

    monkeypatch.setattr(ingest_jobs, "FINISHED_TTL_SECONDS", -1)
    assert ingest_jobs.get(job.job_id, "pokagon") is None


def test_sync_endpoint_still_hides_the_cause(monkeypatch):
    # Unchanged contract: pinned by test_provenance_ingest_contract.
    async def boom(req):
        raise ValueError(f"leak {EPISODE['content']}")

    monkeypatch.setattr(ingest, "_perform_ingest", boom)
    r = client().post("/ingest/episode", json=EPISODE)
    assert r.status_code == 500
    assert r.json() == {"detail": "Ingestion failed"}
