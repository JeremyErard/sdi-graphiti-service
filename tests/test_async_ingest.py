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


def test_a_cancelled_task_does_not_leave_its_job_running(monkeypatch):
    """CancelledError is a BaseException, so `except Exception` missed it.

    A cancelled task used to leave its job pinned at "running" forever: the
    poller asked every 5s, got "running" every time, and spent its entire
    budget on work that had already stopped.
    """
    async def cancelled(req):
        raise asyncio.CancelledError()

    monkeypatch.setattr(ingest, "_perform_ingest", cancelled)
    c = client()
    job_id = c.post("/ingest/episode/async", json=EPISODE).json()["job_id"]
    body = c.post("/ingest/jobs/status", json={"job_id": job_id, "client_slug": "pokagon"}).json()

    assert body["status"] == "failed"
    assert body["status"] != "running"


def test_a_running_job_that_never_reports_is_eventually_abandoned(monkeypatch):
    """Only FINISHED jobs are pruned, so a task that dies without recording an
    outcome would otherwise answer "still working" forever."""
    job = ingest_jobs.create("pokagon")
    assert ingest_jobs.get(job.job_id, "pokagon").status == "running"

    monkeypatch.setattr(ingest_jobs, "RUNNING_MAX_AGE_SECONDS", -1)
    abandoned = ingest_jobs.get(job.job_id, "pokagon")

    # Abandoned, NOT deleted: a poller gets a verdict rather than a 404 it
    # would have to interpret.
    assert abandoned is not None
    assert abandoned.status == "failed"
    assert abandoned.error_type == "IngestJobAbandoned"


def test_an_abandoned_job_is_still_tenant_scoped(monkeypatch):
    job = ingest_jobs.create("pokagon")
    monkeypatch.setattr(ingest_jobs, "RUNNING_MAX_AGE_SECONDS", -1)
    assert ingest_jobs.get(job.job_id, "someone-else") is None


def test_a_busy_service_still_accepts_and_hands_back_a_handle(monkeypatch):
    """Serialising must not make the service look down.

    The semaphore is acquired inside the background task, never in the request
    handler, so a caller always gets its handle immediately and a queued
    episode reports "running" — accurate, since it has been accepted.
    """
    async def slow(req):
        await asyncio.sleep(30)
        return {"episode_id": "never"}

    monkeypatch.setattr(ingest, "_perform_ingest", slow)
    ingest._ingest_slots = None
    c = client()

    first = c.post("/ingest/episode/async", json=EPISODE)
    second = c.post("/ingest/episode/async", json=EPISODE)

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["job_id"] != first.json()["job_id"]
    assert second.json()["status"] == "running"


def test_extraction_is_serialised_to_one_at_a_time(monkeypatch):
    """Two concurrent extractions starved the event loop on a 1-CPU service
    until the HTTP server stopped answering (2026-08-28)."""
    concurrent = 0
    peak = 0

    async def tracked(req):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1
        return {"episode_id": "ep", "entities_extracted": 0, "facts_created": 0, "graph_name": "g"}

    monkeypatch.setattr(ingest, "_perform_ingest", tracked)
    ingest._ingest_slots = None
    c = client()
    for _ in range(4):
        assert c.post("/ingest/episode/async", json=EPISODE).status_code == 202

    assert peak == 1, f"expected extractions serialised, saw {peak} at once"


def test_the_limit_comes_from_config_so_it_can_be_raised_with_the_cpu(monkeypatch):
    monkeypatch.setattr(ingest.settings, "max_concurrent_ingests", 3)
    ingest._ingest_slots = None
    assert ingest._slots()._value == 3
