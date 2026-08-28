"""In-process registry for asynchronous episode ingestion.

WHY THIS EXISTS
---------------
`POST /ingest/episode` ran the whole extraction inside the request. Extraction
is 25-50 sequential model calls, which for ordinary 8-11 KB episodes takes
longer than the edge proxy allows. Observed 2026-08-27: every attempt died at
~301s with the backend seeing

    [graphiti] /ingest/episode returned 520: <!DOCTYPE html>
    [graphiti] /ingest/episode failed: fetch failed

while this service was still working normally — a successful Voyage embedding
mid-flight, no error, no completion. The 520 with an HTML body is the proxy
terminating the connection, not anything here failing. Nothing we control can
raise that ceiling, so the request must stop waiting for the work.

The caller now starts a job, gets a handle immediately, and polls.

WHAT THIS IS NOT
----------------
This registry is IN-PROCESS and therefore NOT durable. A restart or a second
instance loses the record, and a poll for a forgotten job returns 404. That is
deliberate, and callers must treat "unknown job" as a retryable failure rather
than as success — the durable record of the work lives in the caller's own
queue, which already retries. Making this durable means persisting job state,
which is a larger change than the outage warranted.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["running", "succeeded", "failed"]

# Finished jobs are kept this long so a poller that is briefly behind can still
# read the outcome, then pruned so the registry cannot grow without bound.
FINISHED_TTL_SECONDS = 30 * 60

# Hard ceiling on retained jobs. Reached only if something floods the endpoint;
# the oldest finished jobs are dropped first so a live job is never evicted.
MAX_JOBS = 500

# Errors are reported to an authenticated, tenant-scoped caller, but an episode
# body must never come back out inside an exception string. Any contiguous run
# of this many characters from the submitted content disqualifies the message.
CONTENT_ECHO_WINDOW = 32


@dataclass
class IngestJob:
    job_id: str
    # The tenant this job belongs to. A poll must present a matching slug, so
    # one ingest-scoped caller cannot read another tenant's job by guessing a
    # handle. Graphs are per-client and job outcomes name entity/fact counts,
    # so this is a tenant boundary like any other.
    client_slug: str = ""
    status: JobStatus = "running"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None

    def elapsed_ms(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return (end - self.created_at) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms()),
        }
        if self.result is not None:
            body["result"] = self.result
        if self.error_type is not None:
            body["error_type"] = self.error_type
        if self.error_message is not None:
            body["error_message"] = self.error_message
        return body


_jobs: dict[str, IngestJob] = {}


def sanitize_error(error: BaseException, content: str) -> tuple[str, str | None]:
    """Return (error_type, safe_message).

    The type is always safe — it is a class name. The message is only returned
    when it demonstrably does not echo the submitted episode, because the sync
    endpoint's contract (pinned by test_provenance_ingest_contract) is that a
    failure never discloses tenant content.

    This check is what makes the message worth returning at all. On 2026-08-27
    the real cause was

        AsyncMessages.create() got an unexpected keyword argument 'temperature'

    which carries no tenant data and is exactly what nobody could see, because
    every failure reached the queue as "Graphiti ingestion returned null".
    """
    error_type = type(error).__name__
    message = str(error).strip()
    if not message:
        return error_type, None
    if _echoes_content(message, content):
        return error_type, None
    return error_type, message[:300]


def _echoes_content(message: str, content: str) -> bool:
    """True if any CONTENT_ECHO_WINDOW-char run of content appears in message."""
    text = (content or "").strip()
    if len(text) < CONTENT_ECHO_WINDOW:
        # Too short to fingerprint safely; treat any appearance as an echo.
        return bool(text) and text in message
    for i in range(0, len(text) - CONTENT_ECHO_WINDOW + 1):
        if text[i : i + CONTENT_ECHO_WINDOW] in message:
            return True
    return False


def create(client_slug: str) -> IngestJob:
    _prune()
    job = IngestJob(job_id=uuid.uuid4().hex, client_slug=client_slug)
    _jobs[job.job_id] = job
    return job


def get(job_id: str, client_slug: str) -> IngestJob | None:
    """Return the job only if it belongs to this tenant.

    A mismatch is indistinguishable from "no such job" on purpose: telling a
    caller that a handle exists but belongs to someone else discloses that the
    handle is real.
    """
    _prune()
    job = _jobs.get(job_id)
    if job is None or job.client_slug != client_slug:
        return None
    return job


def mark_succeeded(job_id: str, result: dict[str, Any]) -> None:
    job = _jobs.get(job_id)
    if job is None:
        return
    job.status = "succeeded"
    job.result = result
    job.finished_at = time.time()


def mark_failed(job_id: str, error: BaseException, content: str) -> None:
    job = _jobs.get(job_id)
    if job is None:
        return
    job.status = "failed"
    job.error_type, job.error_message = sanitize_error(error, content)
    job.finished_at = time.time()


def _prune() -> None:
    now = time.time()
    for job_id, job in list(_jobs.items()):
        if job.finished_at is not None and now - job.finished_at > FINISHED_TTL_SECONDS:
            _jobs.pop(job_id, None)
    if len(_jobs) <= MAX_JOBS:
        return
    # Over the cap: drop the oldest FINISHED jobs first. A running job is never
    # evicted, because its poller has nothing else to learn the outcome from.
    finished = sorted(
        (j for j in _jobs.values() if j.finished_at is not None),
        key=lambda j: j.finished_at or 0.0,
    )
    for job in finished:
        if len(_jobs) <= MAX_JOBS:
            break
        _jobs.pop(job.job_id, None)


def reset() -> None:
    """Drop all jobs. Tests only — the registry is module-level shared state."""
    _jobs.clear()
