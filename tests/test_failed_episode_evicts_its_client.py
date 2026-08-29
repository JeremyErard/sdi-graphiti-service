"""A timed-out episode must not hand its poisoned pool to the next one.

The outage this guards: FalkorDB was idle, ~100 keys, 53MB, logging no slow
query and no error, and still every ingest died on "Timeout reading from
falkordb:6379" -- including a bare count. Meanwhile the readiness probe, which
builds a FRESH client per call, answered in 1.5s.

The difference was the shared pool. A timeout is a cancelled read; redis-py
asyncio leaves that reply unread in the connection buffer, so the next borrower
reads the previous command's response and desyncs in turn. Once the first query
timed out, every later one did, until the process was restarted by hand.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from app.services import graphiti_client


class _PoisonedClient:
    """Stands in for a Graphiti client whose pool has gone bad."""

    def __init__(self):
        self.closed = False

    async def add_episode(self, **kwargs):
        raise TimeoutError("Timeout reading from falkordb-z6in:6379")

    async def close(self):
        self.closed = True


def _episode(monkeypatch, client):
    monkeypatch.setattr(graphiti_client, "get_client", lambda slug: _wrap(client))
    return graphiti_client.add_episode(
        client_slug="pokagon",
        engagement_id="eng-1",
        name="document_analysis",
        content="body",
        source_description="doc",
        reference_time=datetime.now(timezone.utc),
    )


async def _wrap(client):
    return client


def test_a_timed_out_episode_evicts_the_client_it_used(monkeypatch):
    poisoned = _PoisonedClient()
    graph = graphiti_client._graph_name_for_client("pokagon")
    graphiti_client._clients.clear()
    graphiti_client._clients[graph] = poisoned

    with pytest.raises(TimeoutError):
        asyncio.run(_episode(monkeypatch, poisoned))

    assert graph not in graphiti_client._clients, "next episode must not inherit this pool"
    assert poisoned.closed is True, "its connections must be released, not leaked"


def test_a_cancelled_episode_also_evicts(monkeypatch):
    """CancelledError is a BaseException; catching Exception would miss it --
    and cancellation is precisely what leaves a reply unread."""

    class _Cancelling(_PoisonedClient):
        async def add_episode(self, **kwargs):
            raise asyncio.CancelledError()

    cancelled = _Cancelling()
    graph = graphiti_client._graph_name_for_client("pokagon")
    graphiti_client._clients.clear()
    graphiti_client._clients[graph] = cancelled

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_episode(monkeypatch, cancelled))

    assert graph not in graphiti_client._clients
    assert cancelled.closed is True


def test_a_successful_episode_keeps_its_client(monkeypatch):
    """Eviction is for failures. Discarding a healthy pool every episode would
    reintroduce the per-request connection churn that leaked in the first place."""

    class _Healthy(_PoisonedClient):
        async def add_episode(self, **kwargs):
            class _R:
                nodes: list = []
                edges: list = []
            return _R()

    healthy = _Healthy()
    graph = graphiti_client._graph_name_for_client("pokagon")
    graphiti_client._clients.clear()
    graphiti_client._clients[graph] = healthy

    asyncio.run(_episode(monkeypatch, healthy))

    assert graphiti_client._clients.get(graph) is healthy, "a good pool is reused"
    assert healthy.closed is False
