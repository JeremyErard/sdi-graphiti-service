"""The client graphiti-core writes through must have a finite timeout.

graphiti-core builds its own driver client as FalkorDB(host, port, username,
password), and falkordb.asyncio.FalkorDB defaults socket_timeout and
socket_connect_timeout to None. No timeout at all. A query that never comes back
hangs the coroutine forever: no exception, no completion, no cancellation.

That is why ingestion never finished. Every episode attempted on 2026-08-28/29
ended as "no outcome recorded within 3600s; the task died without reporting" —
not slow, not failing, simply never returning.

#23 gave the SYNCHRONOUS handle a timeout. This is a different client, and the
one the actual ingest writes go through.
"""

import asyncio

from app.config import settings
from app.services import graphiti_client


def test_the_async_driver_client_has_finite_timeouts(monkeypatch):
    captured: dict = {}

    class _CapturingAsyncFalkorDB:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import falkordb.asyncio

    monkeypatch.setattr(falkordb.asyncio, "FalkorDB", _CapturingAsyncFalkorDB)
    graphiti_client.reset_async_falkor_db()
    graphiti_client.get_async_falkor_db()

    # Unbounded is the dangerous value: it makes "slow" mean "never".
    assert captured.get("socket_timeout") is not None
    assert captured.get("socket_connect_timeout") is not None
    assert captured["socket_timeout"] == settings.falkordb_socket_timeout_seconds


def test_the_driver_is_built_with_that_client_not_its_own(monkeypatch):
    """FalkorDriver creates an UNBOUNDED client unless one is injected."""
    sentinel = object()
    seen: dict = {}

    class _CapturingDriver:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(graphiti_client, "FalkorDriver", _CapturingDriver)
    monkeypatch.setattr(graphiti_client, "new_async_falkor_db", lambda: sentinel)

    graphiti_client._create_driver("client_pokagon")

    assert seen.get("falkor_db") is sentinel, "driver must use the injected bounded client"
    assert seen.get("database") == "client_pokagon"
    # host/port must NOT be passed, or the driver builds its own untimed client.
    assert "host" not in seen and "port" not in seen


def test_every_client_is_bounded_however_many_are_built(monkeypatch):
    """What must NOT move: the timeout. Sharing was never the point.

    This assertion used to read `get() is get()`, pinning the client as a
    process-wide singleton. That is the defect: a timeout is a cancelled read,
    the un-read reply stays in the connection buffer, and the next borrower of
    that pooled connection desyncs. One timeout poisoned every later query.

    The invariant worth locking is that no client is ever built unbounded --
    which is what made "slow" mean "never" in the first place.
    """
    built: list = []

    class _AsyncFalkorDB:
        def __init__(self, **kwargs):
            built.append(kwargs)

    import falkordb.asyncio

    monkeypatch.setattr(falkordb.asyncio, "FalkorDB", _AsyncFalkorDB)

    graphiti_client.new_async_falkor_db()
    graphiti_client.new_async_falkor_db()

    assert len(built) == 2, "each call builds its own client"
    for kwargs in built:
        assert kwargs.get("socket_timeout") == settings.falkordb_socket_timeout_seconds
        assert kwargs.get("socket_connect_timeout") is not None


def test_a_failed_episode_does_not_leave_its_pool_behind(monkeypatch):
    """A poisoned client must be discarded, not handed to the next episode."""
    closed: list = []

    class _Client:
        async def close(self):
            closed.append(True)

    graphiti_client._clients.clear()
    graph = graphiti_client._graph_name_for_client("pokagon")
    graphiti_client._clients[graph] = _Client()

    asyncio.run(graphiti_client.evict_client("pokagon"))

    assert graph not in graphiti_client._clients, "cached client must be dropped"
    assert closed == [True], "and closed, so its connections are released"


def test_evicting_survives_a_close_that_itself_fails():
    """Closing a broken pool can raise; eviction must still drop it."""

    class _Client:
        async def close(self):
            raise OSError("pool already broken")

    graphiti_client._clients.clear()
    graph = graphiti_client._graph_name_for_client("pokagon")
    graphiti_client._clients[graph] = _Client()

    asyncio.run(graphiti_client.evict_client("pokagon"))

    assert graph not in graphiti_client._clients


def test_evicting_a_graph_that_was_never_cached_is_a_no_op():
    graphiti_client._clients.clear()
    asyncio.run(graphiti_client.evict_client("pokagon"))
    assert graphiti_client._clients == {}
