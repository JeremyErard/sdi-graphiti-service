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
    monkeypatch.setattr(graphiti_client, "get_async_falkor_db", lambda: sentinel)

    graphiti_client._create_driver("client_pokagon")

    assert seen.get("falkor_db") is sentinel, "driver must reuse the bounded client"
    assert seen.get("database") == "client_pokagon"
    # host/port must NOT be passed, or the driver builds its own untimed client.
    assert "host" not in seen and "port" not in seen


def test_the_client_is_shared_rather_than_rebuilt_per_graph(monkeypatch):
    class _AsyncFalkorDB:
        def __init__(self, **kwargs):
            pass

    import falkordb.asyncio

    monkeypatch.setattr(falkordb.asyncio, "FalkorDB", _AsyncFalkorDB)
    graphiti_client.reset_async_falkor_db()

    assert graphiti_client.get_async_falkor_db() is graphiti_client.get_async_falkor_db()
