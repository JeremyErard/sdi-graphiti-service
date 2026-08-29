"""A synchronous FalkorDB call must not be able to stop the whole service.

`get_falkor_db()` returns a SYNCHRONOUS handle, and it is used from inside async
request paths — including once per episode on the ingest path, to write
provenance anchors. Two things followed from that and both were real.

It had NO socket timeout, so a call against a slow or wedged FalkorDB blocked
the event loop indefinitely. And it ran inline, so even a merely slow call was
charged to every other request.

Observed 2026-08-28: this service stopped answering /health — a 3s redis ping —
for over twenty minutes while a single extraction was in progress. It had not
crashed and had not restarted. Nothing could be scheduled at all.
"""

import asyncio
import time
from datetime import datetime, timezone

from app.config import settings
from app.services import graphiti_client


class _SlowGraph:
    """A graph whose sync query takes real wall-clock time, like a loaded one."""

    def __init__(self, delay: float):
        self.delay = delay
        self.result_set = [["ep-uuid"]]

    def query(self, *_args, **_kwargs):
        time.sleep(self.delay)
        return self


class _SlowDb:
    def __init__(self, delay: float):
        self.graph = _SlowGraph(delay)

    def select_graph(self, _name: str):
        return self.graph


class _FakeEpisode:
    uuid = "11111111-1111-4111-8111-111111111111"


class _FakeResult:
    episode = _FakeEpisode()
    nodes: list = []
    edges: list = []


class _FakeClient:
    async def add_episode(self, **_kwargs):
        return _FakeResult()


def test_the_shared_handle_has_a_finite_socket_timeout(monkeypatch):
    """Unbounded is the dangerous value: it makes 'slow' mean 'forever'."""
    captured: dict = {}

    class _CapturingFalkorDB:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import falkordb

    monkeypatch.setattr(falkordb, "FalkorDB", _CapturingFalkorDB)
    graphiti_client._falkor_db = None
    graphiti_client.get_falkor_db()

    assert captured.get("socket_timeout"), "socket_timeout must be set"
    assert captured.get("socket_connect_timeout"), "socket_connect_timeout must be set"
    assert captured["socket_timeout"] == settings.falkordb_socket_timeout_seconds
    assert isinstance(captured["socket_timeout"], int)


def test_the_provenance_write_does_not_freeze_the_event_loop(monkeypatch):
    """The loop must keep running while the sync anchor write happens.

    This is the property that actually matters. A blocked loop is not a slow
    endpoint — it is a service that answers nothing, including its own health
    check, which is exactly what was observed.
    """
    async def _get_client(_slug):
        return _FakeClient()

    monkeypatch.setattr(graphiti_client, "get_client", _get_client)
    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: _SlowDb(0.4))

    async def scenario() -> int:
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            await graphiti_client.add_episode(
                client_slug="pokagon",
                engagement_id="eng-1",
                name="document_analysis: document/doc-1",
                content="Finance owns the monthly close.",
                source_description="test",
                reference_time=datetime.now(timezone.utc),
                source_id="doc-1",
                source_type="document",
                episode_type="document_analysis",
                anchor_mode="typed_source",
                producer_contract_version="engage_episode_v2",
            )
        finally:
            beat.cancel()
        return ticks

    ticks = asyncio.run(scenario())

    # Inline, the 0.4s query would have blocked every one of these.
    assert ticks >= 5, f"event loop was starved during the sync write (ticks={ticks})"




def test_the_search_path_does_not_freeze_the_event_loop(monkeypatch):
    """Search, not just ingest, must stay off the loop.

    The second freeze on 2026-08-28 happened with NO ingestion running at all.
    The kg-health cron calls /search/context periodically, which goes through
    _search_fast and the same synchronous handle. One slow read there stops the
    whole service, including the health check that would have reported it.
    """
    slow = _SlowDb(0.4)
    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: slow)
    monkeypatch.setattr(graphiti_client, "_ensure_edge_vector_index", lambda *_a, **_k: None)
    monkeypatch.setattr(graphiti_client, "_row_to_edge", lambda row: row)
    monkeypatch.setattr(graphiti_client, "_lucene_sanitize", lambda q: "")

    class _Embedder:
        async def create(self, input_data):
            return [0.0] * 8

    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: _Embedder())

    async def scenario() -> int:
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            await graphiti_client._search_fast("pokagon", "monthly close", 5)
        finally:
            beat.cancel()
        return ticks

    ticks = asyncio.run(scenario())
    assert ticks >= 5, f"event loop was starved during the search read (ticks={ticks})"


def test_the_graph_visualisation_endpoint_does_not_freeze_the_loop(monkeypatch):
    """/graph/nodes-and-edges must not stop the world either.

    Not hypothetical. On 2026-08-29 a single operator call to this endpoint —
    made to check whether the graph contained anything — froze the service and
    destroyed a 28-minute ingestion in progress. It was reading through the
    synchronous handle inline.

    "Bounded by the 120s socket timeout, so not urgent" was the wrong call: 120
    seconds of frozen loop is long enough to kill a long-running extraction, and
    any client or operator request can trigger it.

    The handler is awaited DIRECTLY on the test's own loop. Driving it through
    TestClient would prove nothing — TestClient runs the app on a loop of its
    own, so a blocking handler would not starve this one.
    """
    from app.routers import graph as graph_router

    monkeypatch.setattr(graph_router.graphiti_client, "get_falkor_db", lambda: _SlowDb(0.4))

    async def scenario() -> int:
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            req = graph_router.GraphDataRequest(client_slug="pokagon", max_nodes=10)
            await graph_router.get_graph_data(req)
        except Exception:
            pass  # the fake graph returns no usable rows; only scheduling matters here
        finally:
            beat.cancel()
        return ticks

    ticks = asyncio.run(scenario())
    assert ticks >= 5, f"event loop was starved by the graph endpoint (ticks={ticks})"
