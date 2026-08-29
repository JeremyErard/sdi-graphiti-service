"""A failed ingest must say WHICH query was slow.

Two days of this outage were spent inferring the responsible Cypher statement
from failure DURATIONS. Entity dedup was the leading candidate on
circumstantial evidence, and indexing it did not change the outcome -- which is
what a guess looks like when it is wrong. FalkorDB keeps a per-graph slow log.
"""

import asyncio

from app.config import settings
from app.services import graphiti_client


class _Graph:
    def __init__(self, entries=None, raises=False):
        self.entries = entries or []
        self.raises = raises

    def slowlog(self):
        if self.raises:
            raise RuntimeError("slowlog unsupported")
        return self.entries


class _DB:
    def __init__(self, graph):
        self._g = graph

    def select_graph(self, name):
        return self._g


def _run(monkeypatch, graph):
    def _read(graph_name):
        return graph.slowlog()

    monkeypatch.setattr(graphiti_client, "_read_slowlog", _read)
    asyncio.run(graphiti_client._log_slow_queries("client_pokagon"))


def test_it_reports_the_slowest_query(monkeypatch, caplog):
    graph = _Graph([["1", "GRAPH.QUERY", "MATCH (n:Entity) ...", "412000.5"]])
    with caplog.at_level("WARNING"):
        _run(monkeypatch, graph)
    assert any("MATCH (n:Entity)" in r.message for r in caplog.records)


def test_it_ranks_by_duration_not_recency(monkeypatch, caplog):
    """The newest query is rarely the expensive one."""
    graph = _Graph([
        ["1", "GRAPH.QUERY", "THE SLOW ONE", "900000.0"],
        ["2", "GRAPH.QUERY", "a fast one", "3.0"],
    ])
    with caplog.at_level("WARNING"):
        _run(monkeypatch, graph)
    slow = [r for r in caplog.records if "SLOW" in r.message]
    assert "THE SLOW ONE" in slow[0].message


def test_an_empty_slowlog_says_so_rather_than_staying_silent(monkeypatch, caplog):
    """Silence would read as 'nothing was slow', which is the wrong conclusion:
    it means the cost is not one logged query."""
    with caplog.at_level("WARNING"):
        _run(monkeypatch, _Graph([]))
    assert any("EMPTY" in r.message for r in caplog.records)


def test_diagnostics_never_replace_the_real_error(monkeypatch, caplog):
    """This runs on an already-failing path. It must not raise."""
    with caplog.at_level("WARNING"):
        _run(monkeypatch, _Graph(raises=True))  # must not raise
    assert any("slowlog unavailable" in r.message for r in caplog.records)


def test_the_slowlog_read_never_waits_as_long_as_the_call_that_failed():
    """It reads through its OWN short-timeout client, not the shared 900s one.

    The first version reused the shared handle. On an already-failed FalkorDB
    call that meant waiting another fifteen minutes on the dependency that had
    just died, and the task was torn down before logging anything -- so the
    diagnostic produced exactly nothing in the one case it existed for.
    """
    assert graphiti_client.SLOWLOG_TIMEOUT_SECONDS <= 30
    assert graphiti_client.SLOWLOG_TIMEOUT_SECONDS < settings.falkordb_socket_timeout_seconds


def test_a_hanging_slowlog_still_reports_rather_than_hanging(monkeypatch, caplog):
    """If the slowlog read itself blocks, say so; never inherit the hang."""

    def _hang(graph_name):
        raise TimeoutError("Timeout reading from falkordb")

    monkeypatch.setattr(graphiti_client, "_read_slowlog", _hang)
    with caplog.at_level("WARNING"):
        asyncio.run(graphiti_client._log_slow_queries("client_pokagon"))
    assert any("slowlog unavailable" in r.message for r in caplog.records)
