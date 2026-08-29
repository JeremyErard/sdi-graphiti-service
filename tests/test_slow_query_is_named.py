"""A failed ingest must say WHICH query was slow.

Two days of this outage were spent inferring the responsible Cypher statement
from failure DURATIONS. Entity dedup was the leading candidate on
circumstantial evidence, and indexing it did not change the outcome -- which is
what a guess looks like when it is wrong. FalkorDB keeps a per-graph slow log.
"""

import asyncio

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
    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: _DB(graph))
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
