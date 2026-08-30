"""G9: enumerate what graphs actually exist.

The census is ratified but has never been run. The last recorded count is
2026-06-15, when 8 graphs were pruned to 3 -- two of them ORPHANS created by
naming drift (`client_test` from a slug mismatch, and a phantom
`client_tribal_gaming` built from an industry string used as a slug). Nothing
since reports what exists, so a repeat is invisible until someone looks.
"""

import asyncio

from app.services import graphiti_client


class _Graph:
    def __init__(self, nodes, edges):
        self._n, self._e = nodes, edges

    def query(self, q):
        class _R:
            pass
        r = _R()
        r.result_set = [[self._n if "(n)" in q else self._e]]
        return r


class _DB:
    def __init__(self, graphs):
        self._g = graphs

    def list_graphs(self):
        return list(self._g)

    def select_graph(self, name):
        return self._g[name]


def _run(monkeypatch, graphs):
    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: _DB(graphs))
    asyncio.run(graphiti_client.log_graph_census())


def test_it_reports_every_graph_with_counts(monkeypatch, caplog):
    graphs = {"client_pokagon": _Graph(4204, 15288), "client_mrla": _Graph(412, 1271)}
    with caplog.at_level("WARNING"):
        _run(monkeypatch, graphs)
    msgs = " | ".join(r.message for r in caplog.records)
    assert "client_pokagon: nodes=4204 edges=15288" in msgs
    assert "client_mrla: nodes=412 edges=1271" in msgs
    assert "total_graphs=2" in msgs


def test_an_empty_graph_is_called_out_not_just_listed(monkeypatch, caplog):
    """A graph holding nothing is either fresh or an orphan, and those are
    indistinguishable from a count alone."""
    with caplog.at_level("WARNING"):
        _run(monkeypatch, {"client_ghost": _Graph(0, 0)})
    assert any("EMPTY" in r.message for r in caplog.records)


def test_a_populated_graph_is_not_flagged_empty(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        _run(monkeypatch, {"client_real": _Graph(10, 20)})
    assert any("populated" in r.message for r in caplog.records)
    assert not any("EMPTY" in r.message for r in caplog.records)


def test_the_census_never_blocks_startup(monkeypatch, caplog):
    """It runs in the boot path. A census that can fail a deploy is a liability."""
    class _Exploding:
        def list_graphs(self):
            raise RuntimeError("falkordb unreachable")

    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: _Exploding())
    with caplog.at_level("WARNING"):
        asyncio.run(graphiti_client.log_graph_census())  # must not raise
    assert any("CENSUS unavailable" in r.message for r in caplog.records)


def test_no_graphs_is_reported_rather_than_silent(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        _run(monkeypatch, {})
    assert any("no graphs found" in r.message for r in caplog.records)
