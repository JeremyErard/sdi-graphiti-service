"""FalkorDB has no =~ operator; the provenance reads must not use it.

The first live ``include_provenance=true`` call against the Pokagon graph
(2026-09-10) failed with "FalkorDB does not currently support =~". Every
regex check in the two stats queries goes through string.matchRegEx instead,
with the same parameters, so the audit surface the Phase 2 substrate
milestone depends on can run against the deployed database.
"""

from app.services import provenance_stats


class _Result:
    def __init__(self, rows):
        self.result_set = rows


class _Graph:
    def __init__(self):
        self.queries = []

    def ro_query(self, query, params=None):
        self.queries.append((query, params or {}))
        return _Result([])


def test_the_stats_queries_use_matchRegEx_and_never_the_operator():
    graph = _Graph()
    provenance_stats.provenance_stats_for_graph(graph, "client_pokagon")

    assert len(graph.queries) == 2
    for query, params in graph.queries:
        assert "=~" not in query
        assert "string.matchRegEx(" in query
        assert "$nonblank_text_pattern" in query
        assert "$disallowed_control_pattern" in query
        assert params["nonblank_text_pattern"] == provenance_stats._NONBLANK_TEXT_PATTERN
        assert params["disallowed_control_pattern"] == provenance_stats._DISALLOWED_CONTROL_PATTERN
    episode_query = graph.queries[0][0]
    assert "size(string.matchRegEx(trim(episode.name), $nonblank_text_pattern)) > 0" in episode_query
    # The raw column is never projected: the guard the existing suite keeps.
    for field in ("episode.name,", "episode.source_description,"):
        assert field not in episode_query
    assert "size(string.matchRegEx(trim(episode.name), $disallowed_control_pattern)) > 0" in episode_query
