"""Pure structural provenance aggregate tests."""

import json

from app.services.provenance_stats import (
    StatsEdge,
    StatsEpisode,
    build_provenance_aggregates,
    provenance_stats_for_graph,
)


EPISODE_IDS = [f"90000000-0000-4000-8000-{index:012d}" for index in range(1, 8)]
EDGE_IDS = [f"91000000-0000-4000-8000-{index:012d}" for index in range(1, 9)]


def _episode(episode_id, **overrides):
    values = {
        "uuid": episode_id,
        "has_name": True,
        "has_source_description": True,
        "source_type": "document",
        "source_id": "doc-456",
        "engagement_id": "engagement-123",
        "episode_type": "document_analysis",
        "anchor_mode": "typed_source",
        "producer_contract_version": "structured_provenance_v2",
        "provenance_write_state": "complete",
    }
    values.update(overrides)
    return StatsEpisode(**values)


def _edge(edge_id, episodes):
    return StatsEdge(uuid=edge_id, episodes=episodes)


def _status_counts(result):
    return {
        row["structural_status"]: row["count"]
        for row in result["by_structural_status"]
    }


def test_aggregates_cover_structural_status_episode_type_and_engagement():
    episodes = [
        _episode(EPISODE_IDS[0]),
        _episode(
            EPISODE_IDS[1],
            episode_type="cross_analysis",
            anchor_mode="pre_chain",
            producer_contract_version="legacy_structured_v1",
            provenance_write_state=None,
        ),
        _episode(
            EPISODE_IDS[2],
            engagement_id="engagement-999",
            producer_contract_version="unknown_producer_v1",
            provenance_write_state=None,
        ),
    ]
    edges = [
        _edge(EDGE_IDS[0], [EPISODE_IDS[0]]),
        _edge(EDGE_IDS[1], [EPISODE_IDS[1]]),
        _edge(EDGE_IDS[2], [EPISODE_IDS[2]]),
        _edge(EDGE_IDS[3], None),
        _edge(EDGE_IDS[4], "not-an-episode-list"),
    ]

    result = build_provenance_aggregates(episodes, edges)

    assert result["facts_total"] == 5
    assert _status_counts(result) == {
        "chained": 1,
        "pre_chain": 2,
        "malformed": 2,
    }
    assert {
        (row["structural_status"], row["episode_type"], row["count"])
        for row in result["by_episode_type"]
    } == {
        ("chained", "document_analysis", 1),
        ("pre_chain", "cross_analysis", 1),
        ("pre_chain", "unresolved", 1),
        ("malformed", "document_analysis", 1),
        ("malformed", "unresolved", 1),
    }
    assert {
        (row["structural_status"], row["engagement_id"], row["count"])
        for row in result["by_engagement"]
    } == {
        ("chained", "engagement-123", 1),
        ("pre_chain", "engagement-123", 1),
        ("pre_chain", "unresolved", 1),
        ("malformed", "engagement-999", 1),
        ("malformed", "unresolved", 1),
    }
    serialized = json.dumps(result)
    for forbidden in ("fact text", "episode name", "source description"):
        assert forbidden not in serialized


def test_structured_v2_requires_complete_write_state_but_other_contracts_do_not():
    episodes = [
        _episode(EPISODE_IDS[0], provenance_write_state="staging"),
        _episode(
            EPISODE_IDS[1],
            producer_contract_version="engage_episode_v2",
            provenance_write_state=None,
        ),
        _episode(
            EPISODE_IDS[2],
            anchor_mode="legacy_episode_v0",
            producer_contract_version="legacy_episode_v0",
            provenance_write_state=None,
        ),
    ]
    edges = [
        _edge(EDGE_IDS[0], [EPISODE_IDS[0]]),
        _edge(EDGE_IDS[1], [EPISODE_IDS[1]]),
        _edge(EDGE_IDS[2], [EPISODE_IDS[2]]),
    ]

    result = build_provenance_aggregates(episodes, edges)

    assert _status_counts(result) == {
        "chained": 2,
        "pre_chain": 1,
        "malformed": 0,
    }


def test_duplicate_episode_identity_cannot_win_but_independent_valid_source_can():
    duplicate_id = EPISODE_IDS[0]
    valid_id = EPISODE_IDS[1]
    episodes = [
        _episode(duplicate_id, engagement_id="engagement-123"),
        _episode(duplicate_id, engagement_id="engagement-999"),
        _episode(valid_id, engagement_id="engagement-123"),
    ]

    duplicate_only = build_provenance_aggregates(
        episodes,
        [_edge(EDGE_IDS[0], [duplicate_id])],
    )
    strongest_valid = build_provenance_aggregates(
        episodes,
        [_edge(EDGE_IDS[1], [duplicate_id, valid_id])],
    )

    assert _status_counts(duplicate_only)["malformed"] == 1
    assert _status_counts(strongest_valid)["chained"] == 1
    assert strongest_valid["by_engagement"] == [
        {
            "structural_status": "chained",
            "engagement_id": "engagement-123",
            "count": 1,
        }
    ]


def test_invalid_id_observations_are_bounded_events_not_fact_counts():
    result = build_provenance_aggregates(
        [StatsEpisode(None, False, False, None, None, None, None, None, None)],
        [StatsEdge(None, None)],
    )

    assert result["facts_total"] == 0
    assert result["malformed_response_events"] == 2
    assert _status_counts(result) == {
        "chained": 0,
        "pre_chain": 0,
        "malformed": 0,
    }


def test_graph_query_projects_only_presence_flags_and_safe_anchor_dimensions():
    class _Result:
        def __init__(self, rows):
            self.result_set = rows

    class _Graph:
        def __init__(self):
            self.queries = []

        def query(self, query, params=None):
            self.queries.append((query, params or {}))
            if "MATCH (episode:Episodic)" in query:
                return _Result(
                    [
                        [
                            EPISODE_IDS[0],
                            True,
                            True,
                            "document",
                            "doc-456",
                            "engagement-123",
                            "document_analysis",
                            "typed_source",
                            "structured_provenance_v2",
                            "complete",
                        ]
                    ]
                )
            return _Result([[EDGE_IDS[0], [EPISODE_IDS[0]]]])

    graph = _Graph()

    result = provenance_stats_for_graph(graph, "client_pokagon")

    assert _status_counts(result)["chained"] == 1
    assert len(graph.queries) == 2
    episode_query = graph.queries[0][0]
    assert "episode.name IS NOT NULL" in episode_query
    assert "trim(episode.name)" in episode_query
    assert "episode.source_description IS NOT NULL" in episode_query
    assert "trim(episode.source_description)" in episode_query
    assert "episode.name," not in episode_query
    assert "episode.source_description," not in episode_query
    assert "episode.content" not in episode_query
    serialized = json.dumps(result)
    assert "name" not in serialized
    assert "description" not in serialized


def test_invalid_dimension_values_are_bucketed_without_echoing_graph_text():
    secret = "SECRET TENANT CONTENT MUST NOT LEAK"
    result = build_provenance_aggregates(
        [
            _episode(
                EPISODE_IDS[0],
                engagement_id=secret,
                episode_type=secret,
            )
        ],
        [_edge(EDGE_IDS[0], [EPISODE_IDS[0]])],
    )

    serialized = json.dumps(result)
    assert secret not in serialized
    assert result["by_episode_type"] == [
        {
            "structural_status": "malformed",
            "episode_type": "unresolved",
            "count": 1,
        }
    ]
    assert result["by_engagement"] == [
        {
            "structural_status": "malformed",
            "engagement_id": "unresolved",
            "count": 1,
        }
    ]
