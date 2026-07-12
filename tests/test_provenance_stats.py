"""Pure structural provenance aggregate tests."""

import json
import re

import pytest

from app.services import graphiti_client, provenance_stats
from app.services.provenance_stats import (
    PROVENANCE_STATS_EDGE_ROW_LIMIT_CODE,
    PROVENANCE_STATS_ENGAGEMENT_BUCKET_LIMIT_CODE,
    PROVENANCE_STATS_EPISODE_ROW_LIMIT_CODE,
    ProvenanceStatsReadError,
    StatsEdge,
    StatsEpisode,
    build_provenance_aggregates,
    provenance_stats_for_graph,
)


EPISODE_IDS = [f"90000000-0000-4000-8000-{index:012d}" for index in range(1, 8)]
EDGE_IDS = [f"91000000-0000-4000-8000-{index:012d}" for index in range(1, 9)]
SUBJECT_ID = "92000000-0000-4000-8000-000000000001"
OBJECT_ID = "92000000-0000-4000-8000-000000000002"


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
        "valid_at": None,
        "provenance_write_state": "complete",
    }
    values.update(overrides)
    return StatsEpisode(**values)


def _edge(edge_id, episodes, **overrides):
    values = {
        "uuid": edge_id,
        "episodes": episodes,
        "subject_uuid": SUBJECT_ID,
        "subject_is_entity": True,
        "has_subject_name": True,
        "has_predicate": True,
        "object_uuid": OBJECT_ID,
        "object_is_entity": True,
        "has_object_name": True,
        "has_fact": True,
        "valid_at": None,
        "invalid_at": None,
        "expired_at": None,
    }
    values.update(overrides)
    return StatsEdge(**values)


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


@pytest.mark.parametrize(
    ("valid_at", "status"),
    [
        (None, "chained"),
        ("2026-07-11T12:00:00Z", "chained"),
        ("not-a-time", "malformed"),
        ("x" * 129, "malformed"),
    ],
)
def test_episode_valid_at_matches_search_source_parity(valid_at, status):
    result = build_provenance_aggregates(
        [_episode(EPISODE_IDS[0], valid_at=valid_at)],
        [_edge(EDGE_IDS[0], [EPISODE_IDS[0]])],
    )

    assert _status_counts(result)[status] == 1


def test_invalid_episode_valid_at_overrides_structured_staging_pre_chain():
    result = build_provenance_aggregates(
        [
            _episode(
                EPISODE_IDS[0],
                valid_at="not-a-time",
                provenance_write_state="staging",
            )
        ],
        [_edge(EDGE_IDS[0], [EPISODE_IDS[0]])],
    )

    assert _status_counts(result)["malformed"] == 1


def test_oversized_episode_valid_at_is_rejected_before_datetime_parse(monkeypatch):
    def forbidden_parse(_value):
        raise AssertionError("oversized source temporal must not be parsed")

    monkeypatch.setattr(provenance_stats, "_parse_dt", forbidden_parse)
    result = build_provenance_aggregates(
        [_episode(EPISODE_IDS[0], valid_at="x" * 1_000_000)],
        [_edge(EDGE_IDS[0], [EPISODE_IDS[0]])],
    )

    assert _status_counts(result)["malformed"] == 1


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


def test_missing_episode_row_is_pre_chain_not_malformed():
    missing_episode_id = "90000000-0000-4000-8000-000000000099"

    result = build_provenance_aggregates(
        [],
        [_edge(EDGE_IDS[0], [missing_episode_id])],
    )

    assert _status_counts(result) == {
        "chained": 0,
        "pre_chain": 1,
        "malformed": 0,
    }
    assert result["by_engagement"] == [
        {
            "structural_status": "pre_chain",
            "engagement_id": "unresolved",
            "count": 1,
        }
    ]


@pytest.mark.parametrize(
    "representation",
    [
        [EPISODE_IDS[0]],
        (EPISODE_IDS[0],),
        f"['{EPISODE_IDS[0]}']",
    ],
)
def test_edge_episode_storage_forms_match_search(representation):
    result = build_provenance_aggregates(
        [_episode(EPISODE_IDS[0])],
        [_edge(EDGE_IDS[0], representation)],
    )

    assert _status_counts(result)["chained"] == 1


def test_duplicate_edge_episode_reference_is_malformed_in_search_and_stats():
    duplicate_list = [EPISODE_IDS[0], EPISODE_IDS[0]]

    assert graphiti_client._episode_uuid_list(duplicate_list) == ((), False)
    result = build_provenance_aggregates(
        [_episode(EPISODE_IDS[0])],
        [_edge(EDGE_IDS[0], duplicate_list)],
    )

    assert _status_counts(result)["malformed"] == 1


def test_stats_episode_character_limit_precedes_utf8_encoding():
    class _HugeString(str):
        def encode(self, *_args, **_kwargs):
            raise AssertionError("oversized string must not be encoded")

    assert provenance_stats._episode_uuid_list(
        _HugeString("x" * 100_001)
    ) == ((), False)


def test_duplicate_episode_group_collapses_to_one_unresolved_sentinel(monkeypatch):
    calls = []
    original = provenance_stats._source_claims_malformed

    def counted(source, *, duplicate):
        calls.append((source, duplicate))
        return original(source, duplicate=duplicate)

    monkeypatch.setattr(provenance_stats, "_source_claims_malformed", counted)
    duplicate_rows = [
        _episode(EPISODE_IDS[0], engagement_id=f"engagement-{index}")
        for index in range(1, 1_001)
    ]

    result = build_provenance_aggregates(
        duplicate_rows,
        [_edge(EDGE_IDS[0], [EPISODE_IDS[0]])],
    )

    assert _status_counts(result)["malformed"] == 1
    assert len(calls) == 1
    assert calls[0][1] is True
    assert result["by_engagement"] == [
        {
            "structural_status": "malformed",
            "engagement_id": "unresolved",
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
            raise AssertionError("stats reads must use ro_query")

        def ro_query(self, query, params=None):
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
                            None,
                            "complete",
                        ]
                    ]
                )
            return _Result(
                [
                    [
                        EDGE_IDS[0],
                        [EPISODE_IDS[0]],
                        SUBJECT_ID,
                        True,
                        True,
                        True,
                        OBJECT_ID,
                        True,
                        True,
                        True,
                        None,
                        None,
                        None,
                    ]
                ]
            )

    graph = _Graph()

    result = provenance_stats_for_graph(graph, "client_pokagon")

    assert _status_counts(result)["chained"] == 1
    assert len(graph.queries) == 2
    assert all("LIMIT 100001" in query for query, _ in graph.queries)
    episode_query = graph.queries[0][0]
    assert "episode.name IS NOT NULL" in episode_query
    assert "trim(episode.name)" in episode_query
    assert "episode.source_description IS NOT NULL" in episode_query
    assert "trim(episode.source_description)" in episode_query
    assert "episode.name," not in episode_query
    assert "episode.source_description," not in episode_query
    assert "episode.content" not in episode_query
    assert "size(trim(episode.name)) <= 2000" in episode_query
    assert "size(trim(episode.source_description)) <= 2000" in episode_query
    assert "size(toString(episode.valid_at))" in episode_query
    assert "$disallowed_control_pattern" in episode_query
    assert "$nonblank_text_pattern" in episode_query
    edge_query = graph.queries[1][0]
    assert "MATCH (subject)-[edge:RELATES_TO]->(object)" in edge_query
    assert "'Entity' IN labels(subject) AS subject_is_entity" in edge_query
    assert "'Entity' IN labels(object) AS object_is_entity" in edge_query
    assert "subject.uuid" in edge_query
    assert "object.uuid" in edge_query
    assert "size(trim(subject.name)) <= 2000" in edge_query
    assert "size(trim(edge.name)) <= 160" in edge_query
    assert "size(trim(object.name)) <= 2000" in edge_query
    assert "size(trim(edge.fact)) <= 16000" in edge_query
    assert "size(toString(edge.valid_at)) <= $temporal_storage_limit" in edge_query
    assert "size(toString(edge.invalid_at)) <= $temporal_storage_limit" in edge_query
    assert "size(toString(edge.expired_at)) <= $temporal_storage_limit" in edge_query
    assert "subject.name," not in edge_query
    assert "object.name," not in edge_query
    assert "edge.name," not in edge_query
    assert "edge.fact," not in edge_query
    for _query, params in graph.queries:
        assert params["group_id"] == "client_pokagon"
        assert "disallowed_control_pattern" in params
        assert "nonblank_text_pattern" in params
        assert params["temporal_storage_limit"] == 128
        assert "oversized_temporal_sentinel" in params
    edge_params = graph.queries[1][1]
    assert edge_params["temporal_storage_limit"] == 128
    assert "oversized_temporal_sentinel" in edge_params
    control_pattern = edge_params["disallowed_control_pattern"]
    assert r"\x09" not in control_pattern
    assert r"\x0A" not in control_pattern
    assert r"\x0D" not in control_pattern
    assert r"\x0B" in control_pattern
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
            "structural_status": "chained",
            "episode_type": "unresolved",
            "count": 1,
        }
    ]
    assert result["by_engagement"] == [
        {
            "structural_status": "chained",
            "engagement_id": "unresolved",
            "count": 1,
        }
    ]


@pytest.mark.parametrize(
    ("value", "maximum", "complete"),
    [
        ("x" * 2_000, 2_000, True),
        ("x" * 2_001, 2_000, False),
        ("\t\n\r", 2_000, False),
        ("\tvalue\n", 2_000, True),
        ("line one\r\nline two", 2_000, True),
        ("\u2003value\u2003", 2_000, True),
        ("value\x00hidden", 2_000, False),
        ("value\x7fhidden", 2_000, False),
    ],
)
def test_stats_text_completeness_matches_search_nonempty_string(
    value, maximum, complete
):
    search_complete = graphiti_client._nonempty_string(value, maximum) is not None
    stats_complete = provenance_stats._structural_text(value, maximum) is not None
    normalized = value.strip()
    projected_flag_model = bool(
        re.fullmatch(provenance_stats._NONBLANK_TEXT_PATTERN, value)
    ) and len(normalized) <= maximum and not bool(
        re.fullmatch(provenance_stats._DISALLOWED_CONTROL_PATTERN, normalized)
    )

    assert search_complete is complete
    assert stats_complete is complete
    assert projected_flag_model is complete


@pytest.mark.parametrize(
    "source_overrides",
    [
        {"has_name": False},
        {"has_source_description": False},
    ],
    ids=["oversize_or_control_name", "oversize_or_control_description"],
)
def test_episode_text_incompleteness_is_malformed(source_overrides):
    result = build_provenance_aggregates(
        [_episode(EPISODE_IDS[0], **source_overrides)],
        [_edge(EDGE_IDS[0], [EPISODE_IDS[0]])],
    )

    assert _status_counts(result)["malformed"] == 1


@pytest.mark.parametrize(
    "edge_overrides",
    [
        {"subject_uuid": "not-a-uuid"},
        {"object_uuid": "not-a-uuid"},
        {"subject_is_entity": False},
        {"object_is_entity": False},
        {"has_subject_name": False},
        {"has_predicate": False},
        {"has_object_name": False},
        {"has_fact": False},
        {"valid_at": "not-a-time"},
        {"invalid_at": "not-a-time"},
        {"expired_at": "not-a-time"},
        {"valid_at": "x" * 129},
    ],
    ids=[
        "subject_uuid",
        "object_uuid",
        "subject_not_entity",
        "object_not_entity",
        "blank_or_oversize_subject_name",
        "blank_or_control_predicate",
        "oversize_or_control_object_name",
        "blank_or_oversize_fact",
        "valid_at",
        "invalid_at",
        "expired_at",
        "oversize_temporal",
    ],
)
def test_edge_wire_incompleteness_is_malformed_before_source_status(edge_overrides):
    result = build_provenance_aggregates(
        [_episode(EPISODE_IDS[0])],
        [_edge(EDGE_IDS[0], [EPISODE_IDS[0]], **edge_overrides)],
    )

    assert _status_counts(result) == {
        "chained": 0,
        "pre_chain": 0,
        "malformed": 1,
    }
    assert result["facts_total"] == 1


def test_clean_edge_wire_with_valid_temporals_remains_chained():
    result = build_provenance_aggregates(
        [_episode(EPISODE_IDS[0])],
        [
            _edge(
                EDGE_IDS[0],
                [EPISODE_IDS[0]],
                valid_at="2026-07-11T12:00:00Z",
                invalid_at="2026-07-12T12:00:00+00:00",
            )
        ],
    )

    assert _status_counts(result)["chained"] == 1


def test_oversized_temporal_is_rejected_before_datetime_parse(monkeypatch):
    def forbidden_parse(_value):
        raise AssertionError("oversized temporal must not reach datetime parsing")

    monkeypatch.setattr(provenance_stats, "_parse_dt", forbidden_parse)

    result = build_provenance_aggregates(
        [_episode(EPISODE_IDS[0])],
        [
            _edge(
                EDGE_IDS[0],
                [EPISODE_IDS[0]],
                valid_at="x" * 1_000_000,
            )
        ],
    )

    assert _status_counts(result)["malformed"] == 1


def test_episode_reference_bound_matches_search_and_never_truncates():
    episodes = [
        _episode(
            f"90000000-0000-4000-8001-{index:012d}",
        )
        for index in range(1, 66)
    ]
    result = build_provenance_aggregates(
        episodes,
        [
            _edge(
                EDGE_IDS[0],
                [episode.uuid for episode in episodes],
            )
        ],
    )

    assert _status_counts(result) == {
        "chained": 0,
        "pre_chain": 0,
        "malformed": 1,
    }


def test_engagement_bucket_bound_fails_instead_of_truncating():
    episodes = [
        _episode(
            f"90000000-0000-4000-8002-{index:012d}",
            engagement_id=f"engagement-{index}",
        )
        for index in range(1, 258)
    ]
    edges = [
        _edge(
            f"91000000-0000-4000-8002-{index:012d}",
            [episode.uuid],
        )
        for index, episode in enumerate(episodes, 1)
    ]

    with pytest.raises(ProvenanceStatsReadError) as failure:
        build_provenance_aggregates(episodes, edges)

    assert failure.value.code == PROVENANCE_STATS_ENGAGEMENT_BUCKET_LIMIT_CODE


@pytest.mark.parametrize(
    ("episode_count", "edge_count", "code"),
    [
        (100_001, 0, PROVENANCE_STATS_EPISODE_ROW_LIMIT_CODE),
        (0, 100_001, PROVENANCE_STATS_EDGE_ROW_LIMIT_CODE),
    ],
)
def test_stats_row_sentinel_hard_fails_without_partial_aggregates(
    episode_count, edge_count, code
):
    class _Result:
        def __init__(self, rows):
            self.result_set = rows

    class _BoundGraph:
        def __init__(self):
            self.queries = []

        def query(self, *_args, **_kwargs):
            raise AssertionError("stats reads must never use query")

        def ro_query(self, query, params=None):
            self.queries.append((query, params or {}))
            if "MATCH (episode:Episodic)" in query:
                return _Result([[None] * 10] * episode_count)
            return _Result([[None] * 2] * edge_count)

    graph = _BoundGraph()

    with pytest.raises(ProvenanceStatsReadError) as failure:
        provenance_stats_for_graph(graph, "client_pokagon")

    assert failure.value.code == code
    assert all("LIMIT 100001" in query for query, _ in graph.queries)
