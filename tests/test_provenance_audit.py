"""Dormant provenance audit/backfill safety and idempotency contracts."""

import json

import pytest

from app.services import provenance_ops
from app.services.provenance_ops import (
    APPLY_BLOCKED_CODE,
    AUDIT_EDGE_ROW_LIMIT_CODE,
    AUDIT_EPISODE_ROW_LIMIT_CODE,
    AUDIT_GRAPH_NOT_FOUND_CODE,
    ApplyBlockedError,
    EdgeRecord,
    EpisodeRecord,
    ProvenanceAuditReadError,
    _apply_plan,
    build_provenance_plan,
    normalize_provable_episode_list,
    parse_exact_legacy_anchor,
    run_provenance_audit,
)
from scripts import provenance_audit


EPISODE_ID = "70000000-0000-4000-8000-000000000001"
EDGE_ID = "71000000-0000-4000-8000-000000000001"
SUBJECT_ID = "72000000-0000-4000-8000-000000000001"
OBJECT_ID = "72000000-0000-4000-8000-000000000002"
NAME = "document_analysis: document/doc-456"
DESCRIPTION = "Engagement engagement-123 — document_analysis from document"


def _episode(**overrides):
    values = {
        "uuid": EPISODE_ID,
        "name": NAME,
        "source_description": DESCRIPTION,
    }
    values.update(overrides)
    return EpisodeRecord(**values)


def _edge(**overrides):
    values = {
        "uuid": EDGE_ID,
        "source_uuid": None,
        "target_uuid": None,
        "actual_source_uuid": SUBJECT_ID,
        "actual_target_uuid": OBJECT_ID,
        "episodes": f"['{EPISODE_ID}']",
    }
    values.update(overrides)
    return EdgeRecord(**values)


def test_exact_legacy_formats_produce_only_the_legacy_signature():
    anchor, code = parse_exact_legacy_anchor(NAME, DESCRIPTION)

    assert code == "EPISODE_CANONICAL"
    assert anchor is not None
    assert anchor.source_id == "doc-456"
    assert anchor.source_type == "document"
    assert anchor.engagement_id == "engagement-123"
    assert anchor.episode_type == "document_analysis"
    assert anchor.anchor_mode == "legacy_episode_v0"
    assert anchor.producer_contract_version == "legacy_episode_v0"


@pytest.mark.parametrize(
    ("name", "description", "code"),
    [
        (
            NAME,
            "Engagement engagement-123 - document_analysis from document",
            "EPISODE_UNRESOLVED_FORMAT",
        ),
        (
            "document_analysis: document/doc-456 ",
            DESCRIPTION,
            "EPISODE_UNRESOLVED_FORMAT",
        ),
        (
            "document_analysis: document/doc-456",
            "Engagement engagement-123 — document_analysis from interview",
            "EPISODE_UNRESOLVED_MISMATCH",
        ),
        (
            "invented_type: document/doc-456",
            "Engagement engagement-123 — invented_type from document",
            "EPISODE_UNRESOLVED_EPISODE_TYPE",
        ),
        (
            "document_analysis: document/doc/456",
            DESCRIPTION,
            "EPISODE_UNRESOLVED_FORMAT",
        ),
    ],
)
def test_spoof_and_near_miss_legacy_formats_remain_unresolved(
    name, description, code
):
    anchor, observed_code = parse_exact_legacy_anchor(name, description)

    assert anchor is None
    assert observed_code == code


def test_ambiguous_duplicate_episode_identity_is_never_backfilled():
    plan = build_provenance_plan(
        [_episode(), _episode(name="document_analysis: document/doc-other")],
        [],
    )

    assert plan.episode_updates == ()
    assert plan.codes == {"EPISODE_UNRESOLVED_DUPLICATE_ID": 1}


def test_existing_anchor_conflict_is_preserved_as_unresolved():
    plan = build_provenance_plan(
        [_episode(source_id="different-source")],
        [],
    )

    assert plan.episode_updates == ()
    assert plan.codes == {"EPISODE_UNRESOLVED_ANCHOR_CONFLICT": 1}


def test_only_provable_existing_episode_lists_are_normalized():
    known = frozenset({EPISODE_ID})

    assert normalize_provable_episode_list(
        f"['{EPISODE_ID}']", known_episode_ids=known
    ) == ((EPISODE_ID,), False)
    assert normalize_provable_episode_list(
        [EPISODE_ID], known_episode_ids=known
    ) == ((EPISODE_ID,), True)
    assert normalize_provable_episode_list(
        "not-a-list", known_episode_ids=known
    ) == (None, False)
    assert normalize_provable_episode_list(
        (EPISODE_ID,), known_episode_ids=known
    ) == ((EPISODE_ID,), False)
    assert normalize_provable_episode_list(
        "['70000000-0000-4000-8000-000000000099']",
        known_episode_ids=known,
    ) == (None, False)
    assert normalize_provable_episode_list(
        [EPISODE_ID] * 65,
        known_episode_ids=known,
    ) == (None, False)
    assert normalize_provable_episode_list(
        " " * 100_001,
        known_episode_ids=known,
    ) == (None, False)


def test_episode_storage_limit_is_utf8_bytes_and_precedes_parsing(monkeypatch):
    called = False

    def forbidden_parse(_value):
        nonlocal called
        called = True
        raise AssertionError("oversized storage must not be parsed")

    monkeypatch.setattr(provenance_ops.ast, "literal_eval", forbidden_parse)

    assert normalize_provable_episode_list(
        "é" * 50_001,
        known_episode_ids=frozenset({EPISODE_ID}),
    ) == (None, False)
    assert called is False


def test_plan_repairs_derivable_endpoints_and_converges_idempotently():
    first = build_provenance_plan([_episode()], [_edge()])

    assert len(first.episode_updates) == 1
    assert len(first.edge_updates) == 1
    assert first.edge_updates[0].set_endpoints is True
    assert first.edge_updates[0].set_episodes is True

    converged_episode = _episode(
        source_id="doc-456",
        source_type="document",
        engagement_id="engagement-123",
        episode_type="document_analysis",
        anchor_mode="legacy_episode_v0",
        producer_contract_version="legacy_episode_v0",
    )
    converged_edge = _edge(
        source_uuid=SUBJECT_ID,
        target_uuid=OBJECT_ID,
        episodes=[EPISODE_ID],
    )
    second = build_provenance_plan([converged_episode], [converged_edge])

    assert second.episode_updates == ()
    assert second.edge_updates == ()
    assert second.codes["EPISODE_ALREADY_ANCHORED"] == 1
    assert second.codes["EDGE_ENDPOINT_ALREADY_REPAIRED"] == 1
    assert second.codes["EDGE_EPISODE_LIST_ALREADY_NORMALIZED"] == 1


class _Result:
    def __init__(self, rows):
        self.result_set = rows


class _Graph:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def query(self, query, params=None):
        self.calls.append((query, params or {}))
        if "SET episode.source_id" in query or "SET edge.source_uuid" in query:
            return _Result([[1]])
        raise AssertionError("audit reads must use ro_query")

    def ro_query(self, query, params=None):
        self.calls.append((query, params or {}))
        if "RETURN episode.uuid, episode.name" in query:
            return _Result(
                [[EPISODE_ID, NAME, DESCRIPTION, None, None, None, None, None, None]]
            )
        if "RETURN edge.uuid, edge.source_uuid" in query:
            return _Result(
                [[EDGE_ID, None, None, SUBJECT_ID, OBJECT_ID, f"['{EPISODE_ID}']"]]
            )
        raise AssertionError("unexpected provenance audit query")


class _DB:
    def __init__(self, graph):
        self.graph = graph
        self.selected: list[str] = []

    def select_graph(self, graph_name):
        self.selected.append(graph_name)
        return self.graph

    def list_graphs(self):
        return ["client_pokagon"]


def test_default_audit_mode_performs_no_mutation_and_emits_no_graph_values():
    graph = _Graph()
    db = _DB(graph)

    result = run_provenance_audit(
        "pokagon",
        db_factory=lambda **_kwargs: db,
    )

    assert result["mode"] == "audit"
    assert result["counts"] == {
        "episodes_scanned": 1,
        "edges_scanned": 1,
        "episode_anchor_updates_planned": 1,
        "edge_endpoint_updates_planned": 1,
        "edge_episode_list_updates_planned": 1,
        "apply_attempted": 0,
        "apply_succeeded": 0,
        "apply_conflicts": 0,
    }
    assert len(graph.calls) == 2
    assert all("LIMIT 100001" in query for query, _ in graph.calls)
    assert not any(" SET " in " ".join(query.split()) for query, _ in graph.calls)
    serialized = json.dumps(result)
    for forbidden in (NAME, DESCRIPTION, "doc-456", EPISODE_ID, EDGE_ID):
        assert forbidden not in serialized


def test_apply_mode_is_explicit_and_applies_only_the_planned_records():
    graph = _Graph()
    plan = build_provenance_plan([_episode()], [_edge()])

    succeeded, conflicts = _apply_plan(
        graph,
        "client_pokagon",
        plan,
    )
    result = plan.summary(
        apply=True,
        apply_succeeded=succeeded,
        apply_conflicts=conflicts,
    )

    assert result["mode"] == "apply"
    assert result["counts"]["apply_attempted"] == 2
    assert result["counts"]["apply_succeeded"] == 2
    assert result["counts"]["apply_conflicts"] == 0
    mutation_queries = [query for query, _ in graph.calls if " SET " in f" {query} "]
    assert len(mutation_queries) == 2


def test_duplicate_appearing_at_apply_is_cardinality_blocked_before_mutation():
    class _DuplicateAtApplyGraph(_Graph):
        def __init__(self):
            super().__init__()
            self.mutated = 0

        def query(self, query, params=None):
            if " SET " in f" {query} ":
                self.calls.append((query, params or {}))
                assert "count(candidate) AS cardinality" in query
                assert "cardinality = 1" in query
                # The fake models a duplicate appearing after the audit snapshot:
                # the cardinality WHERE removes every row before SET executes.
                return _Result([[0]])
            return super().query(query, params)

    graph = _DuplicateAtApplyGraph()
    plan = build_provenance_plan([_episode()], [_edge()])

    succeeded, conflicts = _apply_plan(
        graph,
        "client_pokagon",
        plan,
    )
    result = plan.summary(
        apply=True,
        apply_succeeded=succeeded,
        apply_conflicts=conflicts,
    )

    assert result["counts"]["apply_attempted"] == 2
    assert result["counts"]["apply_succeeded"] == 0
    assert result["counts"]["apply_conflicts"] == 2
    assert graph.mutated == 0


def test_cli_defaults_to_audit_and_requires_one_exact_slug_for_apply(monkeypatch, capsys):
    calls: list[tuple[str, bool]] = []

    def fake_run(slug, *, apply=False):
        calls.append((slug, apply))
        return {"mode": "apply" if apply else "audit", "counts": {}, "codes": {}}

    monkeypatch.setattr(provenance_audit, "run_provenance_audit", fake_run)

    assert provenance_audit.main(["pokagon"]) == 0
    assert provenance_audit.main(["pokagon", "--apply"]) == 0
    assert calls == [("pokagon", False), ("pokagon", True)]
    assert "pokagon" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        provenance_audit.build_parser().parse_args(["--apply"])
    with pytest.raises(SystemExit):
        provenance_audit.build_parser().parse_args(["pokagon", "other"])
    with pytest.raises(SystemExit):
        provenance_audit.build_parser().parse_args(["pokagon", "--app"])


def test_cli_apply_exits_nonzero_when_ambiguity_remains(monkeypatch):
    monkeypatch.setattr(
        provenance_audit,
        "run_provenance_audit",
        lambda _slug, *, apply=False: {
            "mode": "apply",
            "counts": {"apply_conflicts": 0},
            "codes": {"EPISODE_UNRESOLVED_DUPLICATE_ID": 1},
        },
    )

    assert provenance_audit.main(["pokagon", "--apply"]) == 2


def test_real_apply_entrypoint_is_blocked_before_database_access(capsys):
    called = False

    def forbidden_db(**_kwargs):
        nonlocal called
        called = True

    with pytest.raises(ApplyBlockedError) as failure:
        run_provenance_audit(
            "pokagon",
            apply=True,
            db_factory=forbidden_db,
        )
    assert str(failure.value) == APPLY_BLOCKED_CODE
    assert called is False

    assert provenance_audit.main(["pokagon", "--apply"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "mode": "apply",
        "counts": {},
        "codes": {APPLY_BLOCKED_CODE: 1},
    }


def test_cli_read_failure_emits_only_fixed_code(monkeypatch, capsys):
    def fail_read(*_args, **_kwargs):
        raise ProvenanceAuditReadError(AUDIT_GRAPH_NOT_FOUND_CODE)

    monkeypatch.setattr(provenance_audit, "run_provenance_audit", fail_read)

    assert provenance_audit.main(["pokagon"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "mode": "audit",
        "counts": {},
        "codes": {AUDIT_GRAPH_NOT_FOUND_CODE: 1},
    }


def test_invalid_slug_is_rejected_before_graph_access():
    called = False

    def forbidden_db(**_kwargs):
        nonlocal called
        called = True

    with pytest.raises(ValueError):
        run_provenance_audit("*", apply=True, db_factory=forbidden_db)
    assert called is False


def test_missing_graph_is_fixed_code_and_never_selected():
    graph = _Graph()
    db = _DB(graph)
    db.list_graphs = lambda: ["client_other"]

    with pytest.raises(ProvenanceAuditReadError) as failure:
        run_provenance_audit(
            "pokagon",
            db_factory=lambda **_kwargs: db,
        )

    assert failure.value.code == AUDIT_GRAPH_NOT_FOUND_CODE
    assert db.selected == []
    assert graph.calls == []


@pytest.mark.parametrize(
    ("episode_count", "edge_count", "code"),
    [
        (100_001, 0, AUDIT_EPISODE_ROW_LIMIT_CODE),
        (0, 100_001, AUDIT_EDGE_ROW_LIMIT_CODE),
    ],
)
def test_audit_row_sentinel_hard_fails_without_a_partial_plan(
    episode_count, edge_count, code
):
    class _BoundGraph:
        def __init__(self):
            self.calls = []

        def query(self, *_args, **_kwargs):
            raise AssertionError("audit reads must never use query")

        def ro_query(self, query, params=None):
            self.calls.append((query, params or {}))
            if "MATCH (episode:Episodic)" in query:
                return _Result([[None] * 9] * episode_count)
            return _Result([[None] * 6] * edge_count)

    graph = _BoundGraph()
    db = _DB(graph)

    with pytest.raises(ProvenanceAuditReadError) as failure:
        run_provenance_audit(
            "pokagon",
            db_factory=lambda **_kwargs: db,
        )

    assert failure.value.code == code
    assert all("LIMIT 100001" in query for query, _ in graph.calls)
