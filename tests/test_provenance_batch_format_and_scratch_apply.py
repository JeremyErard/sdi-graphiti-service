"""The 2026-04-24 batch extraction gets an anchor rule, and apply gets a proving ground.

Measured 2026-09-11 on the Pokagon graph: 106 of 343 episodes were written by
that batch (99 SOPs, 6 reports, 1 map) in a format the legacy parser does not
read, and 2,716 of 7,757 facts hang off them. Their descriptions carry the
source id; the engagement they lack is supplied by the operator, verified
against the tenant database. Apply is allowed only on a scratch copy of a
tenant graph, where the singleton-conditional mutation is proven against the
deployed FalkorDB; tenant graphs stay locked until that proof is recorded.
"""

import pytest

from app.services.provenance_ops import (
    APPLY_BLOCKED_CODE,
    AUDIT_GRAPH_NOT_FOUND_CODE,
    ApplyBlockedError,
    EdgeRecord,
    EpisodeRecord,
    ProvenanceAuditReadError,
    build_provenance_plan,
    parse_batch_extracted_anchor,
    parse_exact_legacy_anchor,
    run_provenance_audit,
)

ENGAGEMENT = "cmmnn76l60000lqzwaua2zed7"
EPISODE_ID = "70000000-0000-4000-8000-000000000011"
EDGE_ID = "71000000-0000-4000-8000-000000000011"
SUBJECT_ID = "72000000-0000-4000-8000-000000000011"
OBJECT_ID = "72000000-0000-4000-8000-000000000012"


def test_the_three_batch_kinds_anchor_to_what_the_tenant_database_holds():
    sop, code = parse_batch_extracted_anchor("SOP: C&E - Table Inventory Audit v2", "Approved artifact (batch-extracted): sop:cmsop123", ENGAGEMENT)
    assert code == "EPISODE_CANONICAL_BATCH"
    assert (sop.episode_type, sop.source_type, sop.source_id, sop.engagement_id) == ("sop_approved", "sop", "cmsop123", ENGAGEMENT)

    map_anchor, _ = parse_batch_extracted_anchor("Map: C&E Game Protection Inspection v4", "Approved artifact (batch-extracted): map:cmpv456", ENGAGEMENT)
    assert (map_anchor.episode_type, map_anchor.source_type, map_anchor.source_id) == ("process_map_approved", "process_version", "cmpv456")

    report, _ = parse_batch_extracted_anchor("Report: 2026-04-17", "Approved artifact (batch-extracted): report:cmir789", ENGAGEMENT)
    assert (report.episode_type, report.source_type, report.source_id) == ("insight_report", "insight_report", "cmir789")


@pytest.mark.parametrize(
    ("name", "description", "engagement", "expected"),
    [
        ("Map: something", "Approved artifact (batch-extracted): sop:cmsop123", ENGAGEMENT, "EPISODE_UNRESOLVED_MISMATCH"),
        ("SOP: something", "Approved artifact (batch-extracted): sop:cmsop123", "", "EPISODE_UNRESOLVED_FORMAT"),
        ("SOP: something", "Approved artifact (batch-extracted): sop:cmsop123", "not valid!", "EPISODE_UNRESOLVED_FORMAT"),
        ("SOP: something", "Approved artifact (batch-extracted): policy:cmsop123", ENGAGEMENT, "EPISODE_UNRESOLVED_FORMAT"),
        ("SOP: something", "Approved artifact (batch-extracted): sop:cmsop123 extra", ENGAGEMENT, "EPISODE_UNRESOLVED_FORMAT"),
        ("document_analysis: document/doc-456", "Engagement e — document_analysis from document", ENGAGEMENT, "EPISODE_UNRESOLVED_FORMAT"),
    ],
)
def test_near_misses_stay_unresolved(name, description, engagement, expected):
    anchor, code = parse_batch_extracted_anchor(name, description, engagement)
    assert anchor is None
    assert code == expected


def test_the_legacy_parser_is_unchanged_by_the_batch_rule():
    anchor, code = parse_exact_legacy_anchor("SOP: C&E - Table Inventory Audit v2", "Approved artifact (batch-extracted): sop:cmsop123")
    assert anchor is None and code == "EPISODE_UNRESOLVED_FORMAT"


def _batch_episode():
    return EpisodeRecord(uuid=EPISODE_ID, name="SOP: C&E - Table Inventory Audit v2", source_description="Approved artifact (batch-extracted): sop:cmsop123")


def _edge():
    return EdgeRecord(uuid=EDGE_ID, source_uuid=None, target_uuid=None, actual_source_uuid=SUBJECT_ID, actual_target_uuid=OBJECT_ID, episodes=f"['{EPISODE_ID}']")


def test_the_plan_anchors_batch_episodes_only_when_the_engagement_is_supplied():
    without = build_provenance_plan([_batch_episode()], [_edge()])
    assert without.codes.get("EPISODE_UNRESOLVED_FORMAT") == 1
    assert len(without.episode_updates) == 0

    with_engagement = build_provenance_plan([_batch_episode()], [_edge()], batch_engagement_id=ENGAGEMENT)
    assert with_engagement.codes.get("EPISODE_ANCHOR_UPDATE_READY") == 1
    assert "EPISODE_UNRESOLVED_FORMAT" not in with_engagement.codes
    (update,) = with_engagement.episode_updates
    assert update.uuid == EPISODE_ID
    assert (update.anchor.episode_type, update.anchor.source_id, update.anchor.engagement_id) == ("sop_approved", "cmsop123", ENGAGEMENT)
    # The edge still normalizes its episode list against the now-anchorable episode.
    assert with_engagement.codes.get("EDGE_EPISODE_LIST_UPDATE_READY") == 1


class _Result:
    def __init__(self, rows):
        self.result_set = rows


class _Graph:
    def __init__(self):
        self.reads = []
        self.writes = []

    def ro_query(self, query, params=None):
        self.reads.append((query, params or {}))
        if "MATCH (episode:Episodic)" in query:
            return _Result([[EPISODE_ID, "SOP: C&E - Table Inventory Audit v2", "Approved artifact (batch-extracted): sop:cmsop123", None, None, None, None, None, None]])
        return _Result([[EDGE_ID, None, None, SUBJECT_ID, OBJECT_ID, f"['{EPISODE_ID}']"]])

    def query(self, query, params=None):
        self.writes.append((query, params or {}))
        return _Result([[1]])


class _DB:
    def __init__(self, graph, graphs):
        self.graph = graph
        self.graphs = graphs

    def list_graphs(self):
        return self.graphs

    def select_graph(self, name):
        self.selected = name
        return self.graph


def test_apply_on_a_tenant_graph_stays_blocked_before_any_read():
    graph = _Graph()
    db = _DB(graph, ["client_pokagon"])
    with pytest.raises(ApplyBlockedError) as caught:
        run_provenance_audit("pokagon", apply=True, db_factory=lambda **_k: db)
    assert str(caught.value) == APPLY_BLOCKED_CODE
    assert graph.reads == [] and graph.writes == []


def test_apply_on_a_scratch_copy_runs_the_guarded_mutations_and_reports_them():
    graph = _Graph()
    db = _DB(graph, ["client_pokagon", "scratch_pokagon_provenance"])
    result = run_provenance_audit("pokagon", apply=True, db_factory=lambda **_k: db, batch_engagement_id=ENGAGEMENT, scratch_graph="scratch_pokagon_provenance")

    assert db.selected == "scratch_pokagon_provenance"
    assert result["mode"] == "apply"
    assert result["counts"]["apply_attempted"] == 2
    assert result["counts"]["apply_succeeded"] == 2
    assert result["counts"]["apply_conflicts"] == 0
    assert all(params["group_id"] == "scratch_pokagon_provenance" for _q, params in graph.reads + graph.writes)
    assert all("cardinality = 1" in query for query, _p in graph.writes)


def test_a_dry_run_on_a_scratch_copy_writes_nothing():
    graph = _Graph()
    db = _DB(graph, ["scratch_pokagon_provenance"])
    result = run_provenance_audit("pokagon", db_factory=lambda **_k: db, batch_engagement_id=ENGAGEMENT, scratch_graph="scratch_pokagon_provenance")
    assert result["mode"] == "audit"
    assert graph.writes == []


@pytest.mark.parametrize("bad", ["client_pokagon", "scratch-pokagon", "SCRATCH_x", "scratch_", "segment_tribal_gaming"])
def test_only_scratch_names_can_be_targeted(bad):
    graph = _Graph()
    db = _DB(graph, [bad])
    with pytest.raises(ProvenanceAuditReadError) as caught:
        run_provenance_audit("pokagon", apply=True, db_factory=lambda **_k: db, scratch_graph=bad)
    assert caught.value.code == AUDIT_GRAPH_NOT_FOUND_CODE
    assert graph.writes == []
