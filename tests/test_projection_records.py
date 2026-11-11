"""Retrieval over projected process records.

The records are the approved operating record written exactly; this checks the
two derived steps around them. Embedding gives a vector to a record that has a
statement and no vector, and never the vector of an earlier wording. Search
returns the records in force (those whose version their workflow points at)
for the engagements asked about, fused from a keyword leg and a vector leg,
and answers empty rather than failing.
"""

import re
import asyncio
from types import SimpleNamespace

import pytest

from app.config import settings
from app.models.projection import CURRENT_VERSION_PROPERTIES, POINTER_RECORD_KINDS, PROJECTION_NODE_LABEL
from app.services import graphiti_client, projection_records


class FakeGraph:
    """Scripted answers. `vector_rows` / `keyword_rows` are the records in force
    a leg would keep; `need_window` is the smallest candidate window at which
    the leg finds them (below it the window is full of history and nothing
    survives); `total` is how many candidates the index holds at all."""

    def __init__(self, pending=None, vector_rows=None, keyword_rows=None, fail=(), need_window=0, total=10_000, refuse_text=None, holds=True):
        self.pending = list(pending or [])
        # Whether the engagements asked about have a workflow that points at a version: the search asks this first.
        self.holds = holds
        self.vector_rows = vector_rows or []
        self.keyword_rows = keyword_rows or []
        self.fail = set(fail)
        self.need_window = need_window
        self.total = total
        # Rows a window narrower than need_window already keeps, and a window size from which reads fail.
        self.early_rows: list = []
        self.fail_from_window = 0
        self.queries: list[tuple[str, dict]] = []
        self.reads: list[tuple[str, dict]] = []
        self.read_timeouts: list[tuple[str, int | None]] = []
        self.written: list[dict] = []
        self.refused: list[dict] = []
        self.index_errors: dict[str, Exception] = {}

    def query(self, statement, params=None, **_):
        params = params or {}
        self.queries.append((statement, params))
        if statement.startswith("CREATE "):
            kind = "FULLTEXT" if "FULLTEXT" in statement else "VECTOR"
            if kind in self.index_errors:
                raise self.index_errors[kind]
            return SimpleNamespace(result_set=[])
        if "text_embedding IS NULL" in statement:
            # As the real query does: a record with a vector, or already refused by this model, is not pending.
            done = {w["uuid"] for w in self.written} | {m["uuid"] for m in self.refused}
            waiting = [r for r in self.pending if r[0] not in done]
            return SimpleNamespace(result_set=waiting[: params["limit"]])
        if "SET n.text_embedding" in statement:
            self.written.append(params)
            return SimpleNamespace(result_set=[[1]])
        if "SET n.text_emb_failed" in statement:
            self.refused.append(params)
            return SimpleNamespace(result_set=[[1]])
        raise AssertionError(f"a write-capable query was used for: {statement}")

    def ro_query(self, statement, params=None, timeout=None):
        params = params or {}
        self.reads.append((statement, params))
        self.read_timeouts.append((statement, timeout))
        if statement == projection_records._HOLDS_RECORDS:
            if "presence" in self.fail:
                raise RuntimeError("the presence read timed out")
            return SimpleNamespace(result_set=[["wf-1"]] if self.holds else [])
        if "db.idx.vector.queryNodes" in statement:
            leg, rows = "vector", self.vector_rows
            window = int(statement.split("'text_embedding', ")[1].split(",")[0])
        elif "db.idx.fulltext.queryNodes" in statement:
            leg, rows = "keyword", self.keyword_rows
            window = int(statement.split("ORDER BY score DESC LIMIT ")[1].split(" ")[0])
        else:
            raise AssertionError(f"unexpected read: {statement}")
        if leg in self.fail or (self.fail_from_window and window >= self.fail_from_window):
            raise RuntimeError(f"no {leg} index")
        seen = min(window, self.total)
        if statement.endswith("RETURN count(n) AS seen"):
            return SimpleNamespace(result_set=[[seen]])
        kept = rows if window >= self.need_window else self.early_rows
        pool = int(statement.rsplit("LIMIT ", 1)[1])
        return SimpleNamespace(result_set=[[*row[:8], row[8], seen] for row in kept[:pool]])


class FakeEmbedder:
    def __init__(self, dim=4, refuse=(), down=False):
        self.dim = dim
        self.refuse = set(refuse)
        self.down = down
        self.batches: list[list[str]] = []

    async def create_batch(self, texts):
        self.batches.append(list(texts))
        if self.down or any(text in self.refuse for text in texts):
            raise ValueError("the provider refused this request")
        return [[float(len(text))] * self.dim for text in texts]

    async def create(self, input_data):
        return [0.5] * self.dim


@pytest.fixture
def wired(monkeypatch):
    def wire(graph, embedder=None):
        monkeypatch.setattr(settings, "embedding_dim", 4, raising=False)
        monkeypatch.setattr(settings, "embedding_model", "unit-embedder", raising=False)
        monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: SimpleNamespace(select_graph=lambda name: graph))
        monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: embedder)
        projection_records._indexed_graphs.clear()
        return graph

    yield wire
    projection_records._indexed_graphs.clear()


def row(record_id, text="a statement", kind="PROCESS_STEP", engagement="eng-1", score=1.0):
    return [record_id, text, kind, f"src-{record_id}", "pv-3", engagement, "wf-1", "Receiving", score]


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


def test_only_a_record_retrieval_could_return_is_pending_and_only_while_its_vector_is_missing_or_from_another_model():
    query = projection_records.PENDING_QUERY
    assert f"(n:{PROJECTION_NODE_LABEL} " in query
    assert "n.text IS NOT NULL" in query
    # A Finding, a role or a workflow that happens to carry text is never embedded.
    assert "n.workflow_node_id IS NOT NULL" in query
    assert "n.text_embedding IS NULL OR n.text_emb_model IS NULL OR n.text_emb_model <> $marker" in query
    # A statement this model refused is set aside: not asked again until some hours have passed, it is
    # restated, or the model changes.
    assert "n.text_emb_failed IS NULL OR n.text_emb_failed <> $marker" in query
    assert "n.text_emb_failed_at < $retry_before" in query
    assert "n.text_emb_failed_at = $now" in projection_records.MARK_REFUSED_QUERY
    assert projection_records.REFUSED_RETRY_AFTER_SECONDS >= 3600
    assert "ORDER BY n.uuid" in query


def test_embedding_writes_one_vector_per_pending_record_and_reports_what_is_left(wired):
    graph = wired(
        FakeGraph(pending=[["r1", "eng-1", "first statement"], ["r2", "eng-1", "second"], ["r3", "eng-1", "third one"]]),
        FakeEmbedder(),
    )
    out = asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1", limit=2))
    assert out == {"graph_name": "client_unit_tenant", "pending": 2, "written": 2, "failed": 0, "remaining": True}
    assert [w["uuid"] for w in graph.written] == ["r1", "r2"]
    assert graph.written[0]["vector"] == [15.0, 15.0, 15.0, 15.0]
    assert graph.written[0]["marker"] == "unit-embedder:4"


def test_the_vector_is_written_only_while_the_statement_is_still_the_one_embedded():
    """A record restated between the read and the write keeps waiting, rather
    than being given the vector of its previous wording."""
    statement = projection_records.SET_VECTOR_QUERY
    assert "n.text = $text" in statement
    assert "engagement_id: $engagement_id" in statement


def test_one_statement_the_provider_refuses_does_not_hold_up_the_rest(wired):
    embedder = FakeEmbedder(refuse={"poison"})
    graph = wired(FakeGraph(pending=[["r1", "eng-1", "fine one"], ["r2", "eng-1", "poison"], ["r3", "eng-1", "fine two"]]), embedder)
    out = asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1"))
    assert (out["pending"], out["written"], out["failed"]) == (3, 2, 1)
    assert [w["uuid"] for w in graph.written] == ["r1", "r3"]
    assert [m["uuid"] for m in graph.refused] == ["r2"]


def test_a_provider_that_is_not_answering_stops_the_pass_at_once_and_blames_no_statement(wired):
    pending = [[f"r{i:03d}", "eng-1", f"statement {i}"] for i in range(150)]
    embedder = FakeEmbedder(down=True)
    graph = wired(FakeGraph(pending=pending), embedder)
    with pytest.raises(projection_records.EmbeddingUnavailable, match="not answering"):
        asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1"))
    # One batch, one statement on its own, one known-good statement: three calls, not one per record.
    assert len(embedder.batches) == 3
    assert embedder.batches[2] == [projection_records.PROVIDER_CHECK_STATEMENT]
    assert graph.written == [] and graph.refused == []


def test_statements_the_provider_refuses_one_after_another_are_marked_and_the_rest_are_embedded(wired):
    """Five refusals in a row with a healthy provider are five bad statements,
    not an outage: nothing behind them may be held up."""
    pending = [[f"r{i:03d}", "eng-1", f"statement {i}"] for i in range(20)]
    poison = {f"statement {i}" for i in range(7)}
    graph = wired(FakeGraph(pending=pending), FakeEmbedder(refuse=poison))
    out = asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1"))
    assert (out["pending"], out["written"], out["failed"], out["remaining"]) == (20, 13, 7, False)
    assert [m["uuid"] for m in graph.refused] == [f"r{i:03d}" for i in range(7)]
    assert graph.refused[0]["marker"] == "unit-embedder:4"


def test_a_refused_statement_at_the_front_does_not_hold_up_the_one_behind_it(wired):
    graph = wired(FakeGraph(pending=[["r1", "eng-1", "poison"], ["r2", "eng-1", "later"]]), FakeEmbedder(refuse={"poison"}))
    first = asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1", limit=1))
    assert (first["pending"], first["written"], first["failed"], first["remaining"]) == (1, 0, 1, True)
    second = asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1", limit=1))
    assert (second["pending"], second["written"], second["failed"], second["remaining"]) == (1, 1, 0, False)
    assert [w["uuid"] for w in graph.written] == ["r2"]


def test_the_provider_is_checked_before_every_statement_is_set_aside(wired):
    """An outage that begins mid-pass must not get healthy statements set aside
    on the strength of a check made earlier."""
    pending = [[f"r{i:03d}", "eng-1", f"statement {i}"] for i in range(12)]
    embedder = FakeEmbedder(refuse={row_[2] for row_ in pending})
    calls = {"check": 0}
    original = embedder.create_batch

    async def flaky(texts):
        if texts == [projection_records.PROVIDER_CHECK_STATEMENT]:
            calls["check"] += 1
            if calls["check"] >= 2:
                embedder.batches.append(list(texts))
                raise ValueError("down now")
        return await original(texts)

    embedder.create_batch = flaky
    graph = wired(FakeGraph(pending=pending), embedder)
    with pytest.raises(projection_records.EmbeddingUnavailable):
        asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1"))
    # Only the statement refused while the provider was still answering is set aside.
    assert [m["uuid"] for m in graph.refused] == ["r000"]
    assert calls["check"] == 2


def test_a_statement_set_aside_is_stamped_and_the_pass_asks_only_for_those_due_again(wired, monkeypatch):
    monkeypatch.setattr(projection_records.time, "time", lambda: 1_000_000)
    graph = wired(FakeGraph(pending=[["r1", "eng-1", "poison"]]), FakeEmbedder(refuse={"poison"}))
    asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1"))
    assert graph.refused[0]["now"] == 1_000_000
    (params,) = [p for q, p in graph.queries if "text_embedding IS NULL" in q]
    assert params["retry_before"] == 1_000_000 - projection_records.REFUSED_RETRY_AFTER_SECONDS


def test_a_batch_is_bounded_by_its_text_as_well_as_its_count(wired, monkeypatch):
    monkeypatch.setattr(projection_records, "EMBED_BATCH_CHARS", 25)
    embedder = FakeEmbedder()
    wired(FakeGraph(pending=[[f"r{i}", "eng-1", "x" * 10] for i in range(5)]), embedder)
    asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1"))
    assert [len(batch) for batch in embedder.batches] == [2, 2, 1]


def test_a_vector_of_the_wrong_size_stops_the_pass_rather_than_being_counted_as_one_bad_statement(wired):
    wired(FakeGraph(pending=[["r1", "eng-1", "statement"]]), FakeEmbedder(dim=3))
    with pytest.raises(projection_records.EmbeddingDimensionMismatch, match="3 dimensions"):
        asyncio.run(projection_records.embed_pending("unit-tenant", "eng-1"))


def test_embedding_with_nothing_pending_does_nothing(wired):
    embedder = FakeEmbedder()
    graph = wired(FakeGraph(pending=[]), embedder)
    out = asyncio.run(projection_records.embed_pending("unit-tenant"))
    assert out["pending"] == 0 and out["written"] == 0 and out["remaining"] is False
    assert embedder.batches == []
    assert graph.written == []


def test_a_graph_is_marked_indexed_only_when_its_indexes_exist(wired):
    graph = wired(FakeGraph(), FakeEmbedder())
    graph.index_errors["VECTOR"] = RuntimeError("connection reset")
    asyncio.run(projection_records.embed_pending("unit-tenant"))
    assert "client_unit_tenant" not in projection_records._indexed_graphs
    graph.index_errors["VECTOR"] = RuntimeError("Attribute 'text_embedding' is already indexed")
    asyncio.run(projection_records.embed_pending("unit-tenant"))
    assert "client_unit_tenant" in projection_records._indexed_graphs


def test_deleting_a_graph_forgets_its_record_indexes_too():
    projection_records._indexed_graphs.add("client_unit_tenant")
    graphiti_client.forget_graph_indexes("client_unit_tenant")
    assert "client_unit_tenant" not in projection_records._indexed_graphs


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_both_legs_return_only_records_whose_version_their_workflow_points_at():
    vector, keyword = projection_records.record_leg_queries(6)
    for leg in (vector, keyword):
        assert "n.engagement_id IN $engagement_ids" in leg
        assert "n.group_id = $group_id" in leg
        assert f"'{PROJECTION_NODE_LABEL}'" in leg
        # From the record to its workflow node, in the same tenant and engagement...
        assert f"MATCH (w:{PROJECTION_NODE_LABEL} {{uuid: n.workflow_node_id}})" in leg
        assert "w.group_id = $group_id AND w.engagement_id = n.engagement_id" in leg
        # ...which must itself be a workflow, not any row that carries a pointer-shaped property...
        assert "w.projection_source_kind = 'WORKFLOW'" in leg
        # ...and in force only when the pointer for the record's own kind names its version.
        assert "(n.projection_source_kind IN ['PROCESS_STEP', 'PROCESS_VERSION'] AND n.projection_source_version_id = w.current_map_version_id)" in leg
        assert "(n.projection_source_kind IN ['SOP', 'SOP_DEFINITION', 'SOP_EXCEPTION', 'SOP_SECTION', 'SOP_STEP'] AND n.projection_source_version_id = w.current_sop_version_id)" in leg
        assert "(n.projection_source_kind IN ['RACI', 'RACI_MATRIX'] AND n.projection_source_version_id = w.current_raci_version_id)" in leg
        # A version is compared with a pointer nowhere outside those three branches.
        assert leg.count("n.projection_source_version_id = w.") == 3
        # Every pointer the model lets a workflow carry is one the search reads, and no other.
        named = sorted(set(re.findall(r"w\.(current_[a-z]+_version_id)", leg)))
        assert named == sorted(CURRENT_VERSION_PROPERTIES)
    # The candidate window is taken before the join, and can be widened.
    assert keyword.index("LIMIT 30") < keyword.index("MATCH (w:")
    assert "'text_embedding', 30," in vector
    assert "'text_embedding', 480," in projection_records.record_leg_queries(6, 480)[0]
    # Equal scores are put in order by record id, so the cut is the same on every ask.
    assert vector.rstrip().endswith("ORDER BY score ASC, uuid ASC LIMIT 6")
    assert keyword.rstrip().endswith("ORDER BY score DESC, uuid ASC LIMIT 6")
    # A vector made by another embedding model is not compared with this one's.
    assert "n.text_emb_model = $marker" in vector


def test_a_record_both_legs_agree_on_ranks_first(wired):
    wired(
        FakeGraph(vector_rows=[row("a"), row("b"), row("c")], keyword_rows=[row("c"), row("d")], total=3),
        FakeEmbedder(),
    )
    out = asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "who checks the seal", 3))
    # "c" is in both legs; "b" and "d" tie on rank, and the vector leg's comes first.
    assert [record["record_id"] for record in out] == ["c", "a", "b"]
    assert out[0] == {
        "record_id": "c",
        "text": "a statement",
        "kind": "PROCESS_STEP",
        "source_id": "src-c",
        "source_version_id": "pv-3",
        "engagement_id": "eng-1",
        "workflow_id": "wf-1",
        "workflow_name": "Receiving",
    }


def test_a_window_full_of_history_is_widened_until_the_records_in_force_are_found(wired):
    """Old versions stay in the index. A first window can be all history; the
    search must widen, not answer short."""
    graph = wired(FakeGraph(vector_rows=[row(f"v{i}") for i in range(6)], keyword_rows=[], need_window=400), FakeEmbedder())
    out = asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal", 3))
    assert [record["record_id"] for record in out] == ["v0", "v1", "v2"]
    windows = [int(q.split("'text_embedding', ")[1].split(",")[0]) for q, _ in graph.reads if "vector.queryNodes" in q and "count(n)" not in q]
    assert windows == [30, 120, 480]


def test_widening_stops_when_the_index_has_nothing_more_or_at_the_cap(wired):
    # The index holds only 4 candidates: one read, and the window is known to be exhausted.
    small = wired(FakeGraph(vector_rows=[row("a")], keyword_rows=[], total=4), FakeEmbedder())
    asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal", 3))
    assert len([q for q, _ in small.reads if "vector.queryNodes" in q]) == 1
    # Nothing in force and nothing more in the index: one read and one count, no walk to the cap.
    empty = wired(FakeGraph(vector_rows=[], keyword_rows=[], total=4), FakeEmbedder())
    assert asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal", 3)) == []
    assert len([q for q, _ in empty.reads if "vector.queryNodes" in q]) == 2
    # Nothing in force anywhere: widen to the cap, then answer what there is.
    crowded = wired(FakeGraph(vector_rows=[], keyword_rows=[], need_window=10**9), FakeEmbedder())
    assert asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal", 3)) == []
    windows = [int(q.split("'text_embedding', ")[1].split(",")[0]) for q, _ in crowded.reads if "vector.queryNodes" in q and "count(n)" not in q]
    assert windows == [30, 120, 480, 1920, projection_records.MAX_CANDIDATES]


def test_engagements_that_hold_no_record_in_force_are_answered_at_once(wired, caplog):
    """A later engagement asking for its own records, or the other engagements
    of the one that holds them all. Nothing to rank: no leg is read, the
    question is not embedded, and nothing warns that the index is crowded."""
    embedder = FakeEmbedder()
    calls: list = []
    embedder.create = lambda input_data: calls.append(input_data)  # must never be reached
    graph = wired(FakeGraph(vector_rows=[], keyword_rows=[], need_window=10**9, holds=False), embedder)
    with caplog.at_level("WARNING", logger="graphiti_service"):
        assert asyncio.run(projection_records.search_records("unit-tenant", ["eng-later", "eng-other"], "seal number", 5)) == []
    assert [(q, p) for q, p in graph.reads] == [
        (projection_records._HOLDS_RECORDS, {"group_id": graphiti_client._graph_name_for_client("unit-tenant"), "engagement_ids": ["eng-later", "eng-other"]})
    ]
    assert calls == []
    assert graph.queries == []
    assert [r.getMessage() for r in caplog.records] == []


def test_the_presence_check_asks_for_a_workflow_that_points_at_a_version_in_this_tenant_and_these_engagements():
    q = projection_records._HOLDS_RECORDS
    # Word for word: every condition is joined by AND, the pointers by OR among
    # themselves. A condition loosened to OR would still say "yes" for every
    # tenant with any workflow, and the check would quietly stop doing its job.
    assert q == (
        "MATCH (w:ProjectionNode) "
        "WHERE w.engagement_id IN $engagement_ids AND w.group_id = $group_id "
        "AND w.projection_source_kind = 'WORKFLOW' "
        "AND (w.current_map_version_id IS NOT NULL OR w.current_raci_version_id IS NOT NULL OR w.current_sop_version_id IS NOT NULL) "
        "RETURN w.uuid LIMIT 1"
    )
    # Every pointer the model knows, from both of its tables, so a new kind of record cannot be left out of the check.
    assert set(POINTER_RECORD_KINDS) == set(CURRENT_VERSION_PROPERTIES)
    for pointer in CURRENT_VERSION_PROPERTIES:
        assert f"w.{pointer} IS NOT NULL" in q
    # Nothing from a caller is written into the statement.
    assert "$" in q and "'" not in q.replace("'WORKFLOW'", "")


def test_the_presence_check_has_a_short_time_limit_of_its_own_and_the_legs_keep_theirs(wired):
    graph = wired(FakeGraph(vector_rows=[row("a")], keyword_rows=[row("a")], total=1), FakeEmbedder())
    asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal number", 3))
    assert projection_records.PRESENCE_TIMEOUT_MS == 400
    assert [timeout for statement, timeout in graph.read_timeouts if statement == projection_records._HOLDS_RECORDS] == [400]
    assert sorted(timeout for statement, timeout in graph.read_timeouts if statement != projection_records._HOLDS_RECORDS) == sorted(
        [settings.search_vector_timeout_ms, settings.search_bm25_timeout_ms]
    )


def test_a_presence_check_that_fails_says_nothing_and_the_search_goes_on(wired, caplog):
    wired(FakeGraph(vector_rows=[row("v1")], keyword_rows=[row("k1")], fail={"presence"}, total=1), FakeEmbedder())
    with caplog.at_level("WARNING", logger="graphiti_service"):
        out = asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal number", 3))
    assert [record["record_id"] for record in out] == ["v1", "k1"]
    assert [r.getMessage() for r in caplog.records] == [
        f"[graphiti] record presence check failed on {graphiti_client._graph_name_for_client('unit-tenant')} error_type=RuntimeError; searching anyway"
    ]


def test_records_found_in_a_narrower_window_are_kept_when_a_wider_one_fails(wired):
    graph = FakeGraph(vector_rows=[row(f"v{i}") for i in range(6)], keyword_rows=[], need_window=400)
    graph.early_rows = [row("v0")]
    graph.fail_from_window = 120
    wired(graph, FakeEmbedder())
    out = asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal", 3))
    assert [record["record_id"] for record in out] == ["v0"]


def test_the_window_is_counted_before_any_filter_so_a_partly_embedded_index_is_not_read_as_exhausted():
    vector, _ = projection_records.record_leg_queries(6)
    assert vector.index("size(candidates) AS seen") < vector.index("n.text_emb_model = $marker")


def test_a_search_reads_and_never_writes_or_creates(wired):
    graph = wired(FakeGraph(vector_rows=[row("a")], keyword_rows=[row("a")], total=1), FakeEmbedder())
    asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal number", 3))
    assert graph.queries == []
    # The presence check, then one read per leg.
    assert [q == projection_records._HOLDS_RECORDS for q, _ in graph.reads] == [True, False, False]


def test_a_property_written_as_a_number_comes_back_as_text_instead_of_failing_the_answer(wired):
    odd = row("a")
    odd[7] = 12345
    wired(FakeGraph(vector_rows=[odd], keyword_rows=[], total=1), FakeEmbedder())
    out = asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal", 3))
    assert out[0]["workflow_name"] == "12345"


def test_the_keyword_leg_sends_the_question_as_an_or_of_its_terms(wired):
    graph = wired(FakeGraph(keyword_rows=[row("a")], total=1), FakeEmbedder())
    asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "Who checks the seal number?", 3))
    (params,) = [p for q, p in graph.reads if "fulltext.queryNodes" in q]
    assert params["query"].startswith("(") and params["query"].endswith("checks | seal | number)")
    assert "@group_id" not in params["query"]
    assert params["engagement_ids"] == ["eng-1"]


def test_a_question_with_no_searchable_terms_skips_the_keyword_leg(wired):
    graph = wired(FakeGraph(vector_rows=[row("v1")], keyword_rows=[row("k1")], total=1), FakeEmbedder())
    out = asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "?? !!", 3))
    assert [record["record_id"] for record in out] == ["v1"]
    assert not [q for q, _ in graph.reads if "fulltext.queryNodes" in q]


@pytest.mark.parametrize("failing, expected", [("vector", ["k1"]), ("keyword", ["v1"])])
def test_one_leg_failing_leaves_the_other_legs_answer(wired, failing, expected):
    wired(FakeGraph(vector_rows=[row("v1")], keyword_rows=[row("k1")], fail={failing}, total=1), FakeEmbedder())
    out = asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal number", 3))
    assert [record["record_id"] for record in out] == expected


def test_search_answers_empty_rather_than_failing(wired):
    wired(FakeGraph(fail={"vector", "keyword"}), FakeEmbedder())
    assert asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal number", 3)) == []
    assert asyncio.run(projection_records.search_records("unit-tenant", [], "seal number", 3)) == []
    assert asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "   ", 3)) == []


def test_without_an_embedder_the_keyword_leg_still_answers(wired):
    wired(FakeGraph(vector_rows=[row("v1")], keyword_rows=[row("k1")], total=1), None)
    out = asyncio.run(projection_records.search_records("unit-tenant", ["eng-1"], "seal number", 3))
    assert [record["record_id"] for record in out] == ["k1"]


# --------------------------------------------------------------------------
# The two routes
# --------------------------------------------------------------------------

import json  # noqa: E402
import time  # noqa: E402
import uuid as uuidlib  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app import auth  # noqa: E402
from app.auth import build_signature  # noqa: E402
from app.main import app  # noqa: E402

INGEST_SECRET = "ingest-secret-that-is-at-least-32-characters"
SEARCH_SECRET = "search-secret-that-is-at-least-32-characters"
ADMIN_SECRET = "admin-secret-that-is-at-least-32-charactersx"


@pytest.fixture
def perimeter(monkeypatch):
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required", raising=False)
    monkeypatch.setattr(settings, "graphiti_ingest_secret", INGEST_SECRET, raising=False)
    monkeypatch.setattr(settings, "graphiti_search_secret", SEARCH_SECRET, raising=False)
    monkeypatch.setattr(settings, "graphiti_admin_secret", ADMIN_SECRET, raising=False)
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", False, raising=False)

    async def consume(scope, nonce):
        return True

    monkeypatch.setattr(auth, "_consume_nonce", consume)


def signed_post(path, body, scope, secret):
    raw = json.dumps(body).encode("utf-8")
    timestamp, nonce = str(int(time.time())), uuidlib.uuid4().hex
    signature = build_signature(
        secret=secret, timestamp=timestamp, nonce=nonce, method="POST", path=path,
        scope=scope, client_slug=body["client_slug"], body=raw,
    )
    with TestClient(app) as client:
        return client.post(
            path,
            content=raw,
            headers={
                "content-type": "application/json",
                auth.AUTH_TIMESTAMP_HEADER: timestamp,
                auth.AUTH_SCOPE_HEADER: scope,
                auth.AUTH_CLIENT_HEADER: body["client_slug"],
                auth.AUTH_NONCE_HEADER: nonce,
                auth.AUTH_SIGNATURE_HEADER: signature,
            },
        )


RECORD_QUESTION = {"client_slug": "unit-tenant", "engagement_id": "eng-1", "query": "who checks the seal"}
RECORD_KEYS = ["record_id", "text", "kind", "source_id", "source_version_id", "engagement_id", "workflow_id", "workflow_name"]


def test_record_search_answers_an_empty_list_and_reads_nothing_while_the_lane_is_off(perimeter, monkeypatch):
    asked = []

    async def spy(**kwargs):
        asked.append(kwargs)
        return []

    monkeypatch.setattr(projection_records, "search_records", spy)
    response = signed_post("/search/records", RECORD_QUESTION, "search", SEARCH_SECRET)
    assert response.status_code == 200, response.text
    assert response.json()["records"] == []
    assert response.json()["contract_version"] == "graphiti_record_search_v1"
    assert asked == []


@pytest.mark.parametrize("mode", ["optional", "off"])
def test_record_search_reads_nothing_unless_the_signed_perimeter_is_closed(perimeter, monkeypatch, mode):
    """In any other auth posture the tenant in the body is not bound to a caller."""
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", True, raising=False)
    monkeypatch.setattr(settings, "graphiti_auth_mode", mode, raising=False)
    asked = []

    async def spy(**kwargs):
        asked.append(kwargs)
        return [dict(zip(RECORD_KEYS, row("a")))]

    monkeypatch.setattr(projection_records, "search_records", spy)
    with TestClient(app) as client:
        response = client.post("/search/records", json=RECORD_QUESTION)
    assert response.status_code == 200
    assert response.json()["records"] == []
    assert asked == []


@pytest.mark.parametrize("mode", ["optional", "off"])
def test_embedding_refuses_to_run_unless_the_signed_perimeter_is_closed(perimeter, monkeypatch, mode):
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", True, raising=False)
    monkeypatch.setattr(settings, "graphiti_auth_mode", mode, raising=False)
    with TestClient(app) as client:
        response = client.post("/ingest/projection/v2/embed", json={"client_slug": "unit-tenant"})
    assert response.status_code == 503


def test_record_search_refuses_a_tenant_name_that_is_not_one(perimeter):
    response = signed_post("/search/records", {**RECORD_QUESTION, "client_slug": "Not A Slug"}, "search", SEARCH_SECRET)
    assert response.status_code == 422


def test_record_search_reads_the_asking_engagement_and_the_others_named(perimeter, monkeypatch):
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", True, raising=False)
    seen = {}

    async def fake(**kwargs):
        seen.update(kwargs)
        return [dict(zip(
            ["record_id", "text", "kind", "source_id", "source_version_id", "engagement_id", "workflow_id", "workflow_name"],
            row("a"),
        ))]

    monkeypatch.setattr(projection_records, "search_records", fake)
    body = {**RECORD_QUESTION, "readable_engagement_ids": ["eng-2", "eng-1"], "max_results": 4}
    response = signed_post("/search/records", body, "search", SEARCH_SECRET)
    assert response.status_code == 200, response.text
    assert seen["engagement_ids"] == ["eng-1", "eng-2"]
    assert seen["max_results"] == 4
    assert response.json()["records"][0]["record_id"] == "a"


def test_a_failing_record_search_answers_empty_not_an_error(perimeter, monkeypatch):
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", True, raising=False)

    async def boom(**_):
        raise RuntimeError("index missing")

    monkeypatch.setattr(projection_records, "search_records", boom)
    response = signed_post("/search/records", RECORD_QUESTION, "search", SEARCH_SECRET)
    assert response.status_code == 200
    assert response.json()["records"] == []


def test_record_search_is_not_open_to_the_ingest_credential(perimeter):
    response = signed_post("/search/records", RECORD_QUESTION, "ingest", INGEST_SECRET)
    assert response.status_code in (401, 403)


def test_embedding_is_refused_while_the_lane_is_off(perimeter, monkeypatch):
    async def must_not_run(**_):
        raise AssertionError("refused before anything is read")

    monkeypatch.setattr(projection_records, "embed_pending", must_not_run)
    response = signed_post("/ingest/projection/v2/embed", {"client_slug": "unit-tenant"}, "ingest", INGEST_SECRET)
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_PROCESS_STRUCTURE" in json.dumps(response.json())


def test_a_provider_outage_is_reported_as_one_not_as_a_success_to_retry(perimeter, monkeypatch):
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", True, raising=False)

    async def down(**_):
        raise projection_records.EmbeddingUnavailable("the embedding provider is not answering")

    monkeypatch.setattr(projection_records, "embed_pending", down)
    response = signed_post("/ingest/projection/v2/embed", {"client_slug": "unit-tenant"}, "ingest", INGEST_SECRET)
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "projection_embed_provider_unavailable"


def test_embedding_runs_for_the_authenticated_tenant_only(perimeter, monkeypatch):
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", True, raising=False)
    seen = {}

    async def fake(**kwargs):
        seen.update(kwargs)
        return {"graph_name": kwargs["graph_name"], "pending": 3, "written": 3, "failed": 0, "remaining": False}

    monkeypatch.setattr(projection_records, "embed_pending", fake)
    response = signed_post(
        "/ingest/projection/v2/embed", {"client_slug": "unit-tenant", "engagement_id": "eng-1", "limit": 50}, "ingest", INGEST_SECRET
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"graph_name": "client_unit_tenant", "pending": 3, "written": 3, "failed": 0, "remaining": False}
    assert seen["graph_name"] == "client_unit_tenant" and seen["engagement_id"] == "eng-1" and seen["limit"] == 50

    refused = signed_post("/ingest/projection/v2/embed", {"client_slug": "unit-tenant"}, "search", SEARCH_SECRET)
    assert refused.status_code in (401, 403)
