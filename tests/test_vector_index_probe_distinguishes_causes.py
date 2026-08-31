"""The vector-index probe must tell the two causes apart.

Production shows Entity.name_embedding being CREATED over and over for the same
graph and never once reporting "already exists", while a query against it fails
with "Invalid arguments" in the same second. Two very different things produce
that, and they need opposite fixes:

  the CREATE is not producing a usable index  -> fix the creation
  the QUERY is wrong for this FalkorDB build  -> fix the query

A probe that only reported "vector search does not work" would leave that
unresolved, and the obvious guess -- rewrite the CREATE -- is wrong half the
time. So the classification is the deliverable, and it is what is tested here.

The tests deliberately do NOT touch FalkorDB. The existing dedup tests already
assert on emitted Cypher against a fake executor, which is exactly why this
defect survived them: code and test shared the assumption that the procedure
signature was valid. These tests pin the reasoning about a real server's reply,
not a substitute for asking it.
"""

from app.diagnostics.vector_index import classify_probe, summarize_indexes


class TestClassifyProbe:
    def test_no_error_means_the_index_exists_and_answers(self):
        assert classify_probe(None).startswith("OK")

    def test_invalid_arguments_points_at_the_query_not_the_index(self):
        # The exact production string, verbatim from the 2026-08-30 logs.
        verdict = classify_probe(
            "Invalid arguments for procedure 'db.idx.vector.queryNodes'"
        )
        assert verdict.startswith("SIGNATURE")
        assert "QUERY is wrong" in verdict

    def test_it_is_case_insensitive_because_driver_casing_varies(self):
        assert classify_probe("INVALID ARGUMENTS for procedure").startswith("SIGNATURE")

    def test_unknown_procedure_means_the_build_has_no_vector_support(self):
        # A different fix again: neither query nor create, but the FalkorDB
        # image itself. Worth its own verdict rather than being lumped in.
        assert classify_probe("Unknown procedure 'db.idx.vector.queryNodes'").startswith(
            "UNSUPPORTED"
        )

    def test_a_missing_index_points_at_the_create(self):
        assert classify_probe("No such index on :Entity(name_embedding)").startswith("ABSENT")

    def test_an_unrecognised_error_is_passed_through_rather_than_guessed_at(self):
        # Silently bucketing an unknown error into one of the known causes
        # would send someone to the wrong file with false confidence.
        verdict = classify_probe("connection reset by peer")
        assert verdict.startswith("OTHER")
        assert "connection reset by peer" in verdict


class TestSummarizeIndexes:
    def test_it_renders_dict_rows(self):
        lines = summarize_indexes([
            {"label": "Entity", "properties": ["name_embedding"], "types": {"name_embedding": ["VECTOR"]}},
        ])
        assert len(lines) == 1
        assert "label=Entity" in lines[0]
        assert "name_embedding" in lines[0]

    def test_it_renders_positional_rows(self):
        # The driver returns positional rows whose arity has changed across
        # versions — which is itself a candidate cause of the failure under
        # investigation, so the reader must see them rather than a parse error.
        lines = summarize_indexes([["Entity", ["name_embedding"], "vector"]])
        assert lines == ["Entity | ['name_embedding'] | vector"]

    def test_it_does_not_crash_on_an_empty_or_missing_result(self):
        assert summarize_indexes([]) == []
        assert summarize_indexes(None) == []  # type: ignore[arg-type]

    def test_an_unexpected_row_shape_is_shown_not_dropped(self):
        # A dropped row would read as "no such index", which is one of the two
        # verdicts this whole script exists to distinguish.
        assert summarize_indexes(["raw-string-row"]) == ["raw-string-row"]
