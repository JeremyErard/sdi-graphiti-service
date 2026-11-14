"""Retrieval over projected process records.

/ingest/projection/v2 writes the approved operating record exactly: a process
version and its steps, an SOP and its steps, exceptions and definitions. Each of
those rows may carry one `text` property, a plain statement of what the record
says, written by the caller from the source record itself. This module makes
those statements findable without changing them:

* embed_pending() gives every record with a statement a vector. The apply
  restates a row's whole property map (that is what makes reprojection
  converge), which drops the vector with it, so "has a statement, has no
  vector" is the complete definition of pending and the pass is idempotent.
* search_records() answers a question with the records in force that match
  it, by keyword and by meaning, fused the same way the fact search fuses its
  legs.

Which records are in force is a pointer, not a flag on every row: a record
names its workflow node, and the workflow node names the version of its map
and of its SOP that is in force. A record is returned only when its own
version is one its workflow points at. Replacing a version is one write to
the workflow node, and the old version's records stop being returned at that
instant.

Records of versions no longer in force stay in the graph as history, and so
in both indexes. A search therefore reads a window of candidates, keeps the
ones in force, and widens the window when too few survive, up to a cap; it
never answers short because the first window happened to be full of history.

Nothing here runs a language model, and nothing here writes anything except
the derived vector and its marker. Searching creates nothing: a tenant with
no projected records has no graph made for it by being asked.

The projection label space is separate from graphiti's: these queries read
ProjectionNode rows, and a section of a law under its own base label
(LawRecordNode), only, and cannot return, or be confused with, an extracted
fact. A section of a law carries no ProjectionNode label, so it is not even
counted in the records' keyword index (see LAW_SECTION_BASE_LABEL).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config import settings
from app.models.projection import (
    POINTER_ANCHOR_KIND,
    POINTER_RECORD_KINDS,
    RECORD_SET_COUNTED_AT,
    LAW_SECTION_BASE_LABEL,
    PROJECTION_NODE_LABEL,
    LAW_TEXT_PROPERTY,
    LAW_VECTOR_PROPERTY,
    RECORD_TEXT_PROPERTY,
    RECORD_WORKFLOW_PROPERTY,
    SOURCE_KIND_NODE_LABEL,
    ProjectionSourceKind,
)
from app.services import graphiti_client, question_vectors, vector_turns

logger = logging.getLogger("graphiti_service")

# Rank constant for reciprocal-rank fusion, as in the fact search.
_RRF_K = 60

# Each leg reads a bounded window of candidates before the join that decides
# which are in force, starting at pool * _OVERFETCH. When fewer than `pool`
# survive and the window was full, the window is widened by _WIDEN_FACTOR and
# the leg asked again, up to MAX_CANDIDATES.
_OVERFETCH = 5
_WIDEN_FACTOR = 4
MAX_CANDIDATES = 2000

EMBED_BATCH = 64
# A batch is also bounded by its text, so no request to the embedding provider
# grows with the length of the statements in it.
EMBED_BATCH_CHARS = 100_000
MAX_EMBED_PER_CALL = 2000

_indexed_graphs: set[str] = set()


class EmbeddingDimensionMismatch(RuntimeError):
    """The embedder and this graph's record index disagree on vector size: a
    configuration fault for the whole pass, not one statement to skip."""


class EmbeddingUnavailable(RuntimeError):
    """The provider is not answering even a known-good statement. The pass
    stops instead of asking it once per pending record."""


# When a single statement is refused, a short known-good statement decides
# whose fault it is, every time, before anything is marked. If that embeds,
# the provider is answering and the refused statement is set aside: marked,
# and the pass goes on. If it does not, the provider is down and the pass
# stops with nothing marked. One extra call per refused statement, on the
# refusal path only.
PROVIDER_CHECK_STATEMENT = "This sentence checks that the embedding provider is answering."

# A refusal may still have been the provider's passing trouble (a rate limit
# the short check statement slipped under). A set-aside statement is therefore
# tried again after this long, not left without a vector until it is restated.
REFUSED_RETRY_AFTER_SECONDS = 6 * 60 * 60


def forget_graph(graph_name: str) -> None:
    """Drop the ensured-index mark for a graph that was deleted and may return."""
    _indexed_graphs.discard(graph_name)


def _ensure_indices(graph: Any, graph_name: str) -> None:
    """Keyword and vector indexes over the record statement. Idempotent.

    Called from the embedding pass only (the ingest side): a search never
    creates anything. The graph is marked only when each index exists, that is
    when its CREATE succeeded or said the index is already there; a failure for
    any other reason leaves the graph unmarked so the next pass tries again.
    """
    if graph_name in _indexed_graphs:
        return
    dim = int(settings.embedding_dim)
    statements = (
        f"CREATE FULLTEXT INDEX FOR (n:{PROJECTION_NODE_LABEL}) ON (n.{RECORD_TEXT_PROPERTY})",
        f"CREATE VECTOR INDEX FOR (n:{PROJECTION_NODE_LABEL}) ON (n.text_embedding) "
        f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}",
        # The law has no index: it is searched by meaning only, every section compared one by one (see _LAW_READ).
    )
    ready = True
    for statement in statements:
        try:
            graph.query(statement)
        except Exception as error:
            if "already indexed" not in str(error).lower():
                ready = False
                logger.warning(
                    "[graphiti] record index could not be ensured on %s error_type=%s",
                    graph_name,
                    type(error).__name__,
                )
    if ready:
        _indexed_graphs.add(graph_name)


def _marker() -> str:
    return f"{settings.embedding_model}:{int(settings.embedding_dim)}"


def _law_form(statement: str) -> str:
    """The same statement for a section of a law. Everything the embedding pass keeps on a record under a name
    that starts with its words' name (`text`, `text_embedding`, `text_emb_model`, `text_emb_failed`,
    `text_emb_failed_at`) a section keeps under the same name with the law's prefix: `law_text`,
    `law_text_embedding`, `law_text_emb_model`, and so on. One rule, so a read and a write cannot disagree.
    A section is found under its own base label, never as a ProjectionNode (LAW_SECTION_BASE_LABEL)."""
    record_node = f"(n:{PROJECTION_NODE_LABEL} "
    if statement.count(record_node) != 1:  # pragma: no cover - a guard on the statements below
        raise RuntimeError("a law form is made from a statement that starts from one record node")
    return statement.replace(record_node, f"(n:{LAW_SECTION_BASE_LABEL} ").replace(
        f"n.{RECORD_TEXT_PROPERTY}", f"n.{LAW_TEXT_PROPERTY}"
    )


# The name a section's vector carries the model it was made by under (see _law_form).
LAW_MODEL_PROPERTY = f"{LAW_TEXT_PROPERTY}_emb_model"
if f"{LAW_TEXT_PROPERTY}_embedding" != LAW_VECTOR_PROPERTY:  # pragma: no cover - a guard on the model's names
    raise RuntimeError("the law's vector name must be the law's text name followed by _embedding")


# Only a record that retrieval could ever return: it has a statement and names
# its workflow. Pending means it has no vector, or its vector was made by a
# different model or size than the one in use now. A statement this model has
# refused (with the provider answering) is set aside for a while: not asked
# again until REFUSED_RETRY_AFTER_SECONDS have passed, it is restated (which
# clears the mark; a kind restated in place every run keeps it while it says
# the same thing), or the model changes. Otherwise the same refusals would
# sit at the front of every pass and nothing behind them would be reached.
# Ordered, so passes walk rows in the same order.
PENDING_QUERY = (
    f"MATCH (n:{PROJECTION_NODE_LABEL} {{group_id: $group_id}}) "
    f"WHERE n.{RECORD_TEXT_PROPERTY} IS NOT NULL AND n.{RECORD_WORKFLOW_PROPERTY} IS NOT NULL "
    "AND (n.text_embedding IS NULL OR n.text_emb_model IS NULL OR n.text_emb_model <> $marker) "
    "AND (n.text_emb_failed IS NULL OR n.text_emb_failed <> $marker "
    "OR n.text_emb_failed_at IS NULL OR n.text_emb_failed_at < $retry_before) "
    "AND ($engagement_id IS NULL OR n.engagement_id = $engagement_id) "
    f"RETURN n.uuid AS uuid, n.engagement_id AS engagement_id, n.{RECORD_TEXT_PROPERTY} AS text "
    "ORDER BY n.uuid LIMIT $limit"
)

SET_VECTOR_QUERY = (
    f"MATCH (n:{PROJECTION_NODE_LABEL} {{uuid: $uuid, group_id: $group_id, "
    "engagement_id: $engagement_id}) "
    f"WHERE n.{RECORD_TEXT_PROPERTY} = $text "
    "SET n.text_embedding = vecf32($vector), n.text_emb_model = $marker "
    "RETURN count(n) AS written"
)


# The same three for the sections of a law (LAW_TEXT_PROPERTY). A row of the law's pending read carries a fourth
# value, so the pass knows which write to use.
LAW_PENDING_QUERY = _law_form(PENDING_QUERY).replace(
    f"n.{LAW_TEXT_PROPERTY} AS text ", f"n.{LAW_TEXT_PROPERTY} AS text, 'law' AS lane "
)
LAW_SET_VECTOR_QUERY = _law_form(SET_VECTOR_QUERY)


MARK_REFUSED_QUERY = (
    f"MATCH (n:{PROJECTION_NODE_LABEL} {{uuid: $uuid, group_id: $group_id, "
    "engagement_id: $engagement_id}) "
    f"WHERE n.{RECORD_TEXT_PROPERTY} = $text "
    "SET n.text_emb_failed = $marker, n.text_emb_failed_at = $now "
    "RETURN count(n) AS marked"
)
LAW_MARK_REFUSED_QUERY = _law_form(MARK_REFUSED_QUERY)


def _of_a_law(row: Any) -> bool:
    return len(row) > 3 and row[3] == "law"


def _batches(rows: list[Any]) -> list[list[Any]]:
    """Rows in batches bounded by count and by total statement length."""
    out: list[list[Any]] = []
    current: list[Any] = []
    size = 0
    for row in rows:
        length = len(str(row[2]))
        if current and (len(current) >= EMBED_BATCH or size + length > EMBED_BATCH_CHARS):
            out.append(current)
            current, size = [], 0
        current.append(row)
        size += length
    if current:
        out.append(current)
    return out


async def embed_pending(
    client_slug: str,
    engagement_id: str | None = None,
    limit: int = MAX_EMBED_PER_CALL,
    graph_name: str | None = None,
) -> dict[str, Any]:
    """Give every record that has a statement and no current vector its vector.

    Returns how many were pending, how many were written, how many could not
    be embedded, and whether more remain, so a caller loops while `remaining`.
    Every pass moves forward: a record it reads is either given its vector or
    marked as refused, so the next pass starts behind it.
    The write is guarded on the statement still being the one that was
    embedded: a record restated between the read and the write is left pending
    rather than given the vector of its previous wording.

    One statement the provider refuses does not hold up the rest: a batch that
    fails is retried one statement at a time, and a statement that still fails
    while the provider answers a known-good one is counted in `failed`, logged
    by id and set aside for a few hours, so later passes go past it. If the provider does not
    answer the known-good statement either, the pass stops with
    EmbeddingUnavailable and nothing is marked: an outage is not a verdict on
    any statement.
    """
    name = graph_name or graphiti_client._graph_name_for_client(client_slug)
    graph = graphiti_client.get_falkor_db().select_graph(name)
    await asyncio.to_thread(_ensure_indices, graph, name)

    embedder = graphiti_client._create_embedder()
    if embedder is None:
        raise RuntimeError("record embedding requires an explicit embedder")

    bounded = max(1, min(int(limit), MAX_EMBED_PER_CALL))
    marker = _marker()
    rows = (
        await asyncio.to_thread(
            graph.query,
            PENDING_QUERY,
            {
                "group_id": name,
                "engagement_id": engagement_id,
                "limit": bounded + 1,
                "marker": marker,
                "retry_before": int(time.time()) - REFUSED_RETRY_AFTER_SECONDS,
            },
        )
    ).result_set
    if len(rows) <= bounded:
        # Room left in this pass: the sections of a law that wait for a vector, after the records.
        rows = list(rows) + list(
            (
                await asyncio.to_thread(
                    graph.query,
                    LAW_PENDING_QUERY,
                    {
                        "group_id": name,
                        "engagement_id": engagement_id,
                        "limit": bounded + 1 - len(rows),
                        "marker": marker,
                        "retry_before": int(time.time()) - REFUSED_RETRY_AFTER_SECONDS,
                    },
                )
            ).result_set
            or []
        )
    remaining = len(rows) > bounded
    rows = rows[:bounded]

    dim = int(settings.embedding_dim)
    written = 0
    failed = 0
    refused_in_a_row = 0  # for the log only: how many refusals the pass met back to back

    async def provider_answers() -> bool:
        try:
            vectors = await embedder.create_batch([PROVIDER_CHECK_STATEMENT])
            return len(vectors) == 1 and len(vectors[0]) == dim
        except Exception:
            return False

    async def embed(batch: list[Any]) -> list[list[float]]:
        vectors = await embedder.create_batch([str(row[2]).replace("\n", " ") for row in batch])
        for vector in vectors:
            if len(vector) != dim:
                raise EmbeddingDimensionMismatch(
                    f"the embedder returned {len(vector)} dimensions; this graph's "
                    f"record index is built for {dim}"
                )
        return vectors

    async def store(row: Any, vector: list[float]) -> int:
        result = await asyncio.to_thread(
            graph.query,
            LAW_SET_VECTOR_QUERY if _of_a_law(row) else SET_VECTOR_QUERY,
            {
                "uuid": row[0],
                "group_id": name,
                "engagement_id": row[1],
                "text": row[2],
                "vector": [float(x) for x in vector],
                "marker": marker,
            },
        )
        return int(result.result_set[0][0]) if result.result_set else 0

    for batch in _batches(rows):
        try:
            vectors = await embed(batch)
        except EmbeddingDimensionMismatch:
            raise
        except Exception as error:
            # The whole batch was refused: find out which statement, one at a time.
            logger.warning(
                "[graphiti] record embedding batch of %d failed on %s error_type=%s; retrying singly",
                len(batch),
                name,
                type(error).__name__,
            )
            for row in batch:
                try:
                    (vector,) = await embed([row])
                    written += await store(row, vector)
                    refused_in_a_row = 0
                except EmbeddingDimensionMismatch:
                    raise
                except Exception as single:
                    refused_in_a_row += 1
                    if not await provider_answers():
                        raise EmbeddingUnavailable(
                            f"the embedding provider is not answering ({written} written, "
                            f"{failed} refused before that)"
                        ) from single
                    failed += 1
                    await asyncio.to_thread(
                        graph.query,
                        LAW_MARK_REFUSED_QUERY if _of_a_law(row) else MARK_REFUSED_QUERY,
                        {
                            "uuid": row[0],
                            "group_id": name,
                            "engagement_id": row[1],
                            "text": row[2],
                            "marker": marker,
                            "now": int(time.time()),
                        },
                    )
                    logger.warning(
                        "[graphiti] record %s on %s could not be embedded error_type=%s",
                        row[0],
                        name,
                        type(single).__name__,
                    )
            continue
        refused_in_a_row = 0
        for row, vector in zip(batch, vectors):
            written += await store(row, vector)

    logger.info(
        "[graphiti] record embedding on %s: %d pending, %d written, %d failed, more=%s",
        name,
        len(rows),
        written,
        failed,
        remaining,
    )
    return {
        "graph_name": name,
        "pending": len(rows),
        "written": written,
        "failed": failed,
        "remaining": remaining,
    }


# From a candidate record to its workflow node (an index probe on uuid), keeping
# the record only when its own version is one the workflow says is in force.
# `seen` is how many candidates the window held, counted before any filter, so
# the caller can tell a full window (there may be more to find) from an
# exhausted one.
_CANDIDATES = (
    "WITH collect({n: n, score: score}) AS candidates "
    "WITH candidates, size(candidates) AS seen "
    "UNWIND candidates AS candidate "
    "WITH candidate.n AS n, candidate.score AS score, seen "
)
# A record is in force when the node it names is of the kind that carries the
# pointer for the record's own kind (a workflow for a map, an SOP or a chart; a
# record set for a summary; a law for a section of it), in the same tenant and
# engagement, and that pointer holds the record's version. The kinds are
# written into the query from the model's own tables (they are enum values,
# never caller text): without them any row carrying a pointer-shaped property
# could stand in for a workflow, and one pointer could put another kind's
# records in force.
# A summary is served only while it is being kept current. An approved map is
# in force until another is approved, however long that takes. A count is
# different: it is only as good as the last time it was made, and nothing
# withdraws one that is no longer being made again (the setting switched off,
# a run that fails night after night). So the record set says when its
# summaries were counted (RECORD_SET_COUNTED_AT, restated on every run that
# counts them), and one counted longer ago than `summaries_served_for_seconds`
# stops feeding retrieval by itself, with nothing to remember to do. A time
# ahead of the clock by more than a day is no time at all (the route refuses
# to write one, and a node written some other way is not served either).
_POINTER_ALSO = {
    "current_summary_version_id": (
        f"w.{RECORD_SET_COUNTED_AT} >= $counted_after AND w.{RECORD_SET_COUNTED_AT} <= $counted_before"
    ),
}
COUNTED_AT_AHEAD_SECONDS = 24 * 3600
_POINTER_BRANCHES = " OR ".join(
    f"(w.projection_source_kind = '{POINTER_ANCHOR_KIND[pointer].value}' "
    "AND n.projection_source_kind IN ["
    + ", ".join(f"'{kind.value}'" for kind in sorted(kinds, key=lambda kind: kind.value))
    + f"] AND n.projection_source_version_id = w.{pointer}"
    + (f" AND {_POINTER_ALSO[pointer]}" if pointer in _POINTER_ALSO else "")
    + ")"
    for pointer, kinds in POINTER_RECORD_KINDS.items()
)
_IN_FORCE = (
    "n.group_id = $group_id AND n.engagement_id IN $engagement_ids "
    f"MATCH (w:{PROJECTION_NODE_LABEL} {{uuid: n.{RECORD_WORKFLOW_PROPERTY}}}) "
    "WHERE w.group_id = $group_id AND w.engagement_id = n.engagement_id "
    f"AND ({_POINTER_BRANCHES}) "
    f"RETURN n.uuid AS uuid, coalesce(n.{RECORD_TEXT_PROPERTY}, n.{LAW_TEXT_PROPERTY}) AS text, "
    "n.projection_source_kind AS kind, n.projection_source_id AS source_id, "
    "n.projection_source_version_id AS version_id, n.engagement_id AS engagement_id, "
    "n.workflow_id AS workflow_id, n.workflow_name AS workflow_name, score, seen"
)


# A summary is found by what it means, never by the words it shares. It is a
# statement about a whole engagement, so it shares common words and a great
# many names with almost any question, and through the keyword leg it turned
# up where it had no business: on a rehearsal with real records (2026-11-11),
# among the five records returned for 22 of 190 questions that were not about
# counts, usually in second place. Found by meaning only, 7 of 190, each of
# them on the question's own subject, and every counting question still got
# its count first. The kinds are the ones a record set puts in force, read
# from the model's own tables.
#
# A summary also never takes a place from an approved record, at any stage.
# Neither leg keeps one: the keyword leg leaves them out before its window is
# cut, the vector leg after its window is read (so the window is judged full or
# not by what the index handed over, as before). They are read on their own
# (_SUMMARY_READ), and placed among the records at the rank they would have had
# in the vector leg. So the approved records a question gets are the ones it got
# before there were summaries, and a summary comes beside them (see the end of
# search_records). Measured on a rehearsal with real records (2026-11-12, one
# engine state): with summaries kept by the vector leg and cut with the records,
# a summary took a place from an approved record on 13 of 196 questions at five
# results and 14 at ten; read on their own, on none. Kept out of the vector
# leg's pool as well, a summary also cannot push a record out of the pool the
# two legs are fused from.
MEANING_ONLY_KINDS = tuple(
    sorted(
        kind.value
        for pointer, kinds in POINTER_RECORD_KINDS.items()
        if POINTER_ANCHOR_KIND[pointer] is ProjectionSourceKind.RECORD_SET
        for kind in kinds
    )
)
_NOT_BY_KEYWORD = "NOT n.projection_source_kind IN [" + ", ".join(f"'{kind}'" for kind in MEANING_ONLY_KINDS) + "]"
# A section of a law never takes a place from an approved process record either,
# and is not returned unless it is asked for. A law's sections are long, share
# their words with most of an engagement's procedures, and number in the
# hundreds: on a rehearsal with real records (2026-11-13, one engine state, two
# laws of 114 and 201 sections beside 128 processes) a section took a place
# from a process record on 5 of 196 questions at five results and 10 at ten,
# some of them the right authority and some of them noise. No answer was lost,
# but which of the two a reader wants is the reader's to say. So the record
# legs leave law out (the keyword leg before its window is cut, the vector leg
# after its window is read, as for summaries), and law has a search of its own
# (search_law), asked for by name. The kinds are the ones a law puts in force,
# read from the model's own tables. (Since sections stopped carrying the
# ProjectionNode label neither record index holds one; the filters stay, so a
# section can never be returned by a record leg whatever is in the index.)
LAW_KINDS = tuple(
    sorted(
        kind.value
        for pointer, kinds in POINTER_RECORD_KINDS.items()
        if POINTER_ANCHOR_KIND[pointer] is ProjectionSourceKind.LAW_SOURCE
        for kind in kinds
    )
)
_LAW_KIND_LIST = "[" + ", ".join(f"'{kind}'" for kind in LAW_KINDS) + "]"
_NOT_LAW = f"NOT n.projection_source_kind IN {_LAW_KIND_LIST}"
if len(LAW_KINDS) != 1:  # pragma: no cover - a guard on the model, not a runtime path
    raise RuntimeError("_LAW_READ reads one node label; a second law record kind needs a read of its own")
# The law is searched by meaning only. Measured 2026-11-14 on two real laws (114
# and 201 sections) and 24 questions whose right section was known (8 questions
# asked of the law directly, and 16 purposes of procedures whose governing
# section an independent reader confirmed): ranked by meaning alone, the right
# section came first for 20 of the 24 and in the first three for 23. Ranked by
# keyword alone, 1 and 3. The two fused with equal weight, as this search did
# before, 12 and 19; weighted fusions fell in between. Every section's text opens
# with its law's title, so a question that names the law matches every section
# of it by keyword, and the shortest sections win. What is given up: a question
# that quotes a section number is no longer matched by its digits. That was not
# measured.
#
# The law sections in force, nearest first, scored exactly. A law has a few
# hundred sections, so they are compared with the question one by one, as the
# summaries are: the same score as the index's, no turn at the index, and no
# section of a law out of force can crowd a window. The node label is the fixed
# label the model maps the kind to, never caller text.
#
# Only a value the engine holds as a vector is compared: one value of another
# type would make the comparison fail, and the read with it, for every section
# of the graph. Only this service writes a section's vector (callers cannot send
# one, see SERVICE_OWNED_PROPERTIES), so this is a second guard, not the first.
# A vector of the wrong size would fail the read too, and the engine offers no
# cheap way to ask a vector's size; the model marker (which names the size) is
# what keeps one out. A known limit.
#
# Another: this read takes no turn at the index (see vector_turns), because it
# does not use one. Measured by an independent reviewer on a real engine
# (2026-11-14): one search over 50,000 sections of 1,024 dimensions took 0.2 s,
# and 64 of them at once 2.7 s. A few hundred sections is the expected size.
_LAW_READ = (
    f"MATCH (n:{SOURCE_KIND_NODE_LABEL[ProjectionSourceKind(LAW_KINDS[0])]}) "
    f"WHERE n.{LAW_MODEL_PROPERTY} = $marker AND typeOf(n.{LAW_VECTOR_PROPERTY}) = 'Vectorf32' "
    f"WITH n, vec.cosineDistance(n.{LAW_VECTOR_PROPERTY}, vecf32($q)) AS score, 0 AS seen "
    "WHERE "
)
# The most law sections one search of the law returns.
MAX_LAW_RESULTS = 10
# The most summaries a search returns beside the records asked for (see the
# end of search_records).
SUMMARIES_BESIDE_RECORDS = 3
if len(MEANING_ONLY_KINDS) != 1:  # pragma: no cover - a guard on the model, not a runtime path
    raise RuntimeError("_SUMMARY_READ reads one node label; a second summary kind needs a read of its own")
# The summaries in force, nearest first, scored exactly. There are a few dozen
# to an engagement, so they are compared with the question one by one instead
# of through the vector index: the score is the same (the index's score is this
# cosine distance), the read needs no turn at the index (see vector_turns), and
# no summary can take a place in an index window. The node label is the fixed
# label the model maps the kind to, never caller text.
_SUMMARY_READ = (
    f"MATCH (n:{SOURCE_KIND_NODE_LABEL[ProjectionSourceKind(MEANING_ONLY_KINDS[0])]}) "
    "WHERE n.text_emb_model = $marker AND n.text_embedding IS NOT NULL "
    "WITH n, vec.cosineDistance(n.text_embedding, vecf32($q)) AS score, 0 AS seen "
    "WHERE "
)


# Is anything in force at all for the engagements asked about? Only a workflow
# that holds a pointer puts a record in force, so one workflow with a pointer is
# the whole question, and the engagement index answers it in one small read.
# The pointers are written in from the model's own table, never caller text.
_HOLDS_RECORDS = (
    f"MATCH (w:{PROJECTION_NODE_LABEL}) "
    "WHERE w.engagement_id IN $engagement_ids AND w.group_id = $group_id "
    "AND ("
    + " OR ".join(
        f"(w.projection_source_kind = '{POINTER_ANCHOR_KIND[pointer].value}' AND w.{pointer} IS NOT NULL"
        + (f" AND {_POINTER_ALSO[pointer]}" if pointer in _POINTER_ALSO else "")
        + ")"
        for pointer in sorted(POINTER_RECORD_KINDS)
    )
    + ") "
    "RETURN w.uuid LIMIT 1"
)
# The check runs before the two legs and its time is not taken from theirs, so
# it gets a short limit of its own. On a real engine the read took under 10 ms
# across 51 engagements and 100,000 records; running out of time only means
# "search anyway".
PRESENCE_TIMEOUT_MS = 400


def _leg_heads(window: int) -> tuple[str, str]:
    """Each leg up to its bounded window of candidates, before anything is filtered."""
    vector = (
        f"CALL db.idx.vector.queryNodes('{PROJECTION_NODE_LABEL}', 'text_embedding', "
        f"{int(window)}, vecf32($q)) YIELD node AS n, score "
    )
    # The kinds found by meaning only are left out before the window is cut, so they never take a place in it
    # from a record the keyword leg could return.
    keyword = (
        f"CALL db.idx.fulltext.queryNodes('{PROJECTION_NODE_LABEL}', $query) "
        "YIELD node AS n, score "
        f"WITH n, score WHERE {_NOT_BY_KEYWORD} AND {_NOT_LAW} "
        f"WITH n, score ORDER BY score DESC, n.uuid ASC LIMIT {int(window)} "
    )
    return vector, keyword


def record_leg_queries(pool: int, window: int | None = None) -> tuple[str, str]:
    """The Cypher of the two legs for a result pool and a candidate window.

    Each leg takes `window` candidates from its index, then keeps the ones in
    force for the engagements asked about, ordered again after the join: the
    vector procedure scores by distance (nearest first), the keyword procedure
    by relevance (highest first). A summary of a whole engagement is kept by
    neither leg: it is found by what it means, not by the words it shares, and
    read on its own (summary_read_query, MEANING_ONLY_KINDS). Two records with the same score are put in
    order by their id, in the final cut and in the keyword leg's window, so
    which of them is kept does not depend on the order the engine happened to
    hand them over. (Which candidates the vector index puts in its own window
    is the index's to decide and is not something a statement can order.) The vector leg only trusts a vector made by
    the embedding model in use, so a change of model cannot silently compare
    vectors from two spaces.
    """
    pool = int(pool)
    window = int(window if window is not None else pool * _OVERFETCH)
    vector_head, keyword_head = _leg_heads(window)
    vector = (
        f"{vector_head}{_CANDIDATES}WHERE n.text_emb_model = $marker AND {_NOT_BY_KEYWORD} AND {_NOT_LAW} AND {_IN_FORCE} "
        f"ORDER BY score ASC, uuid ASC LIMIT {pool}"
    )
    keyword = f"{keyword_head}{_CANDIDATES}WHERE {_IN_FORCE} ORDER BY score DESC, uuid ASC LIMIT {pool}"
    return vector, keyword


def summary_read_query() -> str:
    """The Cypher of the summaries' own read: those in force, nearest first, no more than are ever returned."""
    return f"{_SUMMARY_READ}{_IN_FORCE} ORDER BY score ASC, uuid ASC LIMIT {SUMMARIES_BESIDE_RECORDS}"


def law_query(pool: int) -> str:
    """The Cypher of the law's one read: every section in force, scored exactly by meaning, nearest first. Two
    sections the same distance away are put in order by their id, so the answer does not depend on read order."""
    return f"{_LAW_READ}{_IN_FORCE} ORDER BY score ASC, uuid ASC LIMIT {int(pool)}"


def record_leg_counts(window: int) -> tuple[str, str]:
    """How many candidates each leg's window holds, asked only when nothing survived the filter."""
    vector_head, keyword_head = _leg_heads(window)
    return f"{vector_head}RETURN count(n) AS seen", f"{keyword_head}RETURN count(n) AS seen"


def _keyword_query(question: str) -> str:
    """An OR of the question's terms, stopwords dropped, specials removed.

    The library's own builder, with no group filter: the record index covers
    the statement only, and the tenant, engagement and currency are filtered on
    the node after the procedure yields it.
    """
    from graphiti_core.driver.falkordb.operations.search_ops import (  # noqa: PLC0415
        _build_falkor_fulltext_query,
    )

    built = (_build_falkor_fulltext_query(question, None) or "").strip()
    if not built or built.endswith("()") or "@group_id" in built:
        return ""
    return built


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _row_to_record(row: list[Any]) -> dict[str, Any]:
    """A returned row as plain strings: a property written as a number must not fail the answer."""
    return {
        "record_id": _text(row[0]),
        "text": _text(row[1]),
        "kind": _text(row[2]),
        "source_id": _text(row[3]),
        "source_version_id": _text(row[4]),
        "engagement_id": _text(row[5]),
        "workflow_id": _text(row[6]),
        "workflow_name": _text(row[7]),
    }


def _read(graph: Any, statement: str, params: dict[str, Any], timeout_ms: int) -> Any:
    """Read-only on purpose: a search of a tenant with no graph errors, it does not create one."""
    return graph.ro_query(statement, params, timeout=timeout_ms)


async def _leg_read(
    graph: Any, leg: str, statement: str, params: dict[str, Any], timeout_ms: int, deadline: float
) -> Any:
    """One read of a leg. A vector read waits its turn (see vector_turns); a keyword read does not.

    The wait for a turn comes out of the leg's own time: it is bounded by the
    read's limit and by what is left before the leg's deadline, so a leg takes
    no longer than it could before turns existed. A read that cannot get its
    turn in that time fails like any other read that ran out of time.
    """
    if leg != "vector":
        return await asyncio.to_thread(_read, graph, statement, params, timeout_ms)
    wait = min(timeout_ms / 1000, deadline - time.monotonic())
    return await vector_turns.take_turn(
        vector_turns.RECORDS, lambda: asyncio.to_thread(_read, graph, statement, params, timeout_ms), wait
    )


async def _leg(
    graph: Any,
    name: str,
    leg: str,
    params: dict[str, Any],
    pool: int,
    timeout_ms: int,
) -> list[list[Any]]:
    """One leg, widening its candidate window until enough records in force
    survive, the window is not full (nothing more to find), the cap is hit, or
    the leg has used its time. Rows found in a narrower window are kept if a
    wider one fails or runs out of time."""
    window = pool * _OVERFETCH
    deadline = time.monotonic() + (2 * timeout_ms) / 1000
    best: list[list[Any]] = []
    index = 0 if leg == "vector" else 1
    while True:
        try:
            rows = (
                await _leg_read(graph, leg, record_leg_queries(pool, window)[index], params, timeout_ms, deadline)
            ).result_set or []
            if rows:
                seen = int(rows[0][-1])
            else:
                # Nothing survived: was the window full of records out of force, or is there nothing there?
                counted = (
                    await _leg_read(graph, leg, record_leg_counts(window)[index], params, timeout_ms, deadline)
                ).result_set
                seen = int(counted[0][0]) if counted and counted[0] else 0
        except Exception:
            if not best:
                raise
            return best
        if len(rows) >= len(best):
            best = rows
        exhausted = seen < window
        if len(best) >= pool or exhausted or window >= MAX_CANDIDATES or time.monotonic() >= deadline:
            if len(best) < pool and not exhausted and window >= MAX_CANDIDATES:
                logger.warning(
                    "[graphiti] record %s leg on %s read %d candidates and kept %d of %d wanted; "
                    "records out of force are crowding the index",
                    leg,
                    name,
                    window,
                    len(best),
                    pool,
                )
            return best
        window = min(window * _WIDEN_FACTOR, MAX_CANDIDATES)


async def _holds_records(graph: Any, name: str, scope: dict[str, Any]) -> bool:
    """Whether the engagements asked about have any record in force.

    False only when the read answered and found none. A read that fails or runs
    out of time says nothing either way, so the search goes on as it would have
    without this check."""
    try:
        found = (await asyncio.to_thread(_read, graph, _HOLDS_RECORDS, scope, PRESENCE_TIMEOUT_MS)).result_set
    except Exception as error:
        logger.warning(
            "[graphiti] record presence check failed on %s error_type=%s; searching anyway",
            name,
            type(error).__name__,
        )
        return True
    return bool(found)


async def search_records(
    client_slug: str,
    engagement_ids: list[str],
    query: str,
    max_results: int,
    graph_name: str | None = None,
) -> list[dict[str, Any]]:
    """The process records in force that best match a question.

    Keyword and meaning, each under its own time budget, fused by reciprocal
    rank. A leg that fails or runs out of time answers empty for this request;
    with both legs empty the answer is an empty list, never an error: records
    are an addition to a search, not a dependency of it.
    """
    if not engagement_ids or not query.strip() or int(max_results) < 1:
        return []
    name = graph_name or graphiti_client._graph_name_for_client(client_slug)
    graph = graphiti_client.get_falkor_db().select_graph(name)

    pool = max(int(max_results) * 2, int(max_results))
    now = int(time.time())
    # What every read is asked with. A summary is served only while its counting run is recent (see _POINTER_ALSO).
    scope = {
        "group_id": name,
        "engagement_ids": list(engagement_ids),
        "counted_after": now - int(settings.summaries_served_for_seconds),
        "counted_before": now + COUNTED_AT_AHEAD_SECONDS,
    }

    # A client's records sit under the engagement that approved them, so a
    # search is often asked of engagements that hold none: a later engagement
    # reading its own, or the other engagements of the one that holds them all.
    # There is nothing to rank then. Without this, each leg would walk its
    # window out to the cap through records of other engagements, the question
    # would be embedded for nothing, and both legs would warn that history is
    # crowding the index when nothing is.
    if not await _holds_records(graph, name, scope):
        return []

    by_id: dict[str, dict[str, Any]] = {}
    # How near each vector-leg record and each summary is to the question (cosine distance, lower is nearer).
    distance: dict[str, float] = {}

    def collect(rows: list[list[Any]], keep_distance: bool = False) -> list[str]:
        order: list[str] = []
        for row in rows:
            if not row or not row[0]:
                continue
            record_id = _text(row[0])
            by_id.setdefault(record_id, _row_to_record(row))
            if keep_distance:
                try:
                    distance[record_id] = float(row[-2])
                except (TypeError, ValueError):
                    distance[record_id] = float("inf")
            order.append(record_id)
        return order

    async def summary_read(qvec: list[float]) -> list[str]:
        """The summaries in force nearest the question. Never fails the search: a failure means no summaries."""
        began = time.monotonic()
        try:
            rows = (
                await asyncio.to_thread(
                    _read, graph, summary_read_query(), {**scope, "q": qvec, "marker": _marker()},
                    settings.search_vector_timeout_ms,
                )
            ).result_set or []
            return collect(rows, keep_distance=True)
        except Exception as error:
            logger.warning(
                "[graphiti] record summary read skipped on %s error_type=%s after %d ms",
                name,
                type(error).__name__,
                int((time.monotonic() - began) * 1000),
            )
            return []

    summary_order: list[str] = []

    async def vector_leg() -> list[str]:
        began = time.monotonic()
        try:
            embedder = graphiti_client._create_embedder()
            if embedder is None:
                return []
            # The question's vector is made once and kept: the provider does not
            # give the same vector for the same text twice (see question_vectors).
            qvec = await question_vectors.question_vector(embedder, name, query)
            rows, summaries = await asyncio.gather(
                _leg(
                    graph,
                    name,
                    "vector",
                    {**scope, "q": qvec, "marker": _marker()},
                    pool,
                    settings.search_vector_timeout_ms,
                ),
                summary_read(qvec),
            )
            summary_order.extend(summaries)
            return collect(rows, keep_distance=True)
        except Exception as error:
            logger.warning(
                "[graphiti] record vector leg skipped on %s error_type=%s after %d ms",
                name,
                type(error).__name__,
                int((time.monotonic() - began) * 1000),
            )
            return []

    async def keyword_leg() -> list[str]:
        terms = _keyword_query(query)
        if not terms:
            return []
        began = time.monotonic()
        try:
            return collect(
                await _leg(
                    graph,
                    name,
                    "keyword",
                    {**scope, "query": terms},
                    pool,
                    settings.search_bm25_timeout_ms,
                )
            )
        except Exception as error:
            logger.warning(
                "[graphiti] record keyword leg skipped on %s error_type=%s after %d ms",
                name,
                type(error).__name__,
                int((time.monotonic() - began) * 1000),
            )
            return []

    vector_order, keyword_order = await asyncio.gather(vector_leg(), keyword_leg())

    scores: dict[str, float] = {}
    for ordered in (vector_order, keyword_order):
        for rank, record_id in enumerate(ordered):
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (rank + _RRF_K)
    # A summary goes where it would have stood in the vector leg: just ahead of
    # the first record farther from the question than it, after any record as
    # near or nearer, and after the summaries nearer than it. The records' own
    # ranks are not moved by it, so a record's place in the answer is what it
    # was without summaries. (Half a rank ahead of that record; a thousandth of
    # a rank apart from one summary to the next, and there are never more than
    # SUMMARIES_BESIDE_RECORDS.)
    for place, summary_id in enumerate(summary_order):
        nearer = sum(1 for record_id in vector_order if distance.get(record_id, float("inf")) <= distance[summary_id])
        scores[summary_id] = 1.0 / (nearer - 0.5 + place / 1000 + _RRF_K)
    ranked = sorted(scores, key=lambda record_id: scores[record_id], reverse=True)
    # A summary never takes a place from an approved record. The places asked for
    # go to the best records that are not summaries. A summary is returned as
    # well, at its own rank, only where it ranked ahead of the last of those, and
    # no more than SUMMARIES_BESIDE_RECORDS of them: so a question a count
    # answers gets the count, and every other question gets the records it got
    # before there were counts.
    chosen: list[dict[str, Any]] = []
    records = summaries = 0
    for record_id in ranked:
        found = by_id[record_id]
        if found["kind"] in MEANING_ONLY_KINDS:
            if summaries < SUMMARIES_BESIDE_RECORDS:
                chosen.append(found)
                summaries += 1
            continue
        chosen.append(found)
        records += 1
        if records >= int(max_results):
            break
    return chosen


async def search_law(
    client_slug: str,
    engagement_ids: list[str],
    query: str,
    max_results: int,
    graph_name: str | None = None,
) -> list[dict[str, Any]]:
    """The sections of the laws in force nearest a question in meaning, and nothing else.

    Asked for by name, beside a record search or instead of one. By meaning
    only (see _LAW_READ for why), under the vector read's time limit. A read
    that fails or runs out of time answers an empty list, never an error.
    """
    wanted = min(int(max_results), MAX_LAW_RESULTS)
    if not engagement_ids or not query.strip() or wanted < 1:
        return []
    name = graph_name or graphiti_client._graph_name_for_client(client_slug)
    graph = graphiti_client.get_falkor_db().select_graph(name)
    now = int(time.time())
    scope = {
        "group_id": name,
        "engagement_ids": list(engagement_ids),
        "counted_after": now - int(settings.summaries_served_for_seconds),
        "counted_before": now + COUNTED_AT_AHEAD_SECONDS,
    }
    began = time.monotonic()
    try:
        embedder = graphiti_client._create_embedder()
        if embedder is None:
            return []
        qvec = await question_vectors.question_vector(embedder, name, query)
        rows = (
            await asyncio.to_thread(
                _read, graph, law_query(wanted), {**scope, "q": qvec, "marker": _marker()},
                settings.search_vector_timeout_ms,
            )
        ).result_set or []
    except Exception as error:
        logger.warning(
            "[graphiti] law read skipped on %s error_type=%s after %d ms",
            name, type(error).__name__, int((time.monotonic() - began) * 1000),
        )
        return []
    # In the order the read put them (nearest first, then by id). Only sections: the read can return nothing else,
    # and this keeps it so whatever the read becomes.
    found = [_row_to_record(row) for row in rows if row and row[0]]
    return [record for record in found if record["kind"] in LAW_KINDS][:wanted]
