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
ProjectionNode rows only and cannot return, or be confused with, an extracted
fact.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config import settings
from app.models.projection import (
    PROJECTION_NODE_LABEL,
    RECORD_TEXT_PROPERTY,
    RECORD_WORKFLOW_PROPERTY,
)
from app.services import graphiti_client

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


# Only a record that retrieval could ever return: it has a statement and names
# its workflow. Pending means it has no vector, or its vector was made by a
# different model or size than the one in use now. A statement this model has
# refused (with the provider answering) is set aside for a while: not asked
# again until REFUSED_RETRY_AFTER_SECONDS have passed, it is restated (which
# clears the mark), or the model changes. Otherwise the same refusals would
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


MARK_REFUSED_QUERY = (
    f"MATCH (n:{PROJECTION_NODE_LABEL} {{uuid: $uuid, group_id: $group_id, "
    "engagement_id: $engagement_id}) "
    f"WHERE n.{RECORD_TEXT_PROPERTY} = $text "
    "SET n.text_emb_failed = $marker, n.text_emb_failed_at = $now "
    "RETURN count(n) AS marked"
)


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
            SET_VECTOR_QUERY,
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
                        MARK_REFUSED_QUERY,
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
_IN_FORCE = (
    "n.group_id = $group_id AND n.engagement_id IN $engagement_ids "
    f"MATCH (w:{PROJECTION_NODE_LABEL} {{uuid: n.{RECORD_WORKFLOW_PROPERTY}}}) "
    "WHERE w.group_id = $group_id AND w.engagement_id = n.engagement_id "
    "AND (n.projection_source_version_id = w.current_map_version_id "
    "OR n.projection_source_version_id = w.current_sop_version_id) "
    f"RETURN n.uuid AS uuid, n.{RECORD_TEXT_PROPERTY} AS text, "
    "n.projection_source_kind AS kind, n.projection_source_id AS source_id, "
    "n.projection_source_version_id AS version_id, n.engagement_id AS engagement_id, "
    "n.workflow_id AS workflow_id, n.workflow_name AS workflow_name, score, seen"
)


def _leg_heads(window: int) -> tuple[str, str]:
    """Each leg up to its bounded window of candidates, before anything is filtered."""
    vector = (
        f"CALL db.idx.vector.queryNodes('{PROJECTION_NODE_LABEL}', 'text_embedding', "
        f"{int(window)}, vecf32($q)) YIELD node AS n, score "
    )
    keyword = (
        f"CALL db.idx.fulltext.queryNodes('{PROJECTION_NODE_LABEL}', $query) "
        "YIELD node AS n, score "
        f"WITH n, score ORDER BY score DESC LIMIT {int(window)} "
    )
    return vector, keyword


def record_leg_queries(pool: int, window: int | None = None) -> tuple[str, str]:
    """The Cypher of the two legs for a result pool and a candidate window.

    Each leg takes `window` candidates from its index, then keeps the ones in
    force for the engagements asked about, ordered again after the join: the
    vector procedure scores by distance (nearest first), the keyword procedure
    by relevance (highest first). The vector leg only trusts a vector made by
    the embedding model in use, so a change of model cannot silently compare
    vectors from two spaces.
    """
    pool = int(pool)
    window = int(window if window is not None else pool * _OVERFETCH)
    vector_head, keyword_head = _leg_heads(window)
    vector = (
        f"{vector_head}{_CANDIDATES}WHERE n.text_emb_model = $marker AND {_IN_FORCE} "
        f"ORDER BY score ASC LIMIT {pool}"
    )
    keyword = f"{keyword_head}{_CANDIDATES}WHERE {_IN_FORCE} ORDER BY score DESC LIMIT {pool}"
    return vector, keyword


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
                await asyncio.to_thread(_read, graph, record_leg_queries(pool, window)[index], params, timeout_ms)
            ).result_set or []
            if rows:
                seen = int(rows[0][-1])
            else:
                # Nothing survived: was the window full of records out of force, or is there nothing there?
                counted = (
                    await asyncio.to_thread(_read, graph, record_leg_counts(window)[index], params, timeout_ms)
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
    scope = {"group_id": name, "engagement_ids": list(engagement_ids)}
    by_id: dict[str, dict[str, Any]] = {}

    def collect(rows: list[list[Any]]) -> list[str]:
        order: list[str] = []
        for row in rows:
            if not row or not row[0]:
                continue
            record_id = _text(row[0])
            by_id.setdefault(record_id, _row_to_record(row))
            order.append(record_id)
        return order

    async def vector_leg() -> list[str]:
        began = time.monotonic()
        try:
            embedder = graphiti_client._create_embedder()
            if embedder is None:
                return []
            qvec = await embedder.create(input_data=[query.replace("\n", " ")])
            return collect(
                await _leg(
                    graph,
                    name,
                    "vector",
                    {**scope, "q": qvec, "marker": _marker()},
                    pool,
                    settings.search_vector_timeout_ms,
                )
            )
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
    ranked = sorted(scores, key=lambda record_id: scores[record_id], reverse=True)
    return [by_id[record_id] for record_id in ranked[: int(max_results)]]
