"""Graphiti core wrapper — graph connections, ingestion, and provenance search."""

import ast
import asyncio
import logging
import time
import uuid as uuidlib
from dataclasses import dataclass, replace
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, create_model
from redis.exceptions import ResponseError
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType as GraphitiEpisodeType
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig

from app.config import settings
from app.services.indexed_falkor import FULLTEXT_OVERFETCH, IndexedFalkorDriver, ensure_node_vector_index
from app.graph_names import graph_name_for_client, segment_graph_name
from app.provenance_contract import STRUCTURALLY_ANCHORED_MODES

logger = logging.getLogger("graphiti_service")

# Cache of initialized Graphiti clients per graph name
_clients: dict[str, Graphiti] = {}

# Graphs for which we've already ensured the RELATES_TO fact_embedding vector
# index exists this process-lifetime (so we don't re-issue CREATE every search).
_edge_vindex_ensured: set[str] = set()

_MAX_EPISODE_STORAGE_BYTES = 100_000
_MAX_EPISODES_PER_FACT = 64


@dataclass(frozen=True)
class ResolvedEpisodeAnchor:
    episode_uuid: str
    episode_name: str
    source_description: str
    source_type: str | None
    source_id: str | None
    engagement_id: str | None
    episode_type: str | None
    anchor_mode: str | None
    producer_contract_version: str | None
    valid_at: datetime | None
    provenance_write_state: str | None = None
    malformed: bool = False


@dataclass(frozen=True)
class ResolvedSearchEdge:
    fact_id: str
    subject_uuid: str
    subject_name: str
    predicate: str
    object_uuid: str
    object_name: str
    fact: str
    episode_uuids: tuple[str, ...]
    sources: tuple[ResolvedEpisodeAnchor, ...]
    valid_at: datetime | None
    invalid_at: datetime | None
    expired_at: datetime | None
    malformed: bool = False


RetrievalPath = Literal["fast", "hybrid_fallback"]


class AcceptanceProbeReadError(RuntimeError):
    """Fixed-boundary failure for a dedicated read-only probe process."""


def _select_existing_probe_graph(db: Any, graph_name: str) -> Any:
    """Select only an exact graph already reported by ``GRAPH.LIST``.

    FalkorDB's normal graph selection followed by ``GRAPH.QUERY`` can create an
    empty graph key. Probe mode must never turn a missing-tenant observation
    into graph state, so membership is proven before selection.
    """

    names = db.list_graphs()
    if not isinstance(names, (list, tuple, set, frozenset)) or graph_name not in names:
        raise AcceptanceProbeReadError("acceptance probe graph is unavailable")
    return db.select_graph(graph_name)


def _graph_read(
    graph: Any,
    query: str,
    params: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
) -> Any:
    """Use FalkorDB's read-only command in acceptance-probe processes.

    `timeout_ms` is FalkorDB's per-query TIMEOUT: the server abandons the
    query and raises, instead of the caller waiting out the socket bound.
    """

    command = graph.ro_query if settings.graphiti_acceptance_probe_mode else graph.query
    if timeout_ms is None:
        return command(query, params)
    return command(query, params, timeout=timeout_ms)


class SearchBudgetExceeded(RuntimeError):
    """A fast-search leg exceeded its per-query budget; the request answers empty rather than late."""


def _is_query_timeout(error: BaseException, elapsed_ms: float, budget_ms: int) -> bool:
    """True only for a FalkorDB TIMEOUT that actually ran out.

    FalkorDB abandons a query past its TIMEOUT with the text "Query timed out".
    A rejected TIMEOUT argument (above the server's TIMEOUT_MAX, or a build
    that refuses it) fails at once with text naming the parameter and no
    "timed out"; a redis socket timeout reads "Timeout reading from socket".
    A connection error also carries "timed out" in redis-py's wording
    ("Error 60 connecting to host. Operation timed out.") when the database
    restarts mid-request. None of these is a budget: they must keep the
    fallback behaviour they had, so the error has to be the server's own
    reply, and both its text and the elapsed time have to say the budget
    ran out.
    """
    return (
        isinstance(error, ResponseError)
        and "timed out" in str(error).lower()
        and elapsed_ms >= 0.9 * budget_ms
    )


async def _graph_read_async(
    graph: Any,
    query: str,
    params: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
) -> Any:
    """Run a synchronous graph read WITHOUT charging its duration to the loop.

    The FalkorDB handle is synchronous, so calling it inline on an async path
    stops every other request for as long as the query takes. On 2026-08-28 that
    left this service unable to answer its own /health for over twenty minutes —
    and the second time it happened there was no ingestion running at all, only
    the periodic health probe going through the search path.

    The socket timeout bounds how long one call can take; this keeps that time
    from being taken out of everyone else's.
    """
    return await asyncio.to_thread(_graph_read, graph, query, params, timeout_ms)


def _parse_dt(v: Any) -> datetime | None:
    """Best-effort parse of a stored temporal value to datetime, else None.

    Graphiti stores valid_at/invalid_at/expired_at as ISO strings. Never raises —
    a bad/odd value yields None rather than failing a whole search response."""
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _uuid_string(value: Any) -> str | None:
    """Return a canonical UUID string or ``None`` without coercing bad input."""

    if not isinstance(value, (str, uuidlib.UUID)):
        return None
    try:
        return str(uuidlib.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def _episode_uuid_list(value: Any) -> tuple[tuple[str, ...], bool]:
    """Normalize historical episode-list storage while exposing malformed data."""

    candidate = value
    if candidate is None:
        return (), True
    if isinstance(candidate, str):
        if len(candidate) > _MAX_EPISODE_STORAGE_BYTES:
            return (), False
        try:
            storage_bytes = len(candidate.encode("utf-8"))
        except UnicodeError:
            return (), False
        if storage_bytes > _MAX_EPISODE_STORAGE_BYTES:
            return (), False
        try:
            candidate = ast.literal_eval(candidate)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return (), False
    if not isinstance(candidate, (list, tuple)):
        return (), False
    # A stricter raw-entry bound avoids walking an adversarial native list and
    # never truncates authority. Duplicate-heavy historical rows can be repaired
    # by the governed backfill rather than silently normalized here.
    if len(candidate) > _MAX_EPISODES_PER_FACT:
        return (), False
    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidate:
        episode_uuid = _uuid_string(item)
        if not episode_uuid:
            return (), False
        if episode_uuid in seen:
            return (), False
        normalized.append(episode_uuid)
        seen.add(episode_uuid)
    return tuple(normalized), True


def _nonempty_string(value: Any, maximum: int = 16_000) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > maximum
        or any(
            (ord(character) < 32 and character not in "\t\n\r")
            or ord(character) == 127
            for character in normalized
        )
    ):
        return None
    return normalized


def _graph_name_for_client(client_slug: str) -> str:
    """Backwards-compatible import surface for the central tenant mapper."""
    return graph_name_for_client(client_slug)


def _segment_graph_name(industry: str) -> str:
    """Backwards-compatible import surface for the central segment mapper."""
    return segment_graph_name(industry)


def _load_entity_types() -> list[dict[str, str]]:
    """Load entity types from config.yaml."""
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        return config.get("entity_types", [])
    except Exception as e:
        logger.warning(f"Failed to load entity types from config.yaml: {e}")
        return []


# Built once. The YAML is read from disk and turned into model classes, and
# doing that per episode would re-read and re-build 16 classes on every ingest.
_entity_type_models: dict[str, type[BaseModel]] | None = None


def entity_type_models() -> dict[str, type[BaseModel]]:
    """The domain taxonomy, as the model classes graphiti's extractor wants.

    config.yaml has said "These define what Graphiti extracts from episodes"
    since it was written, and they did not: `_load_entity_types` had ZERO
    callers and `add_episode` was never given an `entity_types` argument, so
    all 16 types were inert and extraction ran untyped.

    graphiti-core takes `dict[str, type[BaseModel]]` and uses each class's
    docstring as the description shown to the extraction model, so the YAML
    description becomes the docstring rather than a field.

    Fail-soft: a malformed or missing taxonomy returns {} and extraction falls
    back to untyped, which is exactly today's behaviour. A broken config file
    must not stop ingestion.
    """
    global _entity_type_models
    if _entity_type_models is not None:
        return _entity_type_models

    models: dict[str, type[BaseModel]] = {}
    for entry in _load_entity_types():
        name = (entry or {}).get("name")
        if not name or not isinstance(name, str):
            continue
        description = (entry or {}).get("description") or name
        # Attributes are optional string fields the extractor fills from the
        # episode text and stores on the node. The people types carry the
        # platform user id (ruled 2026-09-11: the durable key for a person,
        # since two user records can share a name) and the title on record;
        # without a field the extractor drops the id from the summary.
        fields: dict[str, tuple[type, Any]] = {}
        for attribute in (entry or {}).get("attributes") or []:
            attr_name = (attribute or {}).get("name")
            if not attr_name or not isinstance(attr_name, str) or not attr_name.isidentifier():
                continue
            fields[attr_name] = (
                str | None,
                Field(default=None, description=(attribute or {}).get("description") or attr_name),
            )
        # create_model rather than a literal class per entry: the taxonomy is
        # data, and hand-writing 16 near-identical classes would put it in two
        # places.
        model = create_model(name, __doc__=description, **fields)  # type: ignore[call-overload]
        models[name] = model

    _entity_type_models = models
    if models:
        logger.info(f"[graphiti] entity taxonomy active: {len(models)} types")
    else:
        logger.warning("[graphiti] entity taxonomy EMPTY — extraction will be untyped")
    return models


def reset_entity_type_models() -> None:
    """Drop the cached taxonomy so the next call re-reads config.yaml (tests)."""
    global _entity_type_models
    _entity_type_models = None


def _create_llm_client() -> AnthropicClient:
    """Create the Anthropic LLM client for entity extraction.

    Model comes from settings.graphiti_llm_model (GRAPHITI_LLM_MODEL), which must
    be a string graphiti-core's token map recognizes, or the output silently caps
    at 16384. Default is claude-sonnet-4-5-latest (recognized, 65536 cap).
    """
    return AnthropicClient(
        config=LLMConfig(
            api_key=settings.anthropic_api_key,
            model=settings.graphiti_llm_model,
        )
    )


def _create_embedder():
    """Return an explicit embedder.

    Voyage when VOYAGE_API_KEY is set (removes the implicit OpenAI default that is
    currently the single embedding point of failure for every client's graph);
    otherwise None, which Graphiti falls back to its OpenAI default for — so this
    is INERT and behavior-identical until a Voyage key is configured.

    Note: Voyage and OpenAI produce different vector spaces, so cutting over to
    Voyage requires re-embedding existing graphs before retrieval works again.
    """
    if not settings.voyage_api_key:
        return None
    # Lazy import: the voyageai SDK is only needed when Voyage is actually used.
    from graphiti_core.embedder.voyage import VoyageAIEmbedder, VoyageAIEmbedderConfig

    return VoyageAIEmbedder(
        config=VoyageAIEmbedderConfig(
            api_key=settings.voyage_api_key,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        )
    )


# Process-wide FalkorDB handle.
#
# Constructing FalkorDB() opens a connection pool. Twelve call sites built a
# fresh one per request and none of them closed it. `_search_fast` runs on EVERY
# search — which includes every kg-health probe and every entity-resolution step
# inside ingestion — and /graph/nodes-and-edges, /structured, /projection and
# five admin routes did the same. Each request leaked a pool until FalkorDB hit
# its client limit and began refusing every query with "Too many connections".
#
# Observed 2026-08-27: FalkorDB Live and unrestarted since June 15 with a
# healthy disk, yet every search timed out at the caller's 8s ceiling and all
# five kg-ingest jobs failed after exhausting three 300s retries, each recorded
# only as "Graphiti ingestion returned null".
#
# falkordb-py wraps redis-py, whose client is thread-safe and pools internally,
# so a single handle per process is both correct and what the library expects.
_falkor_db: Any = None


def get_falkor_db() -> Any:
    """Return the shared FalkorDB handle, creating it on first use."""
    global _falkor_db
    if _falkor_db is None:
        from falkordb import FalkorDB as _FalkorDB

        _falkor_db = _FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
            # MUST be finite. This handle is SYNCHRONOUS and is used from inside
            # async paths, so an untimed call blocks the entire event loop for
            # as long as FalkorDB takes to answer — unbounded, if FalkorDB is
            # wedged. On 2026-08-28 this service could not serve even /health (a
            # 3s redis ping) while one extraction held the loop.
            socket_connect_timeout=settings.falkordb_socket_timeout_seconds,
            socket_timeout=settings.falkordb_socket_timeout_seconds,
        )
    return _falkor_db


def reset_falkor_db() -> None:
    """Drop the shared handle so the next call reconnects.

    The handle is process-wide by design — one pool per process is the point.
    That makes its lifetime explicit state, so anything that must not reuse an
    existing connection has to say so: tests that inject their own fake
    FalkorDB, and any caller that has just invalidated the connection.
    """
    global _falkor_db
    _falkor_db = None


# The ASYNC FalkorDB client graphiti-core writes through.
#
# graphiti-core builds its own as FalkorDB(host, port, username, password) — and
# falkordb.asyncio.FalkorDB defaults socket_timeout and socket_connect_timeout
# to None. No timeout at all. A query that never comes back therefore hangs the
# coroutine FOREVER: no exception, no completion, no cancellation.
#
# That is why ingestion never finished. Every episode attempted on 2026-08-28/29
# ended as "no outcome recorded within 3600s; the task died without reporting" —
# not slow, not failing, simply never returning. Each ceiling raised during the
# day only changed when the hang was noticed, never prevented it.
#
# The synchronous handle was given a timeout in #23, but that is a DIFFERENT
# client; the driver performing the actual ingest writes still had none.
#
# FalkorDriver accepts an existing instance via falkor_db=, so we inject a
# bounded one. The instance is shared across graphs on purpose — the driver
# selects its own database per call.
# Diagnostics must fail fast. See _log_slow_queries.
SLOWLOG_TIMEOUT_SECONDS = 10

_async_falkor_db: Any = None


def new_async_falkor_db() -> Any:
    """Build a NEW async FalkorDB client with finite timeouts.

    Deliberately not shared. The bounded timeout above is what turns "hangs
    forever" into a raised TimeoutError, and it must stay — but a timeout is a
    CANCELLED read, and redis-py asyncio leaves the un-read reply sitting in
    that connection's buffer. The next borrower of that pooled connection reads
    the previous command's reply, desyncs, and times out in turn. One timeout
    therefore poisoned every subsequent query through the shared pool, which is
    why a bare `MATCH (n) RETURN count(n)` could not complete against a server
    that was idle, ~100 keys, 53MB, and logging no slow query at all — while
    _probe_falkordb, which builds a FRESH client per call, answered in 1.5s.

    A client per graph keeps the blast radius at one graph, and evict_client()
    below discards even that one rather than inheriting it.
    """
    from falkordb.asyncio import FalkorDB as _AsyncFalkorDB

    return _AsyncFalkorDB(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
        socket_timeout=settings.falkordb_socket_timeout_seconds,
        socket_connect_timeout=settings.falkordb_socket_timeout_seconds,
    )


def get_async_falkor_db() -> Any:
    """Back-compat alias; each call returns a NEW bounded client."""
    return new_async_falkor_db()


def reset_async_falkor_db() -> None:
    """Drop the shared async client so the next call reconnects (tests)."""
    global _async_falkor_db
    _async_falkor_db = None


def _create_driver(graph_name: str) -> FalkorDriver:
    """Create a FalkorDB driver targeting a specific named graph.

    IndexedFalkorDriver, not FalkorDriver: graphiti's own entity dedup is a full
    Entity scan with an inline 1024-dim cosine, run once per extracted entity.
    See app/services/indexed_falkor.py.
    """
    return IndexedFalkorDriver(falkor_db=new_async_falkor_db(), database=graph_name)


async def log_graph_census() -> None:
    """Enumerate every graph with its node/edge counts, once, at startup.

    G9 ("knowledge census") is ratified but has never been run, and the last
    count anyone recorded was 2026-06-15 (8 graphs pruned to 3). Nothing since
    reports what graphs EXIST, so orphans -- which have appeared twice from
    naming drift, `client_test` and a phantom `client_tribal_gaming` built from
    an industry string used as a slug -- are invisible until someone looks.

    Read-only and best-effort: a census must never keep the service from
    starting. Runs once per boot rather than per request, because counts scan.
    """
    try:
        def _census() -> list[tuple[str, int, int]]:
            db = get_falkor_db()
            rows: list[tuple[str, int, int]] = []
            for name in sorted(db.list_graphs() or []):
                g = db.select_graph(name)
                nodes = g.query("MATCH (n) RETURN count(n)").result_set[0][0]
                edges = g.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
                rows.append((name, int(nodes), int(edges)))
            return rows

        rows = await asyncio.wait_for(asyncio.to_thread(_census), timeout=120)
    except Exception as exc:  # noqa: BLE001 - never block startup on a census
        logger.warning(f"[graphiti] CENSUS unavailable: {type(exc).__name__}")
        return

    if not rows:
        logger.warning("[graphiti] CENSUS: no graphs found")
        return
    for name, nodes, edges in rows:
        # "empty" is the interesting state: a graph that exists but holds
        # nothing is either freshly provisioned or an orphan, and the two are
        # indistinguishable without this line.
        state = "EMPTY" if nodes == 0 and edges == 0 else "populated"
        logger.warning(f"[graphiti] CENSUS {name}: nodes={nodes} edges={edges} {state}")
    logger.warning(f"[graphiti] CENSUS total_graphs={len(rows)}")


async def _log_memory_usage() -> None:
    """Report what FalkorDB is actually using, versus what it is allowed.

    Its own short-timeout client, for the same reason the slow log has one: a
    diagnostic must never wait as long as the thing it is describing.
    """
    def _read() -> dict:
        import redis as _redis

        r = _redis.Redis(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
            socket_connect_timeout=SLOWLOG_TIMEOUT_SECONDS,
            socket_timeout=SLOWLOG_TIMEOUT_SECONDS,
            decode_responses=True,
        )
        try:
            return dict(r.info("memory"))
        finally:
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass

    try:
        info = await asyncio.wait_for(
            asyncio.to_thread(_read), timeout=SLOWLOG_TIMEOUT_SECONDS + 5
        )
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        logger.warning(f"[graphiti] memory probe unavailable: {type(exc).__name__}")
        return

    logger.warning(
        "[graphiti] MEMORY used=%s peak=%s rss=%s maxmemory=%s frag=%s",
        info.get("used_memory_human"),
        info.get("used_memory_peak_human"),
        info.get("used_memory_rss_human"),
        info.get("maxmemory_human"),
        info.get("mem_fragmentation_ratio"),
    )


async def _log_slow_queries(graph_name: str, top: int = 5) -> None:
    """Report WHICH query was slow, rather than inferring it from timings.

    Two days were spent reasoning about which Cypher statement was responsible
    from failure DURATIONS alone -- entity dedup was the leading candidate, on
    circumstantial evidence, and indexing it did not change the outcome.
    FalkorDB keeps a per-graph slow log; ask it instead of guessing.

    Its OWN short-lived client with a SHORT timeout, deliberately. The first
    version of this reused the shared handle, which carries a 900s socket
    timeout -- so a diagnostic running on an already-failed FalkorDB call sat
    waiting fifteen minutes on the very dependency that had just died, and the
    task was torn down before it logged anything at all. A diagnostic that can
    outlive the failure it is describing is not a diagnostic.
    """
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_read_slowlog, graph_name), timeout=SLOWLOG_TIMEOUT_SECONDS + 5
        )
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        logger.warning(
            f"[graphiti] slowlog unavailable for {graph_name}: {type(exc).__name__} "
            f"(it must never wait as long as the call that failed)"
        )
        return

    if not raw:
        logger.warning(
            f"[graphiti] slowlog for {graph_name} is EMPTY -- the slow work is not a "
            f"single logged query (a server-side abort, or time spent outside Cypher)"
        )
        return

    try:
        ranked = sorted(raw, key=lambda e: float(e[3]), reverse=True)[:top]
    except Exception:  # noqa: BLE001 - shape varies by version
        ranked = list(raw)[-top:]
    for e in ranked:
        # Duration FIRST. Logging the raw entry put the query text ahead of the
        # duration and then truncated -- so the SLOWEST entry, the one with the
        # longest query, was the one whose timing got cut off. The field that
        # ranks the list must survive truncation.
        try:
            ms = float(e[3])
            query = " ".join(str(e[2]).split())
        except Exception:  # noqa: BLE001 - shape varies by version
            logger.warning(f"[graphiti] SLOW {graph_name}: {str(e)[:400]}")
            continue
        logger.warning(f"[graphiti] SLOW {graph_name}: {ms / 1000:.1f}s :: {query[:300]}")


def _read_slowlog(graph_name: str) -> list:
    """One short-lived, short-timeout connection. Never the shared handle."""
    import redis as _redis

    r = _redis.Redis(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
        socket_connect_timeout=SLOWLOG_TIMEOUT_SECONDS,
        socket_timeout=SLOWLOG_TIMEOUT_SECONDS,
        decode_responses=True,
    )
    try:
        return list(r.execute_command("GRAPH.SLOWLOG", graph_name) or [])
    finally:
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass


async def evict_client(client_slug: str) -> None:
    """Discard the cached Graphiti client (and its pool) for one graph.

    Called when a query through it failed. Without this the cached client keeps
    handing out the same desynced connections and every later episode fails the
    same way, recoverable only by restarting the process — which is exactly the
    shape the outage took.
    """
    graph_name = _graph_name_for_client(client_slug)
    client = _clients.pop(graph_name, None)
    if client is None:
        return
    try:
        await client.close()
    except Exception as exc:  # noqa: BLE001 - closing a broken pool may itself fail
        logger.debug(f"[graphiti] evict_client close failed for {graph_name}: {exc}")


async def get_client_for_graph(graph_name: str) -> Graphiti:
    """Get or create a Graphiti client for an exact graph name.

    Tenant graphs go through get_client (slug to name); this is for the
    scratch graphs a rehearsal writes to, so the tenant graph is never the
    target by accident.
    """
    if graph_name not in _clients:
        logger.info(f"[graphiti] Initializing graph: {graph_name}")
        driver = _create_driver(graph_name)
        llm_client = _create_llm_client()
        client = Graphiti(graph_driver=driver, llm_client=llm_client, embedder=_create_embedder())
        _clients[graph_name] = client
    return _clients[graph_name]


async def evict_graph(graph_name: str) -> None:
    """Drop a cached client by graph name; a broken pool must not be reused."""
    client = _clients.pop(graph_name, None)
    if client is None:
        return
    try:
        await client.close()
    except Exception as exc:  # noqa: BLE001 - closing a broken pool may itself fail
        logger.debug(f"[graphiti] evict_graph close failed for {graph_name}: {exc}")


async def get_client(client_slug: str) -> Graphiti:
    """Get or create a Graphiti client for a specific client graph.

    Each client gets a separate FalkorDB named graph via the driver's
    `database` parameter, providing full data isolation.
    """
    graph_name = _graph_name_for_client(client_slug)

    if graph_name not in _clients:
        logger.info(f"[graphiti] Initializing graph: {graph_name}")
        driver = _create_driver(graph_name)
        llm_client = _create_llm_client()
        client = Graphiti(graph_driver=driver, llm_client=llm_client, embedder=_create_embedder())
        _clients[graph_name] = client

    return _clients[graph_name]


async def get_segment_client(industry: str) -> Graphiti:
    """Get or create a Graphiti client for a segment graph."""
    graph_name = _segment_graph_name(industry)

    if graph_name not in _clients:
        logger.info(f"[graphiti] Initializing segment graph: {graph_name}")
        driver = _create_driver(graph_name)
        llm_client = _create_llm_client()
        client = Graphiti(graph_driver=driver, llm_client=llm_client, embedder=_create_embedder())
        _clients[graph_name] = client

    return _clients[graph_name]


async def init_graph(client_slug: str) -> str:
    """Initialize a new graph for a client (create indices and constraints)."""
    client = await get_client(client_slug)
    await client.build_indices_and_constraints()
    graph_name = _graph_name_for_client(client_slug)

    # Create the native vector index now, while the graph is empty, so new clients
    # get incremental HNSW indexing from the first edge onward — avoiding the
    # one-time bulk build that older graphs paid on their first search. Best-effort.
    try:
        # OFF the event loop. This is a synchronous driver call, and running it
        # inline blocks every other request for its full duration — up to the
        # 120s socket timeout, repeatedly, if FalkorDB is not answering.
        #
        # Demonstrated 2026-08-29: calling the new maintenance route against an
        # unresponsive FalkorDB froze this service so completely that /health
        # stopped answering. #25 moved /graph and /structured off the loop and
        # deliberately left the admin callers, reasoning they were manual and
        # rare. "Rare" is not "never", and the one time one was called it took
        # the whole service down.
        # BOTH indexes, off the loop. main added the Entity index here after this
        # branch was written, and it is the more expensive of the two — the
        # slowlog measured a comparable CREATE INDEX on this graph at 102s.
        # Building it inline would freeze the service exactly as the edge index
        # did, so the resolution runs the pair in the worker thread rather than
        # keeping one off the loop and putting a heavier one back on it.
        def _init_vector_index():
            db = get_falkor_db()
            _ensure_edge_vector_index(db.select_graph(graph_name), graph_name)
            ensure_node_vector_index(
                db.select_graph(graph_name), graph_name, int(settings.embedding_dim)
            )

        await asyncio.to_thread(_init_vector_index)
    except Exception as e:
        logger.warning(f"[graphiti] vector index init skipped for {graph_name}: {e}")

    logger.info(f"[graphiti] Graph initialized: {graph_name}")
    return graph_name


async def reset_graph(client_slug: str) -> dict[str, Any]:
    """DESTRUCTIVE: wipe all nodes + relationships from a client's graph,
    then re-initialize indices + constraints. Used for clean-slate backfills.

    Returns { graph_name, nodes_deleted }.
    """
    graph_name = _graph_name_for_client(client_slug)
    logger.warning(f"[graphiti] RESETTING graph: {graph_name}")

    # Drop the cached client so the next operation reconnects fresh after wipe.
    if graph_name in _clients:
        try:
            await _clients[graph_name].close()
        except Exception:
            pass
        del _clients[graph_name]

    # Use FalkorDB's underlying Redis connection to issue GRAPH.DELETE, which
    # removes all data for the named graph in a single atomic op. Reaching
    # through the graphiti FalkorDriver gives us the password + connection.
    driver = _create_driver(graph_name)
    try:
        # falkordb-py exposes driver.client (Redis connection) and a Graph API.
        # The simplest portable approach is to issue a raw command.
        redis_client = driver.client if hasattr(driver, "client") else driver._client
        try:
            redis_client.execute_command("GRAPH.DELETE", graph_name)
            logger.info(f"[graphiti] GRAPH.DELETE {graph_name} succeeded")
        except Exception as del_err:
            # If the graph doesn't exist yet, GRAPH.DELETE errors. That's fine
            # for our reset-or-init semantics — just log and continue.
            logger.info(f"[graphiti] GRAPH.DELETE {graph_name}: {del_err} (expected if graph was empty)")
    finally:
        try:
            await driver.close()
        except Exception:
            pass

    # Re-init: creates indices + constraints on the now-empty graph.
    fresh_client = await get_client(client_slug)
    await fresh_client.build_indices_and_constraints()

    logger.warning(f"[graphiti] Graph reset complete: {graph_name}")
    return {"graph_name": graph_name, "status": "reset_and_reinitialized"}


async def add_episode(
    client_slug: str,
    engagement_id: str,
    name: str,
    content: str,
    source_description: str,
    reference_time: datetime,
    metadata: dict[str, Any] | None = None,
    *,
    source_id: str | None = None,
    source_type: str | None = None,
    episode_type: str | None = None,
    anchor_mode: str | None = None,
    producer_contract_version: str | None = None,
    graph_name_override: str | None = None,
) -> dict[str, Any]:
    """Add an episode to the client's knowledge graph.

    `graph_name_override` (a scratch graph, admin rehearsals only) writes the
    same episode to that graph instead of the tenant's; group_id follows it.

    Uses EpisodeType.text for all ingestion (plain text content).
    group_id is set to the graph name to prevent driver re-cloning.
    """
    del metadata  # Graphiti-core has no structured metadata projection here.
    anchor_values = (
        source_id,
        source_type,
        episode_type,
        anchor_mode,
        producer_contract_version,
    )
    if any(value is not None for value in anchor_values):
        if not _nonempty_string(engagement_id, 240) or not all(
            _nonempty_string(value, 240) for value in anchor_values
        ):
            raise ValueError("episode provenance anchors must be complete")
    if anchor_mode is not None and anchor_mode not in STRUCTURALLY_ANCHORED_MODES:
        raise ValueError("episode anchor_mode is unsupported")

    graph_name = graph_name_override or _graph_name_for_client(client_slug)
    client = await get_client_for_graph(graph_name) if graph_name_override else await get_client(client_slug)

    start = time.time()

    try:
        result = await client.add_episode(
            name=name,
            episode_body=content,
            source_description=source_description,
            reference_time=reference_time,
            source=GraphitiEpisodeType.text,
            group_id=graph_name,
            # The taxonomy config.yaml always claimed to define. Without this
            # argument all 16 types were decoration.
            entity_types=entity_type_models() or None,
        )
    except BaseException:
        # BaseException, not Exception: asyncio.CancelledError has been a
        # BaseException since 3.8, and a cancelled read is the exact case that
        # leaves the connection desynced. Catching only Exception would let the
        # most important one through uncleaned.
        await _log_slow_queries(graph_name)
        await evict_graph(graph_name)
        raise

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"[graphiti] Episode added to {graph_name} "
        f"in {elapsed_ms:.0f}ms: {name}"
    )

    # Report the slow log on SUCCESS too, not only on failure.
    #
    # Reporting only on failure left the most important question unanswerable.
    # The bounded fulltext query (#35/#36) went live in the same minutes that a
    # wedged FalkorDB was restarted, and the two episodes that then succeeded
    # took 758s and 1555s -- long enough to still contain the 588s unbounded
    # query. With no slow log on the success path there was no measurement to
    # separate "the bound works" from "the restart cleared a stuck server", and
    # the fix stayed unproven while looking proven.
    #
    # An ingest that SUCCEEDS is exactly when the timings are worth having.
    await _log_slow_queries(graph_name)

    # AddEpisodeResults has .episode, .nodes, .edges attributes
    entities_extracted = len(result.nodes) if hasattr(result, "nodes") else 0
    facts_created = len(result.edges) if hasattr(result, "edges") else 0
    episode_id = ""
    if hasattr(result, "episode") and hasattr(result.episode, "uuid"):
        episode_id = str(result.episode.uuid)

    if anchor_mode is not None:
        canonical_episode_id = _uuid_string(episode_id)
        if not canonical_episode_id:
            raise RuntimeError("anchored episode ingestion returned no valid episode UUID")

        # Run OFF the event loop. This is a synchronous driver call on the
        # per-episode ingest path; inline it blocks every other request for its
        # duration, which is how a single extraction was able to stop this
        # service answering anything at all. The socket timeout bounds how long
        # it can take; this keeps that time from being charged to the loop.
        def _anchor_provenance():
            graph = get_falkor_db().select_graph(graph_name)
            return graph.query(
                """
                MATCH (ep:Episodic {uuid: $episode_uuid, group_id: $group_id})
                SET ep.source_id = $source_id,
                    ep.source_type = $source_type,
                    ep.engagement_id = $engagement_id,
                    ep.episode_type = $episode_type,
                    ep.anchor_mode = $anchor_mode,
                    ep.producer_contract_version = $producer_contract_version
                RETURN ep.uuid
                """,
                params={
                    "episode_uuid": canonical_episode_id,
                    "group_id": graph_name,
                    "source_id": source_id,
                    "source_type": source_type,
                    "engagement_id": engagement_id,
                    "episode_type": episode_type,
                    "anchor_mode": anchor_mode,
                    "producer_contract_version": producer_contract_version,
                },
            )

        updated = await asyncio.to_thread(_anchor_provenance)
        if not updated.result_set:
            raise RuntimeError("anchored episode provenance update matched no episode")

    return {
        "episode_id": episode_id,
        "entities_extracted": entities_extracted,
        "facts_created": facts_created,
        "elapsed_ms": elapsed_ms,
    }


def _ensure_edge_vector_index(graph, graph_name: str) -> None:
    """Idempotently ensure a FalkorDB HNSW vector index on RELATES_TO.fact_embedding.

    graphiti-core only builds range + fulltext indexes for FalkorDB, so its cosine
    search is an O(N) full scan (~28s on a 7k-edge graph). A native vector index
    makes it O(log N). Creating when one exists raises (caught); an unsupported
    FalkorDB version also raises here and again on query, so the caller falls back
    to graphiti's scan — no regression either way. Cached per process.
    """
    if graph_name in _edge_vindex_ensured:
        return
    dim = int(settings.embedding_dim)
    try:
        graph.query(
            f"CREATE VECTOR INDEX FOR ()-[r:RELATES_TO]->() ON (r.fact_embedding) "
            f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
        )
        logger.info(f"[graphiti] created RELATES_TO.fact_embedding vector index on {graph_name} (dim={dim})")
    except Exception as e:
        logger.debug(f"[graphiti] vector index ensure on {graph_name}: {e}")
    _edge_vindex_ensured.add(graph_name)


# RRF rank constant. graphiti uses 1; 60 is the canonical RRF default and slightly
# smoother. The exact value barely changes the top-K. Kept explicit for clarity.
_RRF_K = 60

# Put the yielded relationship's endpoints in scope WITHOUT a join. The
# former `MATCH (a:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(b:Entity)` was
# planned by FalkorDB as a full Entity label scan per yielded row with an
# edge-index probe for each (profiled 2026-09-14 on client_pokagon: the
# procedure answered 20 rows in 0.5 ms, the join took 1,649 ms and touched
# 116,380 nodes; about 80 ms per candidate, linear in the pool). startNode
# and endNode read the endpoints off the relationship itself.
_EDGE_ENDPOINTS = (
    "WITH e, score, startNode(e) AS a, endNode(e) AS b "
    "WHERE e.group_id = $group_id "
)
_EDGE_RETURN = (
    "RETURN e.uuid AS uuid, e.fact AS fact, e.name AS name, "
    "a.uuid AS src, a.name AS src_name, b.uuid AS tgt, b.name AS tgt_name, "
    "e.episodes AS episodes, e.valid_at AS va, e.invalid_at AS ia, "
    "e.expired_at AS ea"
)
_EDGE_ENDPOINTS_RETURN = _EDGE_ENDPOINTS + _EDGE_RETURN


def fulltext_query_for(query: str, graph_name: str) -> str:
    """The RediSearch query for the BM25 leg, in graphiti's own form.

    graphiti builds `(@group_id:"<graph>") (term | term | ...)`: an OR of the
    question's terms under the group filter, stopwords dropped, specials
    escaped. The leg used to send the bare sanitised phrase, which RediSearch
    reads as an AND of every term; no fact contains all nine words of a real
    question, so the leg matched nothing (profiled 2026-09-14: the procedure
    produced 0 records for every question) and the fast path was vector-only.
    """
    from graphiti_core.driver.falkordb.operations.search_ops import (  # noqa: PLC0415
        _build_falkor_fulltext_query,
    )

    return _build_falkor_fulltext_query(query, [graph_name])


def fast_search_leg_queries(pool: int) -> tuple[str, str]:
    """The exact Cypher of the fast path's two legs for a candidate pool.

    One builder, so the search and the profiler cannot drift: what the
    profiler measures is what the search runs.
    """
    pool = int(pool)
    vector = (
        f"CALL db.idx.vector.queryRelationships('RELATES_TO', 'fact_embedding', {pool}, vecf32($q)) "
        f"YIELD relationship AS e, score {_EDGE_ENDPOINTS_RETURN}"
    )
    bm25 = (
        f"CALL db.idx.fulltext.queryRelationships('RELATES_TO', $query) "
        f"YIELD relationship AS e, score "
        f"WITH e, score ORDER BY score DESC LIMIT {pool * FULLTEXT_OVERFETCH} "
        f"{_EDGE_ENDPOINTS_RETURN} "
        f"ORDER BY score DESC LIMIT {pool}"
    )
    return vector, bm25


def _lucene_sanitize(q: str) -> str:
    """Strip RediSearch/fulltext special chars so a natural-language query never
    breaks the BM25 parser (e.g. '&', '-', ':'). Mirrors graphiti's intent."""
    out = []
    for ch in q:
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def _row_to_edge(row) -> Any:
    return SimpleNamespace(
        uuid=row[0],
        fact=row[1] or "",
        name=row[2] or "",
        source_node_uuid=row[3] or "",
        source_node_name=row[4] or "",
        target_node_uuid=row[5] or "",
        target_node_name=row[6] or "",
        episodes=row[7],
        valid_at=_parse_dt(row[8]),
        invalid_at=_parse_dt(row[9]),
        expired_at=_parse_dt(row[10]),
    )


async def _search_fast(client_slug: str, query: str, max_results: int) -> list[Any]:
    """Hybrid edge search using FalkorDB native indexes — the same shape as
    graphiti's EDGE_HYBRID_SEARCH_RRF (BM25 + cosine + RRF) but fast:

      - cosine via the HNSW vector index (O(log N), not the O(N) scan)
      - BM25 via the existing fulltext index
      - reciprocal-rank-fusion of the two ranked lists

    Embeds the query with the SAME configured embedder used for stored vectors
    (no query/document asymmetry). Returns lightweight edge-like objects exposing
    the attributes search.py reads. Raises on vector-path failure so search() can
    fall back to graphiti's own hybrid search; a BM25 failure degrades to
    vector-only within this function (still fast and relevant).
    """
    graph_name = _graph_name_for_client(client_slug)

    db = get_falkor_db()
    graph = (
        _select_existing_probe_graph(db, graph_name)
        if settings.graphiti_acceptance_probe_mode
        else db.select_graph(graph_name)
    )
    if not settings.graphiti_acceptance_probe_mode:
        await asyncio.to_thread(_ensure_edge_vector_index, graph, graph_name)

    embedder = _create_embedder()
    if embedder is None:
        raise RuntimeError("fast search requires an explicit (Voyage) embedder")
    # This is a query-embedding call only. Probe mode never constructs Graphiti
    # or an extraction/generative client.
    qvec = await embedder.create(input_data=[query.replace("\n", " ")])

    # Pull a candidate pool 2x the requested size from each method, then fuse.
    pool = max(int(max_results) * 2, int(max_results))
    vector_query, bm25_query = fast_search_leg_queries(pool)

    by_uuid: dict[str, Any] = {}

    def _collect(rows) -> list[str]:
        order: list[str] = []
        for row in rows:
            uuid = row[0]
            if uuid and uuid not in by_uuid:
                by_uuid[uuid] = _row_to_edge(row)
            if uuid:
                order.append(uuid)
        return order

    # The two legs run concurrently, each under its own FalkorDB TIMEOUT, so
    # a call costs at most the larger budget rather than their sum: the backend
    # gives a search 8 s and the preview path makes two calls per request.
    # A leg past its budget answers empty for this request rather than late.
    async def vector_leg() -> Any:
        # Cosine via HNSW (k inlined int; vector passed as the proven vecf32($param)).
        budget = settings.search_vector_timeout_ms
        began = time.monotonic()
        try:
            return await _graph_read_async(
                graph,
                vector_query,
                {"q": qvec, "group_id": graph_name},
                timeout_ms=budget,
            )
        except Exception as error:
            if _is_query_timeout(error, (time.monotonic() - began) * 1000, budget):
                logger.warning(
                    "[graphiti] fast vector leg exceeded %s ms on %s; answering empty",
                    budget,
                    graph_name,
                )
                raise SearchBudgetExceeded("fast vector leg exceeded its budget") from error
            raise

    async def bm25_leg() -> Any | None:
        # BM25 via the fulltext index, bounded BEFORE the join: the procedure
        # yields every matching relationship, and joining each back to its
        # endpoints before the limit cost one question 68 s once the graph held
        # thousands of approval facts (2026-09-14; the hybrid path had the same
        # defect and the same fix, in indexed_falkor). Over-fetched because the
        # group filter applies after the bound; the carrying WITH before RETURN
        # mirrors the proven query. Resilient: a parser hiccup or an exceeded
        # budget degrades to vector-only rather than failing the whole search.
        safe_q = fulltext_query_for(query, graph_name)
        if not safe_q:
            return None
        budget = settings.search_bm25_timeout_ms
        began = time.monotonic()
        try:
            return await _graph_read_async(
                graph,
                bm25_query,
                {"query": safe_q, "group_id": graph_name},
                timeout_ms=budget,
            )
        except Exception as error:
            if _is_query_timeout(error, (time.monotonic() - began) * 1000, budget):
                logger.warning(
                    "[graphiti] fast BM25 leg exceeded %s ms on %s; vector-only for this request",
                    budget,
                    graph_name,
                )
            else:
                logger.warning(
                    "[graphiti] fast BM25 leg skipped on %s error_type=%s; vector-only for this request",
                    graph_name,
                    type(error).__name__,
                )
            return None

    vres, bres = await asyncio.gather(vector_leg(), bm25_leg())
    vorder = _collect(vres.result_set)
    border: list[str] = _collect(bres.result_set) if bres is not None else []

    # Reciprocal rank fusion of the two ranked lists.
    scores: dict[str, float] = {}
    for ordered in (vorder, border):
        for rank, uuid in enumerate(ordered):
            scores[uuid] = scores.get(uuid, 0.0) + 1.0 / (rank + _RRF_K)

    ranked = sorted(scores, key=lambda u: scores[u], reverse=True)[: int(max_results)]
    return [by_uuid[u] for u in ranked]


PROFILE_MAX_RESULTS = 30
PROFILE_LEG_TIMEOUT_SECONDS = 30


def _profile_read(graph: Any, query: str, params: dict[str, Any]) -> tuple[float, list[str]]:
    """GRAPH.PROFILE one leg: wall time in ms and the plan's raw lines.

    PROFILE runs the query and returns the plan with per-operation records
    and timings. The driver's ExecutionPlan keeps the server's lines on
    `.plan`; its `str()` re-renders the parsed tree WITHOUT the
    "Records produced / Execution time" segments (it pops them into
    per-operation stats), which is the one thing this route exists to
    return. The result set is not returned, so nothing but operator names
    and numbers leaves the database.
    """
    began = time.monotonic()
    try:
        plan = graph.profile(query, params)
        raw = getattr(plan, "plan", plan)
    except ResponseError:
        raise  # the server refused the query: nothing to read
    except Exception:  # noqa: BLE001 - the driver parses the plan in __init__
        # The server answered but the driver's parser choked on a line (an
        # integer-formatted time, or four spaces inside an argument). Take
        # the raw reply; the lines are what this route returns anyway.
        raw = graph.execute_command(
            "GRAPH.PROFILE", getattr(graph, "name", None) or graph._name, graph._build_params_header(params) + query
        )
    ms = (time.monotonic() - began) * 1000
    lines = []
    for line in raw if isinstance(raw, (list, tuple)) else str(raw).splitlines():
        text = line.decode() if isinstance(line, bytes) else str(line)
        if text.strip():
            lines.append(text.rstrip())
    return ms, lines


async def profile_fast_search(
    client_slug: str,
    query: str,
    max_results: int,
    scratch_graph: str | None = None,
) -> dict[str, Any]:
    """Profile the fast path's two legs for a question, on the live graph.

    Added 2026-09-14: the fast path's latency grew linearly with the candidate
    pool on an idle graph (about 70 ms per vector-index candidate at 25.9k
    edges), which nothing could explain from outside the database. The slow
    log ranks queries but has no operators; this returns the plan FalkorDB
    executed, per leg, with its per-operation timings, for the exact Cypher
    the search runs. Read-only, admin-scoped, and returns no fact text.
    """
    if not 1 <= int(max_results) <= PROFILE_MAX_RESULTS:
        raise ValueError(f"max_results must be 1..{PROFILE_MAX_RESULTS}")
    graph_name = scratch_graph or _graph_name_for_client(client_slug)
    graph = get_falkor_db().select_graph(graph_name)
    embedder = _create_embedder()
    if embedder is None:
        raise RuntimeError("fast search requires an explicit (Voyage) embedder")
    pool = max(int(max_results) * 2, int(max_results))
    vector_query, bm25_query = fast_search_leg_queries(pool)
    began = time.monotonic()
    qvec = await embedder.create(input_data=[query.replace("\n", " ")])
    embed_ms = (time.monotonic() - began) * 1000

    # GRAPH.PROFILE takes no per-query TIMEOUT in the driver, so each leg is
    # bounded here. A leg past PROFILE_LEG_TIMEOUT_SECONDS is reported as
    # exceeded and the other leg is still profiled: "this leg exceeds the
    # bound" is an answer, not a failure. The worker thread and its pooled
    # connection stay busy until the database answers or the socket bound
    # (900 s) fires; one of each per exceeded leg.
    async def bounded(leg_query: str, params: dict[str, Any]) -> tuple[float, list[str]]:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_profile_read, graph, leg_query, params), PROFILE_LEG_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            return PROFILE_LEG_TIMEOUT_SECONDS * 1000.0, [f"(exceeded the {PROFILE_LEG_TIMEOUT_SECONDS} s profile bound)"]

    vector_ms, vector_plan = await bounded(vector_query, {"q": qvec, "group_id": graph_name})
    safe_q = fulltext_query_for(query, graph_name)
    if safe_q:
        bm25_ms, bm25_plan = await bounded(bm25_query, {"query": safe_q, "group_id": graph_name})
    else:
        bm25_ms, bm25_plan = 0.0, ["(no searchable terms after sanitising)"]
    return {
        "graph_name": graph_name,
        "pool": pool,
        "embed_ms": round(embed_ms),
        "vector": {"ms": round(vector_ms), "query": vector_query, "plan": vector_plan},
        "bm25": {"ms": round(bm25_ms), "query": bm25_query, "plan": bm25_plan},
    }


async def search_with_path(
    client_slug: str,
    query: str,
    max_results: int = 10,
) -> tuple[Any, RetrievalPath]:
    """Search the client's knowledge graph for the most relevant facts (edges).

    Tries the vector-index fast path first; on ANY error OR an empty result (e.g.
    the index is still building, or this FalkorDB lacks vector indexes) falls back
    to graphiti's hybrid (BM25 + full-scan cosine + RRF) search. The fallback is
    why this is zero-regression: worst case is the prior behavior.

    Returns edge-like objects exposing: fact, name, source_node_uuid,
    target_node_uuid, valid_at / invalid_at / expired_at.
    """
    graph_name = _graph_name_for_client(client_slug)
    start = time.time()

    probe_mode = settings.graphiti_acceptance_probe_mode
    try:
        edges = await _search_fast(client_slug, query, max_results)
        if edges:
            logger.info(
                f"[graphiti] Search(fast) in {graph_name}: {len(edges)} edges "
                f"({(time.time() - start) * 1000:.0f}ms)"
            )
            return edges, "fast"
        if probe_mode:
            raise AcceptanceProbeReadError(
                "acceptance probe fast search returned no results"
            )
        logger.info(f"[graphiti] fast search returned 0 on {graph_name}; falling back to hybrid")
    except Exception as error:
        if probe_mode:
            logger.warning(
                "[graphiti] acceptance probe fast search failed on %s "
                "error_type=%s",
                graph_name,
                type(error).__name__,
            )
            if isinstance(error, AcceptanceProbeReadError):
                raise
            raise AcceptanceProbeReadError(
                "acceptance probe fast search is unavailable"
            ) from error
        if isinstance(error, SearchBudgetExceeded):
            # The hybrid fallback is slower still (an O(N) cosine scan); a
            # request past its budget answers empty, inside the budget.
            logger.warning(
                "[graphiti] Search(fast) in %s exceeded its budget after %.0fms; answering empty",
                graph_name,
                (time.time() - start) * 1000,
            )
            return [], "fast"
        logger.warning(
            "[graphiti] fast search failed on %s error_type=%s; falling back "
            "to hybrid",
            graph_name,
            type(error).__name__,
        )

    client = await get_client(client_slug)
    edges = await client.search(
        query=query,
        num_results=max_results,
        group_ids=[graph_name],
    )
    edge_count = len(edges) if isinstance(edges, list) else "unsupported"
    logger.info(
        "[graphiti] Search(hybrid-fallback) in %s: %s edges (%sms)",
        graph_name,
        edge_count,
        round((time.time() - start) * 1000),
    )
    return edges, "hybrid_fallback"


async def search(
    client_slug: str,
    query: str,
    max_results: int = 10,
) -> list[Any]:
    """Backwards-compatible list-only search API."""

    edges, _retrieval_path = await search_with_path(
        client_slug=client_slug,
        query=query,
        max_results=max_results,
    )
    return edges if isinstance(edges, list) else []


async def resolve_search_provenance(
    client_slug: str,
    edges: list[Any],
) -> tuple[dict[str, ResolvedSearchEdge], int]:
    """Resolve endpoint names and episode anchors for either retrieval path.

    The lookup is batched and read-only. Returned source fields are graph claims,
    not proof that a relational source exists. Edges with stable identities but
    malformed graph structure remain represented with ``malformed=True`` so the
    router can account for and suppress them exactly once.
    """

    ordered_edge_ids: list[str] = []
    seen_edge_ids: set[str] = set()
    malformed_response_events = 0
    for edge in edges:
        edge_id = _uuid_string(getattr(edge, "uuid", None))
        if not edge_id:
            malformed_response_events += 1
            continue
        if edge_id not in seen_edge_ids:
            ordered_edge_ids.append(edge_id)
            seen_edge_ids.add(edge_id)
    if not ordered_edge_ids:
        return {}, malformed_response_events


    graph_name = _graph_name_for_client(client_slug)
    db = get_falkor_db()
    graph = (
        _select_existing_probe_graph(db, graph_name)
        if settings.graphiti_acceptance_probe_mode
        else db.select_graph(graph_name)
    )
    edge_result = await _graph_read_async(
        graph,
        """
        MATCH (subject:Entity)-[edge:RELATES_TO]->(object:Entity)
        WHERE edge.uuid IN $edge_uuids AND edge.group_id = $group_id
        RETURN edge.uuid, subject.uuid, subject.name, edge.name,
               object.uuid, object.name, edge.fact, edge.episodes,
               edge.valid_at, edge.invalid_at, edge.expired_at
        """,
        params={"edge_uuids": ordered_edge_ids, "group_id": graph_name},
    )
    edge_rows = edge_result.result_set

    preliminary: dict[str, dict[str, Any]] = {}
    corrupted_edge_ids: set[str] = set()
    episode_ids: list[str] = []
    seen_episode_ids: set[str] = set()
    for row in edge_rows:
        fact_id = (
            _uuid_string(row[0])
            if isinstance(row, (list, tuple)) and row
            else None
        )
        if not isinstance(row, (list, tuple)) or len(row) < 11:
            if fact_id in seen_edge_ids:
                corrupted_edge_ids.add(fact_id)
            else:
                malformed_response_events += 1
            continue
        if not fact_id:
            malformed_response_events += 1
            continue
        if fact_id not in seen_edge_ids:
            malformed_response_events += 1
            continue
        if fact_id in preliminary:
            corrupted_edge_ids.add(fact_id)
            continue
        episodes, episodes_valid = _episode_uuid_list(row[7])
        for episode_id in episodes:
            if episode_id not in seen_episode_ids:
                episode_ids.append(episode_id)
                seen_episode_ids.add(episode_id)
        valid_at = _parse_dt(row[8])
        invalid_at = _parse_dt(row[9])
        expired_at = _parse_dt(row[10])
        malformed_temporal = any(
            raw_value not in (None, "") and parsed_value is None
            for raw_value, parsed_value in (
                (row[8], valid_at),
                (row[9], invalid_at),
                (row[10], expired_at),
            )
        )
        preliminary[fact_id] = {
            "fact_id": fact_id,
            "subject_uuid": _uuid_string(row[1]) or "",
            "subject_name": _nonempty_string(row[2], 2_000) or "",
            "predicate": _nonempty_string(row[3], 160) or "",
            "object_uuid": _uuid_string(row[4]) or "",
            "object_name": _nonempty_string(row[5], 2_000) or "",
            "fact": _nonempty_string(row[6], 16_000) or "",
            "episode_uuids": episodes,
            "valid_at": valid_at,
            "invalid_at": invalid_at,
            "expired_at": expired_at,
            "malformed": not episodes_valid or malformed_temporal,
        }

    source_by_episode: dict[str, ResolvedEpisodeAnchor] = {}
    corrupted_episode_ids: set[str] = set()
    if episode_ids:
        source_result = await _graph_read_async(
            graph,
            """
            MATCH (episode:Episodic)
            WHERE episode.uuid IN $episode_uuids AND episode.group_id = $group_id
            RETURN episode.uuid, episode.name, episode.source_description,
                   episode.source_type, episode.source_id,
                   episode.engagement_id, episode.episode_type,
                   episode.anchor_mode, episode.producer_contract_version,
                   episode.valid_at, episode.provenance_write_state
            """,
            params={"episode_uuids": episode_ids, "group_id": graph_name},
        )
        source_rows = source_result.result_set
        for row in source_rows:
            episode_id = (
                _uuid_string(row[0])
                if isinstance(row, (list, tuple)) and row
                else None
            )
            if not isinstance(row, (list, tuple)) or len(row) < 11:
                if episode_id in seen_episode_ids:
                    corrupted_episode_ids.add(episode_id)
                else:
                    malformed_response_events += 1
                continue
            if not episode_id or episode_id not in seen_episode_ids:
                malformed_response_events += 1
                continue
            if episode_id in source_by_episode:
                corrupted_episode_ids.add(episode_id)
                continue
            valid_at = _parse_dt(row[9])
            if row[9] not in (None, "") and valid_at is None:
                corrupted_episode_ids.add(episode_id)
            source_by_episode[episode_id] = ResolvedEpisodeAnchor(
                episode_uuid=episode_id,
                episode_name=_nonempty_string(row[1], 2_000) or "",
                source_description=_nonempty_string(row[2], 2_000) or "",
                source_type=_nonempty_string(row[3], 64),
                source_id=_nonempty_string(row[4], 240),
                engagement_id=_nonempty_string(row[5], 240),
                episode_type=_nonempty_string(row[6], 64),
                anchor_mode=_nonempty_string(row[7], 64),
                producer_contract_version=_nonempty_string(row[8], 64),
                valid_at=valid_at,
                provenance_write_state=_nonempty_string(row[10], 32),
            )

        for episode_id in corrupted_episode_ids:
            source = source_by_episode.get(episode_id)
            if source is not None:
                source_by_episode[episode_id] = replace(source, malformed=True)
            else:
                source_by_episode[episode_id] = ResolvedEpisodeAnchor(
                    episode_uuid=episode_id,
                    episode_name="",
                    source_description="",
                    source_type=None,
                    source_id=None,
                    engagement_id=None,
                    episode_type=None,
                    anchor_mode=None,
                    producer_contract_version=None,
                    provenance_write_state=None,
                    valid_at=None,
                    malformed=True,
                )

    resolved: dict[str, ResolvedSearchEdge] = {}
    for edge_id in ordered_edge_ids:
        row = preliminary.get(edge_id)
        if row is None:
            resolved[edge_id] = ResolvedSearchEdge(
                fact_id=edge_id,
                subject_uuid="",
                subject_name="",
                predicate="",
                object_uuid="",
                object_name="",
                fact="",
                episode_uuids=(),
                sources=(),
                valid_at=None,
                invalid_at=None,
                expired_at=None,
                malformed=True,
            )
            continue
        if edge_id in corrupted_edge_ids:
            row["malformed"] = True
        sources = tuple(
            source_by_episode[episode_id]
            for episode_id in row["episode_uuids"]
            if episode_id in source_by_episode
        )
        resolved[edge_id] = ResolvedSearchEdge(
            **row,
            sources=sources,
        )
    # Response observations have no stable-ID denominator and are counted
    # separately, but never beyond the bounded producer pool inspected here.
    return resolved, min(malformed_response_events, len(edges))


async def search_segment(
    industry: str,
    query: str,
    max_results: int = 5,
) -> list[Any]:
    """Search the segment knowledge graph. Returns list of EntityEdge."""
    try:
        client = await get_segment_client(industry)
        graph_name = _segment_graph_name(industry)
        results = await client.search(
            query=query,
            num_results=max_results,
            group_ids=[graph_name],
        )
        return results
    except Exception as e:
        logger.warning(f"[graphiti] Segment search failed for {industry}: {e}")
        return []


async def close_all():
    """Close all cached Graphiti clients."""
    for name, client in _clients.items():
        try:
            await client.close()
            logger.info(f"[graphiti] Closed graph client: {name}")
        except Exception as e:
            logger.warning(f"[graphiti] Error closing {name}: {e}")
    _clients.clear()
