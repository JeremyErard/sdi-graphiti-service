"""Graphiti core wrapper — graph connections, ingestion, and provenance search."""

import ast
import logging
import time
import uuid as uuidlib
from dataclasses import dataclass, replace
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Literal

import yaml
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType as GraphitiEpisodeType
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig

from app.config import settings
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


def _graph_read(graph: Any, query: str, params: dict[str, Any] | None = None) -> Any:
    """Use FalkorDB's read-only command in acceptance-probe processes."""

    if settings.graphiti_acceptance_probe_mode:
        return graph.ro_query(query, params)
    return graph.query(query, params)


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
        if episode_uuid not in seen:
            normalized.append(episode_uuid)
            seen.add(episode_uuid)
            if len(normalized) > _MAX_EPISODES_PER_FACT:
                return (), False
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


def _create_driver(graph_name: str) -> FalkorDriver:
    """Create a FalkorDB driver targeting a specific named graph."""
    return FalkorDriver(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
        database=graph_name,
    )


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
        from falkordb import FalkorDB

        db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
        )
        _ensure_edge_vector_index(db.select_graph(graph_name), graph_name)
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
) -> dict[str, Any]:
    """Add an episode to the client's knowledge graph.

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

    client = await get_client(client_slug)
    graph_name = _graph_name_for_client(client_slug)

    start = time.time()

    result = await client.add_episode(
        name=name,
        episode_body=content,
        source_description=source_description,
        reference_time=reference_time,
        source=GraphitiEpisodeType.text,
        group_id=graph_name,
    )

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"[graphiti] Episode added to {graph_name} "
        f"in {elapsed_ms:.0f}ms: {name}"
    )

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
        from falkordb import FalkorDB

        db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
        )
        graph = db.select_graph(graph_name)
        updated = graph.query(
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

# Join the index-procedure relationship back to its endpoints. The relationship
# procedures do not otherwise put endpoint nodes in scope.
_EDGE_MATCH_RETURN = (
    "MATCH (a:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(b:Entity) "
    "WHERE e.group_id = $group_id "
    "RETURN e.uuid AS uuid, e.fact AS fact, e.name AS name, "
    "a.uuid AS src, a.name AS src_name, b.uuid AS tgt, b.name AS tgt_name, "
    "e.episodes AS episodes, e.valid_at AS va, e.invalid_at AS ia, "
    "e.expired_at AS ea"
)


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
    from falkordb import FalkorDB

    db = FalkorDB(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
    )
    graph = (
        _select_existing_probe_graph(db, graph_name)
        if settings.graphiti_acceptance_probe_mode
        else db.select_graph(graph_name)
    )
    if not settings.graphiti_acceptance_probe_mode:
        _ensure_edge_vector_index(graph, graph_name)

    embedder = _create_embedder()
    if embedder is None:
        raise RuntimeError("fast search requires an explicit (Voyage) embedder")
    # This is a query-embedding call only. Probe mode never constructs Graphiti
    # or an extraction/generative client.
    qvec = await embedder.create(input_data=[query.replace("\n", " ")])

    # Pull a candidate pool 2x the requested size from each method, then fuse.
    pool = max(int(max_results) * 2, int(max_results))

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

    # Cosine via HNSW (k inlined int; vector passed as the proven vecf32($param)).
    vres = _graph_read(
        graph,
        f"CALL db.idx.vector.queryRelationships('RELATES_TO', 'fact_embedding', {pool}, vecf32($q)) "
        f"YIELD relationship AS rel, score {_EDGE_MATCH_RETURN}",
        {"q": qvec, "group_id": graph_name},
    )
    vorder = _collect(vres.result_set)

    # BM25 via the fulltext index. Resilient: a parser hiccup degrades to
    # vector-only rather than failing the whole search.
    border: list[str] = []
    safe_q = _lucene_sanitize(query)
    if safe_q:
        try:
            bres = _graph_read(
                graph,
                f"CALL db.idx.fulltext.queryRelationships('RELATES_TO', $query) "
                f"YIELD relationship AS rel, score {_EDGE_MATCH_RETURN} LIMIT {pool}",
                {"query": safe_q, "group_id": graph_name},
            )
            border = _collect(bres.result_set)
        except Exception as error:
            logger.debug(
                "[graphiti] fast BM25 leg skipped on %s error_type=%s",
                graph_name,
                type(error).__name__,
            )

    # Reciprocal rank fusion of the two ranked lists.
    scores: dict[str, float] = {}
    for ordered in (vorder, border):
        for rank, uuid in enumerate(ordered):
            scores[uuid] = scores.get(uuid, 0.0) + 1.0 / (rank + _RRF_K)

    ranked = sorted(scores, key=lambda u: scores[u], reverse=True)[: int(max_results)]
    return [by_uuid[u] for u in ranked]


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

    from falkordb import FalkorDB

    graph_name = _graph_name_for_client(client_slug)
    db = FalkorDB(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
    )
    graph = (
        _select_existing_probe_graph(db, graph_name)
        if settings.graphiti_acceptance_probe_mode
        else db.select_graph(graph_name)
    )
    edge_rows = _graph_read(
        graph,
        """
        MATCH (subject:Entity)-[edge:RELATES_TO]->(object:Entity)
        WHERE edge.uuid IN $edge_uuids AND edge.group_id = $group_id
        RETURN edge.uuid, subject.uuid, subject.name, edge.name,
               object.uuid, object.name, edge.fact, edge.episodes,
               edge.valid_at, edge.invalid_at, edge.expired_at
        """,
        params={"edge_uuids": ordered_edge_ids, "group_id": graph_name},
    ).result_set

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
        source_rows = _graph_read(
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
        ).result_set
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
