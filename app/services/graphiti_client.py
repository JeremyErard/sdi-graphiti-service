"""Graphiti core wrapper — manages graph connections, episodes, and search."""

import logging
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import yaml
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType as GraphitiEpisodeType
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig

from app.config import settings

logger = logging.getLogger("graphiti_service")

# Cache of initialized Graphiti clients per graph name
_clients: dict[str, Graphiti] = {}

# Graphs for which we've already ensured the RELATES_TO fact_embedding vector
# index exists this process-lifetime (so we don't re-issue CREATE every search).
_edge_vindex_ensured: set[str] = set()


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


def _graph_name_for_client(client_slug: str) -> str:
    """Map client slug to isolated graph name."""
    safe_slug = "".join(c for c in client_slug if c.isalnum() or c == "_").lower()
    return f"client_{safe_slug}"


def _segment_graph_name(industry: str) -> str:
    """Map industry to segment graph name."""
    safe_industry = "".join(c for c in industry if c.isalnum() or c == "_").lower()
    return f"segment_{safe_industry}"


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
) -> dict[str, Any]:
    """Add an episode to the client's knowledge graph.

    Uses EpisodeType.text for all ingestion (plain text content).
    group_id is set to the graph name to prevent driver re-cloning.
    """
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

# Columns returned for an edge so we can build a search.py-compatible object.
_EDGE_RETURN = (
    "RETURN e.uuid AS uuid, e.fact AS fact, e.name AS name, "
    "e.source_uuid AS src, e.target_uuid AS tgt, "
    "e.valid_at AS va, e.invalid_at AS ia, e.expired_at AS ea"
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
        target_node_uuid=row[4] or "",
        valid_at=_parse_dt(row[5]),
        invalid_at=_parse_dt(row[6]),
        expired_at=_parse_dt(row[7]),
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
    embedder = _create_embedder()
    if embedder is None:
        raise RuntimeError("fast search requires an explicit (Voyage) embedder")
    qvec = await embedder.create(input_data=[query.replace("\n", " ")])

    graph_name = _graph_name_for_client(client_slug)
    from falkordb import FalkorDB

    db = FalkorDB(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
    )
    graph = db.select_graph(graph_name)
    _ensure_edge_vector_index(graph, graph_name)

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
    vres = graph.query(
        f"CALL db.idx.vector.queryRelationships('RELATES_TO', 'fact_embedding', {pool}, vecf32($q)) "
        f"YIELD relationship AS e, score {_EDGE_RETURN}",
        {"q": qvec},
    )
    vorder = _collect(vres.result_set)

    # BM25 via the fulltext index. Resilient: a parser hiccup degrades to
    # vector-only rather than failing the whole search.
    border: list[str] = []
    safe_q = _lucene_sanitize(query)
    if safe_q:
        try:
            bres = graph.query(
                f"CALL db.idx.fulltext.queryRelationships('RELATES_TO', $query) "
                f"YIELD relationship AS e, score {_EDGE_RETURN} LIMIT {pool}",
                {"query": safe_q},
            )
            border = _collect(bres.result_set)
        except Exception as e:
            logger.debug(f"[graphiti] fast BM25 leg skipped on {graph_name}: {e}")

    # Reciprocal rank fusion of the two ranked lists.
    scores: dict[str, float] = {}
    for ordered in (vorder, border):
        for rank, uuid in enumerate(ordered):
            scores[uuid] = scores.get(uuid, 0.0) + 1.0 / (rank + _RRF_K)

    ranked = sorted(scores, key=lambda u: scores[u], reverse=True)[: int(max_results)]
    return [by_uuid[u] for u in ranked]


async def search(
    client_slug: str,
    query: str,
    max_results: int = 10,
) -> list[Any]:
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

    try:
        edges = await _search_fast(client_slug, query, max_results)
        if edges:
            logger.info(
                f"[graphiti] Search(fast) in {graph_name}: {len(edges)} edges "
                f"({(time.time() - start) * 1000:.0f}ms)"
            )
            return edges
        logger.info(f"[graphiti] fast search returned 0 on {graph_name}; falling back to hybrid")
    except Exception as e:
        logger.warning(f"[graphiti] fast search failed on {graph_name} ({e}); falling back to hybrid")

    client = await get_client(client_slug)
    edges = await client.search(
        query=query,
        num_results=max_results,
        group_ids=[graph_name],
    )
    logger.info(
        f"[graphiti] Search(hybrid-fallback) in {graph_name}: {len(edges)} edges "
        f"({(time.time() - start) * 1000:.0f}ms)"
    )
    return edges


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
