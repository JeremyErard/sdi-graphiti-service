"""Admin endpoints — graph initialization and management."""

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services import graphiti_client

logger = logging.getLogger("graphiti_service")

router = APIRouter()

REEMBED_CONFIRM = "I understand this overwrites all embeddings"


def _chunk(items: list, size: int):
    """Yield successive `size`-length slices of `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


class InitGraphRequest(BaseModel):
    client_slug: str


class InitGraphResponse(BaseModel):
    graph_name: str
    status: str


class ResetGraphRequest(BaseModel):
    client_slug: str
    confirm: str  # must equal "I understand this wipes all data"


class ResetGraphResponse(BaseModel):
    graph_name: str
    status: str


class DeleteGraphRequest(BaseModel):
    client_slug: str
    confirm: str  # must equal "I understand this wipes all data"


class DeleteGraphResponse(BaseModel):
    graph_name: str
    status: str


@router.post("/init-graph", response_model=InitGraphResponse)
async def init_graph(req: InitGraphRequest):
    """Initialize a new knowledge graph for a client.

    Creates indices and constraints in FalkorDB. Idempotent — safe to call multiple times.
    Called during client provisioning from the Engage backend.
    """
    try:
        graph_name = await graphiti_client.init_graph(req.client_slug)
        logger.info(f"[graphiti] Graph initialized: {graph_name}")
        return InitGraphResponse(graph_name=graph_name, status="initialized")
    except Exception as e:
        logger.error(f"[graphiti] Graph init failed for {req.client_slug}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Graph initialization failed: {str(e)}"
        )


@router.post("/reset-graph", response_model=ResetGraphResponse)
async def reset_graph(req: ResetGraphRequest):
    """DESTRUCTIVE: wipe all data from a client's graph and re-initialize.

    Requires confirm = "I understand this wipes all data" to prevent accidents.
    Used for clean-slate backfills when we need to guarantee graph state.
    """
    if req.confirm != "I understand this wipes all data":
        raise HTTPException(
            status_code=400,
            detail='Confirmation required: set confirm="I understand this wipes all data"',
        )
    try:
        result = await graphiti_client.reset_graph(req.client_slug)
        logger.warning(f"[graphiti] Graph reset: {result['graph_name']}")
        return ResetGraphResponse(graph_name=result["graph_name"], status=result["status"])
    except Exception as e:
        logger.error(f"[graphiti] Graph reset failed for {req.client_slug}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Graph reset failed: {str(e)}"
        )


@router.post("/delete-graph", response_model=DeleteGraphResponse)
async def delete_graph(req: DeleteGraphRequest):
    """DESTRUCTIVE: drop a client's graph entirely (no re-init).

    Use for removing obsolete or test graphs. Unlike /admin/reset-graph, this
    leaves the graph fully deleted — no indices, no empty shell, no entry in
    /health's `graphs` list.
    """
    if req.confirm != "I understand this wipes all data":
        raise HTTPException(
            status_code=400,
            detail='Confirmation required: set confirm="I understand this wipes all data"',
        )
    try:
        from app.services import graphiti_client as gc
        from falkordb import FalkorDB

        graph_name = gc._graph_name_for_client(req.client_slug)
        # Evict cached Graphiti client so a new one won't reference a stale graph.
        if graph_name in gc._clients:
            try:
                await gc._clients[graph_name].close()
            except Exception:
                pass
            del gc._clients[graph_name]

        # Use the falkordb-py library directly. graphiti_core's FalkorDriver
        # wraps the connection in a way that doesn't expose raw Redis commands,
        # so reaching through driver.client.execute_command silently fails for
        # graph-key cleanup. The native FalkorDB().select_graph(name).delete()
        # API issues GRAPH.DELETE properly and removes the graph from GRAPH.LIST.
        db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
        )
        graph = db.select_graph(graph_name)
        try:
            graph.delete()
            logger.warning(f"[graphiti] Graph deleted via falkordb-py: {graph_name}")
        except Exception as del_err:
            # Most common reason for delete to error: graph already gone.
            logger.info(f"[graphiti] graph.delete() {graph_name}: {del_err} (likely already absent)")

        return DeleteGraphResponse(graph_name=graph_name, status="deleted")
    except Exception as e:
        logger.error(f"[graphiti] Graph delete failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Graph delete failed: {str(e)}")


class ReembedGraphRequest(BaseModel):
    client_slug: str
    confirm: str  # must equal REEMBED_CONFIRM
    dry_run: bool = False
    # Process at most `max_items` STALE embeddings per call so a large graph
    # (Pokagon ~10k vectors) never exceeds the Render request timeout. The caller
    # simply re-calls until `done` is true — no offset bookkeeping, because each
    # call grabs the next batch of not-yet-migrated items (staleness-based, so it
    # is convergent, idempotent, and immune to concurrent ingestion).
    max_items: int = 1500
    batch_size: int = 128


class ReembedGraphResponse(BaseModel):
    graph_name: str
    embedder_model: str
    embedding_dim: int
    marker: str
    nodes_total: int
    edges_total: int
    nodes_reembedded: int  # this call
    edges_reembedded: int  # this call
    stale_nodes_remaining: int  # after this call
    stale_edges_remaining: int  # after this call
    failures: int
    done: bool
    dry_run: bool
    elapsed_ms: float
    sample: dict


@router.post("/reembed-graph", response_model=ReembedGraphResponse)
async def reembed_graph(req: ReembedGraphRequest):
    """Recompute embeddings IN PLACE for an existing client graph.

    Used to migrate the embedder (e.g. OpenAI -> Voyage). Rewrites ONLY the
    `name_embedding` (Entity nodes) and `fact_embedding` (RELATES_TO edges)
    vector properties using the service's currently-configured embedder, and
    stamps each with an `emb_model` provenance marker. It does NOT add, delete,
    or otherwise modify nodes/edges, and it does NOT re-run LLM entity
    extraction — graph CONTENT is preserved byte-for-byte; only the vectors (and
    the marker) change.

    Staleness-based: each call selects items whose `emb_model` is not yet the
    current marker, so it is convergent (loop until `done`), idempotent (re-runs
    are no-ops), and concurrency-safe (a node freshly ingested with the new
    embedder is simply re-stamped, never skipped). `done` (stale remaining == 0)
    is a definitive completeness proof — important because a single un-migrated
    vector of the wrong dimension can make FalkorDB cosine search error.

    Fail-loud: refuses to run unless an explicit embedder is configured
    (VOYAGE_API_KEY set), so it never silently re-embeds with the exhausted
    OpenAI default we are migrating off of.
    """
    if req.confirm != REEMBED_CONFIRM:
        raise HTTPException(
            status_code=400,
            detail=f'Confirmation required: set confirm="{REEMBED_CONFIRM}"',
        )

    # Use the SAME embedder factory the service uses for ingest + search, so the
    # vectors we write are produced by an identical code path to the query
    # vectors they will be compared against (no query/document asymmetry).
    embedder = graphiti_client._create_embedder()
    if embedder is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No explicit embedder configured (VOYAGE_API_KEY unset). Refusing "
                "to re-embed with the default OpenAI path. Set VOYAGE_API_KEY first."
            ),
        )

    graph_name = graphiti_client._graph_name_for_client(req.client_slug)
    marker = f"{settings.embedding_model}:{settings.embedding_dim}"
    start = time.time()
    failures = 0
    sample: dict = {}

    def _count(q: str) -> int:
        return graph.query(q, {"marker": marker}).result_set[0][0]

    try:
        from falkordb import FalkorDB

        db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
        )
        graph = db.select_graph(graph_name)

        NODE_FILTER = "n.name IS NOT NULL AND n.name <> ''"
        EDGE_FILTER = "e.fact IS NOT NULL AND e.fact <> ''"
        NODE_STALE = f"({NODE_FILTER}) AND (n.emb_model IS NULL OR n.emb_model <> $marker)"
        EDGE_STALE = f"({EDGE_FILTER}) AND (e.emb_model IS NULL OR e.emb_model <> $marker)"

        nodes_total = graph.query(f"MATCH (n:Entity) WHERE {NODE_FILTER} RETURN count(n)").result_set[0][0]
        edges_total = graph.query(f"MATCH ()-[e:RELATES_TO]->() WHERE {EDGE_FILTER} RETURN count(e)").result_set[0][0]

        budget = max(0, req.max_items)
        nodes_done = 0
        edges_done = 0

        # ---- Nodes: re-embed name -> name_embedding (+ marker) ----
        if budget > 0:
            rows = graph.query(
                f"MATCH (n:Entity) WHERE {NODE_STALE} "
                f"RETURN n.uuid AS uuid, n.name AS name LIMIT {int(budget)}",
                {"marker": marker},
            ).result_set
            budget -= len(rows)
            for batch in _chunk(rows, req.batch_size):
                texts = [r[1].replace("\n", " ") for r in batch]
                uuids = [r[0] for r in batch]
                try:
                    vectors = await embedder.create_batch(texts)
                except Exception as e:
                    logger.error(f"[graphiti] reembed node batch embed failed ({graph_name}): {e}")
                    failures += len(batch)
                    continue
                for uuid, name, vec in zip(uuids, texts, vectors):
                    if not sample:
                        sample = {"kind": "node", "uuid": uuid, "name": name, "new_dim": len(vec)}
                    if req.dry_run:
                        nodes_done += 1
                        continue
                    try:
                        graph.query(
                            "MATCH (n:Entity {uuid:$uuid}) "
                            "SET n.name_embedding = vecf32($emb), n.emb_model = $marker",
                            {"uuid": uuid, "emb": vec, "marker": marker},
                        )
                        nodes_done += 1
                    except Exception as e:
                        logger.error(f"[graphiti] reembed node write failed {uuid} ({graph_name}): {e}")
                        failures += 1

        # ---- Edges: re-embed fact -> fact_embedding (+ marker), remaining budget ----
        if budget > 0:
            rows = graph.query(
                f"MATCH ()-[e:RELATES_TO]->() WHERE {EDGE_STALE} "
                f"RETURN e.uuid AS uuid, e.fact AS fact LIMIT {int(budget)}",
                {"marker": marker},
            ).result_set
            for batch in _chunk(rows, req.batch_size):
                texts = [r[1].replace("\n", " ") for r in batch]
                uuids = [r[0] for r in batch]
                try:
                    vectors = await embedder.create_batch(texts)
                except Exception as e:
                    logger.error(f"[graphiti] reembed edge batch embed failed ({graph_name}): {e}")
                    failures += len(batch)
                    continue
                for uuid, fact, vec in zip(uuids, texts, vectors):
                    if req.dry_run:
                        edges_done += 1
                        continue
                    try:
                        graph.query(
                            "MATCH ()-[e:RELATES_TO {uuid:$uuid}]->() "
                            "SET e.fact_embedding = vecf32($emb), e.emb_model = $marker",
                            {"uuid": uuid, "emb": vec, "marker": marker},
                        )
                        edges_done += 1
                    except Exception as e:
                        logger.error(f"[graphiti] reembed edge write failed {uuid} ({graph_name}): {e}")
                        failures += 1

        # Re-count stale AFTER writes — the definitive completeness signal,
        # immune to anything ingested concurrently.
        stale_nodes = _count(f"MATCH (n:Entity) WHERE {NODE_STALE} RETURN count(n)")
        stale_edges = _count(f"MATCH ()-[e:RELATES_TO]->() WHERE {EDGE_STALE} RETURN count(e)")
        done = (not req.dry_run) and stale_nodes == 0 and stale_edges == 0
        elapsed_ms = (time.time() - start) * 1000
        logger.warning(
            f"[graphiti] reembed {graph_name} dry_run={req.dry_run} marker={marker}: "
            f"nodes +{nodes_done} edges +{edges_done} failures={failures} "
            f"stale_remaining n={stale_nodes} e={stale_edges} done={done} ({elapsed_ms:.0f}ms)"
        )

        return ReembedGraphResponse(
            graph_name=graph_name,
            embedder_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            marker=marker,
            nodes_total=nodes_total,
            edges_total=edges_total,
            nodes_reembedded=nodes_done,
            edges_reembedded=edges_done,
            stale_nodes_remaining=stale_nodes,
            stale_edges_remaining=stale_edges,
            failures=failures,
            done=done,
            dry_run=req.dry_run,
            elapsed_ms=elapsed_ms,
            sample=sample,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[graphiti] reembed failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Re-embed failed: {str(e)}")
