"""Admin endpoints — graph initialization and management."""

import logging
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.config import settings
from app.services import graphiti_client
from app.services.provenance_stats import (
    ProvenanceStatsReadError,
    provenance_stats_for_graph,
)

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

    def _assert_dim(vectors: list) -> None:
        # The emb_model marker encodes the dimension; if the embedder ever returns
        # a different dimension than configured (e.g. embedding_dim set above what
        # the model can produce, so truncation can't reach it), writing would
        # stamp a marker that lies about the vector. Abort before any such write
        # so the graph can never end up mixed-dimension. Never triggers when
        # model + embedding_dim agree (voyage-4-large -> 1024 == 1024).
        if vectors and len(vectors[0]) != settings.embedding_dim:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Embedder produced {len(vectors[0])}-dim vectors but config "
                    f"embedding_dim={settings.embedding_dim} (marker '{marker}'); "
                    "aborting to keep the graph single-dimension."
                ),
            )

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
                _assert_dim(vectors)
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
                _assert_dim(vectors)
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


@router.post("/falkordb-save")
async def falkordb_save():
    """Force an RDB snapshot (BGSAVE) so the current in-memory state — graphs,
    re-embeddings, vector indexes — is flushed to the mounted persistent disk.
    Use before any FalkorDB restart/upgrade. Returns post-save persistence stats."""
    import asyncio

    import redis.asyncio as redis

    r = redis.Redis(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
        decode_responses=True,
    )
    try:
        try:
            await r.bgsave()
        except Exception as e:
            # A bgsave already in progress is fine; surface other errors below.
            logger.info(f"[graphiti] bgsave: {e}")
        for _ in range(40):
            info = await r.info("persistence")
            if not int(info.get("rdb_bgsave_in_progress", 0) or 0):
                break
            await asyncio.sleep(0.5)
        info = await r.info("persistence")
        cfg = await r.config_get("dir")
        result = {
            "status": "saved",
            "dir": cfg.get("dir", ""),
            "rdb_last_save_time": info.get("rdb_last_save_time"),
            "rdb_changes_since_last_save": info.get("rdb_changes_since_last_save"),
            "rdb_last_bgsave_status": info.get("rdb_last_bgsave_status"),
        }
        logger.warning(f"[graphiti] forced RDB save: {result}")
        return result
    except Exception as e:
        logger.error(f"[graphiti] falkordb-save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")
    finally:
        try:
            await r.aclose()
        except Exception:
            pass


class ExportGraphRequest(BaseModel):
    client_slug: str
    kind: str  # "nodes" | "edges" | "all_nodes" | "all_edges"
    offset: int = 0
    limit: int = 500
    graph_name: str | None = None  # raw graph name override (e.g. segment_*)


@router.post("/export-graph")
async def export_graph(req: ExportGraphRequest):
    """Paginated content export of a graph's Entity nodes and RELATES_TO edges
    (uuid + text + structural props; embeddings omitted — they are regenerable
    verbatim via re-embed from the same text + embedder). A safety-net backup
    that can fully reconstruct the searchable substrate if needed. Read-only.
    """
    from falkordb import FalkorDB

    graph_name = req.graph_name or graphiti_client._graph_name_for_client(req.client_slug)
    db = FalkorDB(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
    )
    graph = db.select_graph(graph_name)
    off, lim = int(req.offset), int(req.limit)

    try:
        if req.kind == "nodes":
            total = graph.query("MATCH (n:Entity) RETURN count(n)").result_set[0][0]
            rows = graph.query(
                "MATCH (n:Entity) RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, "
                "n.group_id AS group_id, n.created_at AS created_at, labels(n) AS labels "
                f"ORDER BY n.uuid SKIP {off} LIMIT {lim}"
            ).result_set
            out = [
                {"uuid": r[0], "name": r[1], "summary": r[2], "group_id": r[3],
                 "created_at": r[4], "labels": r[5]}
                for r in rows
            ]
        elif req.kind == "edges":
            total = graph.query(
                "MATCH ()-[e:RELATES_TO]->() RETURN count(e)"
            ).result_set[0][0]
            rows = graph.query(
                "MATCH ()-[e:RELATES_TO]->() RETURN e.uuid AS uuid, e.fact AS fact, e.name AS name, "
                "e.source_uuid AS src, e.target_uuid AS tgt, e.group_id AS group_id, "
                "e.created_at AS created_at, e.valid_at AS valid_at, e.invalid_at AS invalid_at, "
                "e.expired_at AS expired_at, e.episodes AS episodes "
                f"ORDER BY e.uuid SKIP {off} LIMIT {lim}"
            ).result_set
            out = [
                {"uuid": r[0], "fact": r[1], "name": r[2], "source_uuid": r[3],
                 "target_uuid": r[4], "group_id": r[5], "created_at": r[6],
                 "valid_at": r[7], "invalid_at": r[8], "expired_at": r[9], "episodes": r[10]}
                for r in rows
            ]
        elif req.kind == "all_nodes":
            # FULL-FIDELITY: every node of every label, all properties (embeddings
            # stripped — regenerated verbatim by re-embed). ORDER BY id(n) for
            # stable pagination.
            total = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
            rows = graph.query(
                f"MATCH (n) RETURN labels(n) AS labels, properties(n) AS props "
                f"ORDER BY id(n) SKIP {off} LIMIT {lim}"
            ).result_set
            out = []
            for r in rows:
                props = dict(r[1]) if r[1] else {}
                # Strip the vector AND the emb_model provenance marker — both are
                # regenerated by re-embed. Keeping the marker would make restored
                # nodes look already-embedded and skip re-embedding (no vectors).
                props.pop("name_embedding", None)
                props.pop("emb_model", None)
                out.append({"labels": list(r[0] or []), "props": props})
        elif req.kind == "all_edges":
            total = graph.query("MATCH ()-[e]->() RETURN count(e)").result_set[0][0]
            rows = graph.query(
                f"MATCH (a)-[e]->(b) RETURN type(e) AS t, properties(e) AS props, "
                f"a.uuid AS src, b.uuid AS tgt ORDER BY id(e) SKIP {off} LIMIT {lim}"
            ).result_set
            out = []
            for r in rows:
                props = dict(r[1]) if r[1] else {}
                props.pop("fact_embedding", None)
                props.pop("emb_model", None)
                out.append({"type": r[0], "props": props, "src": r[2], "tgt": r[3]})
        else:
            raise HTTPException(
                status_code=400, detail="kind must be nodes|edges|all_nodes|all_edges"
            )

        return {
            "graph_name": graph_name,
            "kind": req.kind,
            "total": total,
            "offset": off,
            "count": len(out),
            "done": off + len(out) >= total,
            "rows": out,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[graphiti] export-graph failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


def _san_label(s: str) -> str:
    return "".join(c for c in str(s) if c.isalnum() or c == "_")


class ImportGraphRequest(BaseModel):
    client_slug: str
    kind: str  # "nodes" | "edges"
    rows: list[dict]
    confirm: str  # must equal "import"
    graph_name: str | None = None  # raw graph name override (e.g. segment_*)


@router.post("/import-graph")
async def import_graph(req: ImportGraphRequest):
    """Recreate nodes/edges from a full-fidelity export (all_nodes / all_edges).

    Nodes are grouped by label-set and bulk-CREATEd with their full property map
    (labels inlined + sanitized — graphiti labels are alphanumeric/underscore).
    Edges are grouped by type and connected by endpoint uuid. Embeddings are NOT
    restored here (regenerated afterward by /admin/reembed-graph). Import nodes
    before edges so the endpoint MATCHes resolve. Idempotent on a fresh graph.
    """
    if req.confirm != "import":
        raise HTTPException(status_code=400, detail='Confirmation required: confirm="import"')

    from collections import defaultdict

    from falkordb import FalkorDB

    graph_name = req.graph_name or graphiti_client._graph_name_for_client(req.client_slug)
    db = FalkorDB(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
    )
    graph = db.select_graph(graph_name)
    imported = 0
    skipped = 0

    try:
        if req.kind == "nodes":
            groups: dict[tuple, list] = defaultdict(list)
            for row in req.rows:
                labels = tuple(
                    _san_label(x) for x in (row.get("labels") or []) if _san_label(x)
                )
                props = dict(row.get("props") or {})
                # Never restore embeddings or the provenance marker — re-embed
                # regenerates both; a stale marker would suppress re-embedding.
                props.pop("name_embedding", None)
                props.pop("emb_model", None)
                groups[labels].append(props)
            for labels, propslist in groups.items():
                lbl = "".join(f":{x}" for x in labels) or ":Entity"
                graph.query(
                    f"UNWIND $rows AS p CREATE (n{lbl}) SET n = p", {"rows": propslist}
                )
                imported += len(propslist)
        elif req.kind == "edges":
            groups_e: dict[str, list] = defaultdict(list)
            for row in req.rows:
                t = _san_label(row.get("type") or "RELATES_TO")
                if not row.get("src") or not row.get("tgt"):
                    skipped += 1
                    continue
                props = dict(row.get("props") or {})
                props.pop("fact_embedding", None)
                props.pop("emb_model", None)
                groups_e[t].append({"props": props, "src": row["src"], "tgt": row["tgt"]})
            for t, items in groups_e.items():
                res = graph.query(
                    f"UNWIND $rows AS r MATCH (a {{uuid: r.src}}) MATCH (b {{uuid: r.tgt}}) "
                    f"CREATE (a)-[e:{t}]->(b) SET e = r.props",
                    {"rows": items},
                )
                # relationships_created tells us how many actually connected.
                created = getattr(res, "relationships_created", None)
                imported += created if created is not None else len(items)
        else:
            raise HTTPException(status_code=400, detail="kind must be 'nodes' or 'edges'")

        return {"graph_name": graph_name, "kind": req.kind, "imported": imported, "skipped": skipped}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[graphiti] import-graph failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


class PersistToDiskRequest(BaseModel):
    confirm: str  # must equal "persist to /data"
    target_dir: str = "/data"


@router.post("/falkordb-persist-to-disk")
async def falkordb_persist_to_disk(req: PersistToDiskRequest):
    """Point FalkorDB's persistence dir at the mounted disk and flush RDB + AOF
    there, so the data is durable on the disk before a mount-path fix/restart.

    This is the in-place prep for relocating the disk to where FalkorDB writes:
    after this runs, the persistent disk holds a full copy; remounting that disk
    to FalkorDB's default data dir then lets a restart auto-restore. Idempotent.
    """
    if req.confirm != "persist to /data":
        raise HTTPException(status_code=400, detail='Confirmation required: confirm="persist to /data"')

    import asyncio

    import redis.asyncio as redis

    r = redis.Redis(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
        decode_responses=True,
    )
    try:
        before = {
            "dir": (await r.config_get("dir")).get("dir"),
            "dbfilename": (await r.config_get("dbfilename")).get("dbfilename"),
            "appendonly": (await r.config_get("appendonly")).get("appendonly"),
            "appenddirname": (await r.config_get("appenddirname")).get("appenddirname"),
        }
        # Repoint persistence at the mounted disk.
        await r.config_set("dir", req.target_dir)

        aof_on = str(before.get("appendonly", "no")).lower() == "yes"
        if aof_on:
            try:
                await r.execute_command("BGREWRITEAOF")
            except Exception as e:
                logger.info(f"[graphiti] bgrewriteaof: {e}")
        try:
            await r.bgsave()
        except Exception as e:
            logger.info(f"[graphiti] bgsave: {e}")

        # Wait for both rewrite + save to finish.
        for _ in range(120):
            info = await r.info("persistence")
            busy = int(info.get("rdb_bgsave_in_progress", 0) or 0) or int(
                info.get("aof_rewrite_in_progress", 0) or 0
            )
            if not busy:
                break
            await asyncio.sleep(0.5)

        info = await r.info("persistence")
        after_dir = (await r.config_get("dir")).get("dir")
        result = {
            "status": "persisted",
            "before": before,
            "after_dir": after_dir,
            "aof_enabled": info.get("aof_enabled"),
            "aof_last_bgrewrite_status": info.get("aof_last_bgrewrite_status"),
            "rdb_last_bgsave_status": info.get("rdb_last_bgsave_status"),
            "rdb_last_save_time": info.get("rdb_last_save_time"),
            "rdb_changes_since_last_save": info.get("rdb_changes_since_last_save"),
        }
        logger.warning(f"[graphiti] persist-to-disk: {result}")
        return result
    except Exception as e:
        logger.error(f"[graphiti] persist-to-disk failed: {e}")
        raise HTTPException(status_code=500, detail=f"Persist failed: {str(e)}")
    finally:
        try:
            await r.aclose()
        except Exception:
            pass


class ProvenanceStatusCount(BaseModel):
    structural_status: Literal["chained", "pre_chain", "malformed"]
    count: int


class ProvenanceEpisodeTypeCount(ProvenanceStatusCount):
    episode_type: str


class ProvenanceEngagementCount(ProvenanceStatusCount):
    engagement_id: str


class ProvenanceGraphStats(BaseModel):
    facts_total: int
    malformed_response_events: int
    by_structural_status: list[ProvenanceStatusCount]
    by_episode_type: list[ProvenanceEpisodeTypeCount]
    by_engagement: list[ProvenanceEngagementCount]


class GraphStat(BaseModel):
    graph_name: str
    nodes: int
    edges: int
    provenance: ProvenanceGraphStats | None = None


class GraphStatsResponse(BaseModel):
    graphs: list[GraphStat]
    graph_count: int


class GraphStatsRequest(BaseModel):
    client_slug: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")
    include_provenance: bool = False

    @model_validator(mode="after")
    def require_tenant_for_provenance(self):
        if self.include_provenance and self.client_slug is None:
            raise ValueError(
                "include_provenance=true requires one exact client_slug"
            )
        return self


@router.post(
    "/graph-stats",
    response_model=GraphStatsResponse,
    response_model_exclude_none=True,
)
async def graph_stats(req: GraphStatsRequest):
    """Read-only node/edge totals and opt-in provenance aggregates per graph.

    Fills the observability gap where totals were previously obtainable only
    via the mutating /admin/reembed-graph or the heavyweight /admin/export-graph
    — routine monitoring and the ingestion completeness audit (substrate P0)
    need a cheap answer. The optional client slug is carried in the signed JSON
    body, not an unsigned query string; otherwise every graph is counted. Uses
    POST so the existing backend/operator HMAC clients bind the exact method,
    body, scope, and tenant without inventing a second canonicalization contract.
    Inherits the admin router's auth scope. The default path remains the original
    two COUNT queries. ``include_provenance=true`` adds metadata-only queries and
    returns no fact text, names, descriptions, or source content.
    """
    try:
        from falkordb import FalkorDB

        db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
        )
        listed_graphs = db.list_graphs()
        if not isinstance(listed_graphs, (list, tuple, set, frozenset)):
            raise RuntimeError("unsupported graph inventory response")
        available_graphs = frozenset(listed_graphs)
        if req.client_slug:
            requested_name = graphiti_client._graph_name_for_client(req.client_slug)
            if requested_name not in available_graphs:
                raise HTTPException(
                    status_code=404,
                    detail="GRAPH_STATS_GRAPH_NOT_FOUND",
                )
            names = [requested_name]
        else:
            names = sorted(available_graphs)

        stats: list[GraphStat] = []
        for name in names:
            graph = db.select_graph(name)
            # The two default COUNT reads stay on GRAPH.QUERY, the command this
            # endpoint has always shipped. GRAPH.RO_QUERY has not been exercised
            # against the deployed FalkorDB, and this path runs on every call in
            # every mode; it is not the place to introduce an unproven command.
            # The opt-in provenance aggregates below do use ro_query, and that
            # path is separately activation-blocked pending the live proof
            # recorded in docs/PROVENANCE_OPS.md.
            nodes = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
            edges = graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
            provenance = (
                ProvenanceGraphStats.model_validate(
                    provenance_stats_for_graph(graph, name)
                )
                if req.include_provenance
                else None
            )
            stats.append(
                GraphStat(
                    graph_name=name,
                    nodes=int(nodes),
                    edges=int(edges),
                    provenance=provenance,
                )
            )
        return GraphStatsResponse(graphs=stats, graph_count=len(stats))
    except ProvenanceStatsReadError as exc:
        raise HTTPException(status_code=409, detail=exc.code) from None
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - connection-level failures
        logger.error("graph-stats failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="graph-stats failed")
