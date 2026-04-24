"""Admin endpoints — graph initialization and management."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import graphiti_client

logger = logging.getLogger("graphiti_service")

router = APIRouter()


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

        graph_name = gc._graph_name_for_client(req.client_slug)
        # Evict cached Graphiti client so a new one won't reference a stale graph.
        if graph_name in gc._clients:
            try:
                await gc._clients[graph_name].close()
            except Exception:
                pass
            del gc._clients[graph_name]

        driver = gc._create_driver(graph_name)
        try:
            redis_client = driver.client if hasattr(driver, "client") else driver._client
            # GRAPH.DELETE removes graph data. DEL removes the underlying key so
            # the graph no longer appears in GRAPH.LIST.
            try:
                redis_client.execute_command("GRAPH.DELETE", graph_name)
            except Exception as graph_err:
                logger.info(f"[graphiti] GRAPH.DELETE {graph_name}: {graph_err} (continuing to DEL)")
            try:
                redis_client.execute_command("DEL", graph_name)
            except Exception as del_err:
                logger.info(f"[graphiti] DEL {graph_name}: {del_err}")
            logger.warning(f"[graphiti] Graph fully deleted: {graph_name}")
        finally:
            try:
                await driver.close()
            except Exception:
                pass

        return DeleteGraphResponse(graph_name=graph_name, status="deleted")
    except Exception as e:
        logger.error(f"[graphiti] Graph delete failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Graph delete failed: {str(e)}")
