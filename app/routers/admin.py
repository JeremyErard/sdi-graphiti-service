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
