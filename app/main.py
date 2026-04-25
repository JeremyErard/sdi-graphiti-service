"""SDI Graphiti Service — Temporal Knowledge Graph API for Engage Platform."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routers import admin, graph, health, ingest, outbound, search, structured
from app.services import graphiti_client

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("graphiti_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info(
        f"[graphiti] Starting service — FalkorDB at "
        f"{settings.falkordb_host}:{settings.falkordb_port}"
    )
    yield
    logger.info("[graphiti] Shutting down — closing graph connections")
    await graphiti_client.close_all()


app = FastAPI(
    title="SDI Graphiti Service",
    description="Temporal knowledge graph API for the SDI Engage platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(structured.router, prefix="/ingest", tags=["ingestion"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(graph.router, prefix="/graph", tags=["graph"])
app.include_router(outbound.router, prefix="/outbound", tags=["outbound-gtm"])
