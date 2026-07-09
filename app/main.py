"""SDI Graphiti Service — Temporal Knowledge Graph API for Engage Platform."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.auth import require_scope, validate_auth_configuration
from app.config import settings
from app.routers import admin, graph, health, ingest, search, structured
from app.services import graphiti_client, graphiti_patches

# Install runtime patches over graphiti-core BEFORE any router import paths
# trigger an ingestion / read. See graphiti_patches.py for the bug detail.
graphiti_patches.install()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("graphiti_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    validate_auth_configuration()
    if settings.graphiti_auth_mode != "required":
        logger.warning(
            "[graphiti-auth] service perimeter is %s; set "
            "GRAPHITI_AUTH_MODE=required after the coordinated credential rollout",
            settings.graphiti_auth_mode,
        )
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
app.include_router(
    ingest.router,
    prefix="/ingest",
    tags=["ingestion"],
    dependencies=[Depends(require_scope("ingest"))],
)
app.include_router(
    structured.router,
    prefix="/ingest",
    tags=["ingestion"],
    dependencies=[Depends(require_scope("ingest"))],
)
app.include_router(
    search.router,
    prefix="/search",
    tags=["search"],
    dependencies=[Depends(require_scope("search"))],
)
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_scope("admin"))],
)
app.include_router(
    graph.router,
    prefix="/graph",
    tags=["graph"],
    dependencies=[Depends(require_scope("search"))],
)
