"""Health check endpoint."""

import logging

from fastapi import APIRouter

from app.config import settings

logger = logging.getLogger("graphiti_service")

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check with FalkorDB connectivity and memory stats."""
    import redis.asyncio as redis

    status = {
        "status": "ok",
        "service": "sdi-graphiti-service",
        "falkordb": {"host": settings.falkordb_host, "port": settings.falkordb_port},
    }

    try:
        r = redis.Redis(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
            decode_responses=True,
        )
        info = await r.info("memory")
        status["falkordb"]["connected"] = True
        status["falkordb"]["used_memory_human"] = info.get("used_memory_human", "unknown")
        status["falkordb"]["used_memory_peak_human"] = info.get(
            "used_memory_peak_human", "unknown"
        )
        status["falkordb"]["maxmemory_human"] = info.get("maxmemory_human", "unknown")

        # Get list of graphs (FalkorDB-specific command)
        try:
            graphs = await r.execute_command("GRAPH.LIST")
            status["falkordb"]["graphs"] = graphs if graphs else []
            status["falkordb"]["graph_count"] = len(graphs) if graphs else 0
        except Exception:
            status["falkordb"]["graphs"] = []
            status["falkordb"]["graph_count"] = 0

        await r.aclose()
    except Exception as e:
        status["status"] = "degraded"
        status["falkordb"]["connected"] = False
        status["falkordb"]["error"] = str(e)
        logger.error(f"[graphiti] Health check — FalkorDB connection failed: {e}")

    return status
