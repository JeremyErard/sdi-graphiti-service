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


@router.get("/ready")
async def readiness_check():
    """Deep readiness probe.

    /health only verifies FalkorDB connectivity and always returns HTTP 200, so
    it reports "ok" even when the embedding provider is exhausted and retrieval
    is dark for every client (the gray-failure problem). /ready actively
    exercises the embedder and the extraction LLM and checks FalkorDB memory
    headroom, and returns HTTP 503 when a CRITICAL dependency (the embedder or
    FalkorDB connectivity) is unavailable — so an outage pages instead of
    hiding. The LLM check is informational and does not force a 503.

    Each dependency is probed inside its own try/except; the endpoint itself
    never raises.
    """
    from fastapi.responses import JSONResponse

    checks: dict = {}
    critical_ok = True

    # --- FalkorDB: connectivity + memory headroom (critical) ---
    try:
        import redis.asyncio as redis

        r = redis.Redis(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
        )
        info = await r.info("memory")
        await r.aclose()
        used = int(info.get("used_memory", 0) or 0)
        maxm = int(info.get("maxmemory", 0) or 0)
        headroom_pct = round(100 * (1 - used / maxm), 1) if maxm else None
        checks["falkordb"] = {
            "ok": True,
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "maxmemory_human": info.get("maxmemory_human", "unknown"),
            "headroom_pct": headroom_pct,
            "low_headroom": headroom_pct is not None and headroom_pct < 10,
        }
    except Exception as e:
        critical_ok = False
        checks["falkordb"] = {"ok": False, "error": str(e)[:200]}

    # --- Embedder: a minimal real embedding call (critical — the live SPOF) ---
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=15)
        await client.embeddings.create(model="text-embedding-3-small", input="readiness probe")
        checks["embedder"] = {"ok": True, "provider": "openai", "model": "text-embedding-3-small"}
    except Exception as e:
        critical_ok = False
        checks["embedder"] = {"ok": False, "provider": "openai", "error": str(e)[:200]}

    # --- Extraction LLM: minimal ping (informational; does NOT force 503) ---
    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=15)
        await client.messages.create(
            model=settings.graphiti_llm_model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        checks["llm"] = {"ok": True, "provider": "anthropic", "model": settings.graphiti_llm_model}
    except Exception as e:
        checks["llm"] = {
            "ok": False,
            "provider": "anthropic",
            "model": settings.graphiti_llm_model,
            "error": str(e)[:200],
        }

    body = {
        "status": "ready" if critical_ok else "degraded",
        "service": "sdi-graphiti-service",
        "checks": checks,
    }
    if not critical_ok:
        logger.error(f"[graphiti] Readiness check FAILED: {checks}")
    return JSONResponse(status_code=200 if critical_ok else 503, content=body)
