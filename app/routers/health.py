"""Health check endpoint."""

import asyncio
import logging
import os
import re
import time

from fastapi import APIRouter

from app.config import settings

logger = logging.getLogger("graphiti_service")

router = APIRouter()

_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


def _deploy_commit() -> str:
    """Return a bounded deploy identity without reflecting arbitrary env text."""
    candidate = os.environ.get("RENDER_GIT_COMMIT", "").strip()
    return candidate.lower() if _GIT_COMMIT_RE.fullmatch(candidate) else "unknown"


@router.get("/health")
async def health_check():
    """Minimal public liveness check; never expose tenants or infrastructure."""
    import redis.asyncio as redis

    status = {
        "status": "ok",
        "service": "sdi-graphiti-service",
        "commit": _deploy_commit(),
        "falkordb": {"connected": False},
    }

    r = None
    try:
        r = redis.Redis(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        await r.ping()
        status["falkordb"]["connected"] = True
    except Exception as e:
        status["status"] = "degraded"
        logger.error(f"[graphiti] Health check — FalkorDB connection failed: {e}")
    finally:
        if r is not None:
            try:
                await r.aclose()
            except Exception:
                pass

    return status


# /ready result is cached briefly so repeated hits (uptime monitors, an attacker,
# a mis-set healthCheckPath) cannot amplify into unbounded paid embedding/LLM
# calls — at most one real probe per TTL window (adversarial findings B1/B3).
_READY_CACHE_TTL_S = 60.0
_ready_cache: dict | None = None


async def _probe_falkordb() -> dict:
    """Connectivity + memory headroom. Raises on connection failure."""
    import redis.asyncio as redis

    r = redis.Redis(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
        decode_responses=True,
        socket_connect_timeout=8,
        socket_timeout=8,
    )
    try:
        info = await r.info("memory")
    finally:
        try:
            await r.aclose()
        except Exception:
            pass
    used = int(info.get("used_memory", 0) or 0)
    maxm = int(info.get("maxmemory", 0) or 0)
    if maxm > 0:
        headroom_pct = round(100 * (1 - used / maxm), 1)
        low = headroom_pct < 10
    else:
        # Unbounded (or maxmemory directive dropped): headroom is unknown, not
        # healthy — surface None rather than a green low_headroom=False (B5).
        headroom_pct = None
        low = None
    # Logged, not returned. The readiness RESPONSE stays a boolean per check --
    # widening it would leak capacity detail to an unauthenticated caller. But
    # the numbers are already computed here and were being discarded, which is
    # why "is this instance bigger than it needs to be?" could only be answered
    # by inference. It was inferred once, wrongly, and the wrong answer is
    # still being paid for monthly.
    logger.info(
        "[graphiti] falkordb memory used=%s peak=%s rss=%s maxmemory=%s frag=%s",
        info.get("used_memory_human"),
        info.get("used_memory_peak_human"),
        info.get("used_memory_rss_human"),
        info.get("maxmemory_human"),
        info.get("mem_fragmentation_ratio"),
    )
    return {
        "ok": True,
        "used_memory_human": info.get("used_memory_human", "unknown"),
        "maxmemory_human": info.get("maxmemory_human", "unknown"),
        "headroom_pct": headroom_pct,
        "low_headroom": low,
    }


async def _probe_embedder() -> dict:
    """One real embedding call, single attempt. Raises on failure.

    Probes whichever embedder the service is actually configured to use: Voyage
    when VOYAGE_API_KEY is set (the post-cutover path), else the OpenAI default.
    This keeps /ready truthful about the embedder retrieval really depends on —
    otherwise it would keep pinging exhausted OpenAI and report degraded even
    when Voyage is healthy. Dedicated acceptance-probe processes bypass this
    provider-selection probe and exercise ``_probe_fast_path_embedder`` instead.
    """
    if settings.voyage_api_key:
        import voyageai

        client = voyageai.AsyncClient(api_key=settings.voyage_api_key, max_retries=0, timeout=8)
        await client.embed(["readiness probe"], model=settings.embedding_model)
        return {"ok": True, "provider": "voyage", "model": settings.embedding_model}

    from openai import AsyncOpenAI

    # `or None` so an empty config falls back to the OPENAI_API_KEY env var the
    # graphiti embedder itself uses, instead of sending an empty string (B6).
    client = AsyncOpenAI(api_key=settings.openai_api_key or None, timeout=8, max_retries=0)
    await client.embeddings.create(model="text-embedding-3-small", input="readiness probe")
    return {"ok": True, "provider": "openai", "model": "text-embedding-3-small"}


async def _probe_fast_path_embedder() -> dict:
    """Exercise the exact explicit embedder required by indexed fast search."""

    from app.services.graphiti_client import _create_embedder

    embedder = _create_embedder()
    if embedder is None:
        raise RuntimeError("explicit fast-path embedder is not configured")
    vector = await embedder.create(input_data=["readiness probe"])
    if not isinstance(vector, list) or len(vector) != settings.embedding_dim:
        raise RuntimeError("fast-path embedder dimension mismatch")
    return {
        "ok": True,
        "provider": "voyage",
        "model": settings.embedding_model,
    }


async def _probe_llm() -> dict:
    """One minimal LLM ping, single attempt. Raises on failure."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.anthropic_api_key or None, timeout=8, max_retries=0)
    await client.messages.create(
        model=settings.graphiti_llm_model,
        max_tokens=1,
        messages=[{"role": "user", "content": "ping"}],
    )
    return {"ok": True, "provider": "anthropic", "model": settings.graphiti_llm_model}


@router.get("/ready")
async def readiness_check():
    """Deep readiness probe.

    /health only verifies FalkorDB connectivity and always returns HTTP 200, so
    it reports "ok" even when the embedding provider is exhausted and retrieval
    is dark for every client (the gray-failure problem). /ready actively exercises
    the embedder and the extraction LLM and checks FalkorDB memory headroom, and
    returns HTTP 503 when a CRITICAL dependency (the embedder, or FalkorDB
    connectivity/headroom) is unavailable. The LLM check is informational.

    Hardened against the probe becoming a liability: the result is cached for
    ~60s so it cannot be amplified into unbounded paid calls (keep Render's
    healthCheckPath on the free /health); each dependency is probed exactly once
    (max_retries=0) and normal-mode dependencies run concurrently under one
    overall deadline. Probe mode checks only FalkorDB plus the exact fast-path
    embedder and never calls the generative provider.
    """
    from fastapi.responses import JSONResponse

    global _ready_cache

    now = time.monotonic()
    if _ready_cache is not None and now - _ready_cache["at"] < _READY_CACHE_TTL_S:
        return JSONResponse(
            status_code=_ready_cache["status_code"],
            content={**_ready_cache["body"], "cached": True},
        )

    try:
        if settings.graphiti_acceptance_probe_mode:
            fdb, emb = await asyncio.wait_for(
                asyncio.gather(
                    _probe_falkordb(),
                    _probe_fast_path_embedder(),
                    return_exceptions=True,
                ),
                timeout=12,
            )
            # Query embeddings are part of accepted retrieval behavior. A probe
            # process never exercises the generative/extraction provider.
            llm = {"ok": False, "skipped": True}
        else:
            fdb, emb, llm = await asyncio.wait_for(
                asyncio.gather(
                    _probe_falkordb(),
                    _probe_embedder(),
                    _probe_llm(),
                    return_exceptions=True,
                ),
                timeout=12,
            )
    except asyncio.TimeoutError:
        deadline = TimeoutError("overall readiness deadline (12s) exceeded")
        fdb = emb = deadline
        llm = (
            {"ok": False, "skipped": True}
            if settings.graphiti_acceptance_probe_mode
            else deadline
        )

    def _coerce(x: object, provider: str | None) -> dict:
        if isinstance(x, BaseException):
            out: dict = {"ok": False, "error": str(x)[:200]}
            if provider:
                out["provider"] = provider
            return out
        return x  # type: ignore[return-value]

    emb_provider = (
        "voyage"
        if settings.graphiti_acceptance_probe_mode or settings.voyage_api_key
        else "openai"
    )
    checks = {
        "falkordb": _coerce(fdb, None),
        "embedder": _coerce(emb, emb_provider),
        "llm": _coerce(llm, "anthropic"),
    }

    # Critical = embedder reachable AND FalkorDB reachable with headroom. A True
    # low_headroom (near/at the maxmemory ceiling) fails readiness because writes
    # stop under noeviction even though the read-only info() call still succeeds
    # (B4). low_headroom None (unknown/unbounded) does not fail critical.
    critical_ok = (
        bool(checks["embedder"].get("ok"))
        and bool(checks["falkordb"].get("ok"))
        and checks["falkordb"].get("low_headroom") is not True
    )

    public_checks = {
        "data_store": {"ready": bool(checks["falkordb"].get("ok"))},
        "retrieval": {"ready": bool(checks["embedder"].get("ok"))},
        "generation": {"ready": bool(checks["llm"].get("ok"))},
    }
    body = {
        "status": "ready" if critical_ok else "degraded",
        "service": "sdi-graphiti-service",
        "commit": _deploy_commit(),
        "checks": public_checks,
    }
    status_code = 200 if critical_ok else 503
    if not critical_ok:
        logger.error(f"[graphiti] Readiness check FAILED: {checks}")
    _ready_cache = {"at": now, "status_code": status_code, "body": body}
    return JSONResponse(status_code=status_code, content=body)
