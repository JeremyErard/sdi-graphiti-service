"""Public liveness must not disclose tenants or infrastructure details."""

import asyncio

import pytest
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import health
from app.services import graphiti_client

DEPLOY_COMMIT = "273e49df13ad42854c68d11670a20514ba1bb7c2"


class FakeRedis:
    def __init__(self, *, should_fail: bool = False, **_kwargs):
        self.should_fail = should_fail

    async def ping(self):
        if self.should_fail:
            raise ConnectionError("private-hostname:6379 unavailable")
        return True

    async def aclose(self):
        return None


def client() -> TestClient:
    app = FastAPI()
    app.include_router(health.router)
    return TestClient(app)


def test_health_exposes_only_minimal_liveness(monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", DEPLOY_COMMIT.upper())
    monkeypatch.setattr(redis, "Redis", FakeRedis)
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "sdi-graphiti-service",
        "commit": DEPLOY_COMMIT,
        "falkordb": {"connected": True},
    }


def test_health_does_not_echo_private_failure_details(monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "private-hostname:6379/injected")
    monkeypatch.setattr(redis, "Redis", lambda **kwargs: FakeRedis(should_fail=True, **kwargs))
    response = client().get("/health")
    body = response.json()
    assert body == {
        "status": "degraded",
        "service": "sdi-graphiti-service",
        "commit": "unknown",
        "falkordb": {"connected": False},
    }
    serialized = response.text.lower()
    for forbidden in ("client_", "host", "port", "memory", "persistence", "private-hostname"):
        assert forbidden not in serialized


def test_ready_exposes_only_component_booleans(monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", DEPLOY_COMMIT)

    async def falkordb_probe():
        return {
            "ok": True,
            "used_memory_human": "private-memory-detail",
            "maxmemory_human": "private-limit-detail",
            "headroom_pct": 42.0,
            "low_headroom": False,
        }

    async def embedder_probe():
        return {"ok": True, "provider": "private-provider", "model": "private-model"}

    async def llm_probe():
        return {"ok": True, "provider": "private-llm", "model": "private-llm-model"}

    monkeypatch.setattr(health, "_probe_falkordb", falkordb_probe)
    monkeypatch.setattr(health, "_probe_embedder", embedder_probe)
    monkeypatch.setattr(health, "_probe_llm", llm_probe)
    monkeypatch.setattr(health, "_ready_cache", None)

    response = client().get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "sdi-graphiti-service",
        "commit": DEPLOY_COMMIT,
        "checks": {
            "data_store": {"ready": True},
            "retrieval": {"ready": True},
            "generation": {"ready": True},
        },
    }
    serialized = response.text.lower()
    for forbidden in ("memory", "headroom", "provider", "model", "error", "falkordb"):
        assert forbidden not in serialized


def test_deploy_commit_is_exact_or_bounded(monkeypatch):
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    assert health._deploy_commit() == "unknown"

    monkeypatch.setenv("RENDER_GIT_COMMIT", "a" * 39)
    assert health._deploy_commit() == "unknown"

    monkeypatch.setenv("RENDER_GIT_COMMIT", "B" * 40)
    assert health._deploy_commit() == "b" * 40


def test_probe_process_readiness_never_calls_generative_provider(monkeypatch):
    async def falkordb_probe():
        return {"ok": True, "low_headroom": False}

    async def fast_path_embedder_probe():
        return {"ok": True, "provider": "embedding", "model": "query-model"}

    async def forbidden_default_embedder_probe():
        raise AssertionError("probe readiness must use the exact fast-path embedder")

    async def forbidden_llm_probe():
        raise AssertionError("probe process must not make a generative call")

    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", True)
    monkeypatch.setattr(health, "_probe_falkordb", falkordb_probe)
    monkeypatch.setattr(
        health,
        "_probe_fast_path_embedder",
        fast_path_embedder_probe,
    )
    monkeypatch.setattr(health, "_probe_embedder", forbidden_default_embedder_probe)
    monkeypatch.setattr(health, "_probe_llm", forbidden_llm_probe)
    monkeypatch.setattr(health, "_ready_cache", None)

    response = client().get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {
        "data_store": {"ready": True},
        "retrieval": {"ready": True},
        "generation": {"ready": False},
    }


def test_probe_readiness_fails_without_the_exact_fast_path_embedder(monkeypatch):
    async def falkordb_probe():
        return {"ok": True, "low_headroom": False}

    async def forbidden_llm_probe():
        raise AssertionError("probe process must not make a generative call")

    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", True)
    monkeypatch.setattr(health, "_probe_falkordb", falkordb_probe)
    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: None)
    monkeypatch.setattr(health, "_probe_llm", forbidden_llm_probe)
    monkeypatch.setattr(health, "_ready_cache", None)

    response = client().get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "data_store": {"ready": True},
        "retrieval": {"ready": False},
        "generation": {"ready": False},
    }


def test_probe_readiness_uses_search_embedder_factory_and_dimension(monkeypatch):
    observed = []

    class _Embedder:
        async def create(self, *, input_data):
            observed.append(input_data)
            return [0.1] * settings.embedding_dim

    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: _Embedder())

    result = asyncio.run(health._probe_fast_path_embedder())

    assert result["ok"] is True
    assert result["provider"] == "voyage"
    assert observed == [["readiness probe"]]


def test_probe_readiness_rejects_fast_path_dimension_mismatch(monkeypatch):
    class _Embedder:
        async def create(self, *, input_data):
            return [0.1] * (settings.embedding_dim - 1)

    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: _Embedder())

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        asyncio.run(health._probe_fast_path_embedder())
