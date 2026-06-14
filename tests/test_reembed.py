"""Tests for the in-place re-embed migration endpoint.

The FalkorDB data path is proven on real prod graphs (MRLA before Pokagon);
these tests lock the safety gates and prove the embed half end-to-end through
graphiti-core's exact VoyageAIEmbedder code path using the configured key.

Run: .venv/bin/python -m pytest tests/test_reembed.py -q
  or: .venv/bin/python tests/test_reembed.py   (standalone)
"""

import asyncio

from fastapi.testclient import TestClient

from app.main import app
from app.routers import admin
from app.services import graphiti_client

client = TestClient(app)

REEMBED = "/admin/reembed-graph"
CONFIRM = "I understand this overwrites all embeddings"


def test_wrong_confirm_is_rejected_before_any_work():
    """No confirm string -> 400, and we never touch the embedder or FalkorDB."""
    r = client.post(REEMBED, json={"client_slug": "pokagon", "confirm": "nope"})
    assert r.status_code == 400, r.text
    assert "Confirmation required" in r.json()["detail"]


def test_fail_loud_when_no_explicit_embedder(monkeypatch):
    """Right confirm but no Voyage embedder -> 400 refusal, never the OpenAI default."""
    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: None)
    r = client.post(REEMBED, json={"client_slug": "pokagon", "confirm": CONFIRM})
    assert r.status_code == 400, r.text
    assert "VOYAGE_API_KEY" in r.json()["detail"]


def test_chunk_helper_partitions_exactly():
    items = list(range(259))
    chunks = list(admin._chunk(items, 128))
    assert [len(c) for c in chunks] == [128, 128, 3]
    assert [x for c in chunks for x in c] == items
    assert list(admin._chunk([], 128)) == []


def test_voyage_embedder_path_produces_1024_dim_vectors():
    """Prove the configured embedder (the one /reembed will use) works end-to-end:
    _create_embedder() returns a Voyage client and create_batch yields 1024-dim
    vectors via graphiti-core's VoyageAIEmbedder. Requires VOYAGE_API_KEY in env."""
    emb = graphiti_client._create_embedder()
    assert emb is not None, "VOYAGE_API_KEY must be set for this proof"
    vecs = asyncio.get_event_loop().run_until_complete(
        emb.create_batch(["institutional knowledge capture", "cash recycler access review"])
    )
    assert len(vecs) == 2
    assert all(len(v) == 1024 for v in vecs), [len(v) for v in vecs]
    assert all(isinstance(x, float) for x in vecs[0])
    # Distinct inputs -> distinct vectors (sanity: not a constant/zero embedder).
    assert vecs[0] != vecs[1]


if __name__ == "__main__":
    test_wrong_confirm_is_rejected_before_any_work()
    print("PASS  wrong-confirm rejected")
    test_chunk_helper_partitions_exactly()
    print("PASS  chunk helper")
    test_voyage_embedder_path_produces_1024_dim_vectors()
    print("PASS  voyage embedder -> 1024-dim")
    # monkeypatch test needs pytest; run via pytest for full coverage.
    print("OK (run `pytest tests/test_reembed.py` for the monkeypatch gate too)")
