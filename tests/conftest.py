"""Shared test setup.

`graphiti_client` holds ONE FalkorDB handle per process — that is the whole
point of the change that introduced it, since building a pool per request
exhausted FalkorDB's client limit in production on 2026-08-27.

A process-wide handle is shared state, and tests inject their own fake
FalkorDB. Without this reset the first test to touch it pins its fake for the
rest of the session and later tests silently exercise the wrong object.
"""

import pytest

from app.routers import projection
from app.services import graphiti_client


@pytest.fixture(autouse=True)
def _reset_shared_falkordb():
    graphiti_client.reset_falkor_db()
    projection._reset_falkor_db()
    yield
    graphiti_client.reset_falkor_db()
    projection._reset_falkor_db()
