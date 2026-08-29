"""The socket timeout must be a safety net, never the deciding limit.

Every ingest through 2026-08-28/29 failed at ~121s against a 120s socket
timeout, and when the value moved the failure time moved with it. That is the
signature of a bound set BELOW the work it is bounding: the read was cut off
mid-flight and reported as "Timeout reading from falkordb:6379", which reads
like a broken datastore and is not one.

graphiti-core dedups extracted entities by computing cosine distance inline in
Cypher, with no vector index on entity embeddings -- an O(N) scan per entity,
25-50 entities per episode, against every embedded node in the graph. It is
legitimately slow and grows with the graph. Our RELATES_TO.fact_embedding HNSW
index accelerates SEARCH and does not touch that path.
"""

from app.config import settings

# Mirrors INGEST_POLL_BUDGET_MS in sdi-engage-backend/src/lib/graphiti.ts.
INGEST_POLL_BUDGET_SECONDS = 3600


def test_the_socket_outlives_a_slow_dedup_rather_than_cutting_it_off():
    """A real episode must be able to finish inside one socket read."""
    assert settings.falkordb_socket_timeout_seconds >= 600, (
        "below this the socket, not the work, decides the outcome -- the exact "
        "failure mode that read as a FalkorDB outage for two days"
    )


def test_the_socket_still_gives_up_before_the_poller_does():
    """Bounded is the point: 'slow' must never become 'hangs forever'.

    It must also expire BEFORE the ingest poll budget, so the job reports a
    real outcome instead of the socket dying first and losing the reason.
    """
    assert settings.falkordb_socket_timeout_seconds < INGEST_POLL_BUDGET_SECONDS, (
        "the poller must outlive the socket, or the outcome is lost"
    )


def test_it_is_bounded_at_all():
    """None means a hung coroutine with no exception and no completion."""
    assert settings.falkordb_socket_timeout_seconds is not None
    assert settings.falkordb_socket_timeout_seconds > 0
