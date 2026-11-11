"""A question's vector, made once and kept.

The embedding provider does not give the same vector for the same text twice.
Measured on 2026-11-11: one question asked twelve times came back as up to
seven different vectors, about 1e-3 apart in cosine distance. A stored vector
is made once, so the records and facts never moved. The question did, on every
ask, and whatever sat close together traded places from one ask to the next.

So the first vector made for a question is kept in the graph database's own
key space, and every later ask of the same question, on the same graph, with
the same model, uses that one: in this process or another, before or after a
restart.

Nothing here can cost a search its answer. A store that cannot be reached, a
value that does not decode, a slow read: each means the provider is asked, as
it was before this existed, and the store is left alone for a minute.

What is kept is a hash and a vector. The key is a SHA-256 of the model, the
graph and the question, so no question text is stored and one tenant's key is
never another's.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import struct
import time
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

KEY_PREFIX = "sdi:question-vector:v1:"
INDEX_KEY = "sdi:question-vector:v1:index"
# The most vectors kept. Past it the oldest go first. At 1,024 numbers a vector
# is about 5.5 KB as stored, so the cap is about 110 MB at the very most.
MAX_STORED = 20_000
# A backstop, not the bound: a vector whose index entry was lost still goes.
KEEP_SECONDS = 365 * 24 * 3600
# The store has its own connection so that it never waits on the graph
# handle's long socket limit: a read or write here gives up in half a second,
# and is never tried a second time (the client library retries a failed
# connection with growing waits unless told not to: on a real engine that
# turned half a second into 4.7 s).
STORE_SOCKET_SECONDS = 0.5
# However the client library behaves, a search waits no longer than this for
# the store, once for the read and once for the write.
STORE_WAIT_SECONDS = 1.0
# After a failure the store is not tried again for this long.
STORE_REST_SECONDS = 60.0
# Connections the store may hold. A call that finds them all busy waits for one
# (up to the socket limit) instead of failing: a burst of questions is not a
# store in trouble, and must not send it to rest.
STORE_CONNECTIONS = 8
# Questions whose vector is never kept. A question asked only to find out
# whether search works end to end has to reach the provider every time, or a
# provider that has stopped answering would hide behind a kept vector. The
# platform's health monitor asks exactly this.
NEVER_KEPT = frozenset({"health probe"})
# Removes a value only if it is still the one that was read: two asks that both
# found the same unreadable value then race for the key like any two first
# asks, and the first writer wins.
_DROP_IF_UNCHANGED = (
    "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0"
)

_redis: Any = None
_resting_until = 0.0


def reset() -> None:
    """Forget the connection and any rest. For tests and for a reconnect."""
    global _redis, _resting_until
    _redis = None
    _resting_until = 0.0


def _client() -> Any:
    """The store's own connection to the graph database, made on first use."""
    global _redis
    if _redis is None:
        import redis  # noqa: PLC0415
        from redis.backoff import NoBackoff  # noqa: PLC0415
        from redis.retry import Retry  # noqa: PLC0415

        _redis = redis.Redis(
            connection_pool=redis.BlockingConnectionPool(
                max_connections=STORE_CONNECTIONS,
                timeout=STORE_SOCKET_SECONDS,
                host=settings.falkordb_host,
                port=settings.falkordb_port,
                password=settings.falkordb_password or None,
                socket_connect_timeout=STORE_SOCKET_SECONDS,
                socket_timeout=STORE_SOCKET_SECONDS,
                retry=Retry(NoBackoff(), 0),
                retry_on_timeout=False,
                decode_responses=True,
            )
        )
    return _redis


def _store() -> Any:
    """The store, or None when it must not be used for this ask.

    A probe process only ever reads the graph, so it keeps nothing. A store
    that has just failed rests, so a database in trouble is not asked again by
    every search that follows.
    """
    if settings.graphiti_acceptance_probe_mode:
        return None
    if time.monotonic() < _resting_until:
        return None
    return _client()


def _rest(error: BaseException) -> None:
    global _resting_until
    _resting_until = time.monotonic() + STORE_REST_SECONDS
    logger.warning(
        "[graphiti] question vector store failed error_type=%s; asking the provider each time for %d s",
        type(error).__name__,
        int(STORE_REST_SECONDS),
    )


async def _within(call: Any, *args: Any) -> Any:
    """One store call off the event loop, given up on after STORE_WAIT_SECONDS."""
    return await asyncio.wait_for(asyncio.to_thread(call, *args), STORE_WAIT_SECONDS)


def question_key(graph_name: str, text: str) -> str:
    """Where a question's vector is kept: a hash of the model, the graph and the text."""
    marker = f"{settings.embedding_model}:{int(settings.embedding_dim)}"
    digest = hashlib.sha256(f"{marker}\x00{graph_name}\x00{text}".encode()).hexdigest()
    return f"{KEY_PREFIX}{digest}"


def encode(vector: list[float]) -> str:
    """A vector as text: 32-bit numbers, which is what the index compares, in base64."""
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def decode(value: Any) -> list[float] | None:
    """The vector a stored value holds, or None if it is not one of this model's size."""
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    if not isinstance(value, str):
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError:
        return None
    size = int(settings.embedding_dim)
    if len(raw) != 4 * size:
        return None
    vector = list(struct.unpack(f"<{size}f", raw))
    return vector if all(math.isfinite(number) for number in vector) else None


def _storable(vector: Any) -> bool:
    """Whether a vector can be written as 32-bit numbers at all.

    Its size and whether every number is finite are decode()'s to judge, on
    the way back: what does not decode is used once and not kept. (A number
    that is not finite fails the comparison here as well.)
    """
    return isinstance(vector, list) and all(
        isinstance(number, (int, float)) and not isinstance(number, bool) and abs(number) < 3.0e38
        for number in vector
    )


def _write(store: Any, key: str, encoded: str, unreadable: Any) -> Any:
    """Keep a vector and return the value the key holds afterwards.

    The first writer wins, always: an ask that finds another got there first
    takes that one's vector, so two asks that raced end on the same vector. A
    value that was there but did not decode (`unreadable`) is removed first,
    and only if it is still the value that was read, so two asks that both
    found it race for the key like any two first asks. The index is what holds
    the store to its cap: the newest goes in, and past the cap the oldest come
    out.
    """
    if unreadable is not None:
        store.eval(_DROP_IF_UNCHANGED, 1, key, unreadable)
    for _attempt in range(2):
        if store.set(key, encoded, ex=KEEP_SECONDS, nx=True):
            break
        held = store.get(key)
        if held is not None:
            return held
        # Another ask's vector was there and is gone again (expired or trimmed) between the two calls: try once more.
    else:
        return None
    now = time.time()
    pipe = store.pipeline(transaction=False)
    pipe.zremrangebyscore(INDEX_KEY, 0, now - KEEP_SECONDS)
    pipe.zadd(INDEX_KEY, {key: now})
    pipe.zcard(INDEX_KEY)
    over = int(pipe.execute()[-1]) - MAX_STORED
    if over > 0:
        oldest = [name for name, _score in store.zpopmin(INDEX_KEY, over)]
        if oldest:
            store.delete(*oldest)
    return encoded


async def question_vector(embedder: Any, graph_name: str, query: str) -> list[float]:
    """The vector for a question: the one kept for it, or a new one that is then kept.

    The provider is asked exactly as before (one text, line breaks as spaces)
    and its errors are the caller's, as before. What comes back from here on a
    first ask is the vector as stored, so the first ask and every later one
    hand the index the same numbers.
    """
    text = query.replace("\n", " ")
    store = None if text.strip().lower() in NEVER_KEPT else _store()
    key = question_key(graph_name, text)
    found: Any = None
    if store is not None:
        try:
            found = await _within(store.get, key)
        except Exception as error:
            _rest(error)
            store = None
        else:
            kept = decode(found)
            if kept is not None:
                return kept

    vector = await embedder.create(input_data=[text])
    if store is None or not _storable(vector):
        return vector

    encoded = encode(vector)
    settled = decode(encoded)
    if settled is None:
        return vector
    try:
        held = await _within(_write, store, key, encoded, found)
    except Exception as error:
        _rest(error)
        return settled
    return decode(held) or settled
