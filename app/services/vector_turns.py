"""One nearest-neighbour read of a node index at a time, in this process.

On a real engine (2026-11-11) two `db.idx.vector.queryNodes` reads running at
the same moment came back wrong about 1 to 3 times in 100: the nearest node
repeated in every row. One at a time, never. The bare procedure shows it with
nothing after it in the statement, so no wording avoids it, and it showed on
both node indexes this service reads (records and entities). The lookup over
relationships that fact search uses did not show it in 400 tries, and neither
did keyword reads in 300; those do not wait.

This service is the only client of the database and runs as one process, so
taking turns here closes it. A second process, or a second client, would need
a lock they share.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")

_turns: dict[int, tuple[Any, asyncio.Lock]] = {}


def turn() -> asyncio.Lock:
    """The turn for the event loop this is running on (one loop, one turn)."""
    loop = asyncio.get_running_loop()
    held = _turns.get(id(loop))
    if held is None or held[0] is not loop:
        for key in [key for key, (other, _lock) in _turns.items() if other.is_closed()]:
            del _turns[key]
        held = _turns[id(loop)] = (loop, asyncio.Lock())
    return held[1]


def _end(lock: asyncio.Lock, reading: asyncio.Future) -> None:
    lock.release()
    if not reading.cancelled():
        reading.exception()  # read here, so a caller that stopped waiting leaves no unread error behind


async def take_turn(start: Callable[[], Awaitable[T]], wait_seconds: float) -> T:
    """Run one node-vector read when no other is running.

    `start` begins the read and is only called once the turn is held. Waiting
    for the turn is bounded: past `wait_seconds` this raises TimeoutError and
    the read is never started. The turn ends when the read does, not when the
    caller stops waiting, so a request that is cancelled part-way never lets
    the next read start beside one the engine is still running.
    """
    lock = turn()
    await asyncio.wait_for(lock.acquire(), max(0.0, wait_seconds))
    try:
        reading = asyncio.ensure_future(start())
    except BaseException:
        lock.release()
        raise
    reading.add_done_callback(lambda done: _end(lock, done))
    return await asyncio.shield(reading)
