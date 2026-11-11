"""One nearest-neighbour read of a node index at a time, in this process.

On a real engine (2026-11-11) two `db.idx.vector.queryNodes` reads running at
the same moment came back wrong about 1 to 3 times in 100: the nearest node
repeated in every row. One at a time, never. The bare procedure shows it with
nothing after it in the statement, so no wording avoids it, and it showed on
both node indexes this service reads (records and entities). The lookup over
relationships that fact search uses did not show it in 400 tries, and neither
did keyword reads in 300; those do not wait.

Each index has its own turn. Reads of one index were never seen to disturb
reads of another: with record reads taking turns and entity reads not, 2,400
record answers of 2,400 were right while entity answers beside them went
wrong, and the other way about. So a record search waits only for record
reads, and a lookup made while facts are ingested waits only for others of its
kind. That matters when one read hangs: it holds its own index's turn until it
ends, and must not take the other index's readers down with it.

This service is the only client of the database and runs as one process, so
taking turns here closes it. A second process, or a second client, would need
a lock they share.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")

# The two node indexes this service reads by nearest neighbour.
RECORDS = "records"
ENTITIES = "entities"

_turns: dict[tuple[int, str], tuple[Any, asyncio.Lock]] = {}


def turn(index: str) -> asyncio.Lock:
    """The turn for one index, on the event loop this is running on (one loop and index, one turn)."""
    loop = asyncio.get_running_loop()
    held = _turns.get((id(loop), index))
    if held is None or held[0] is not loop:
        for key in [key for key, (other, _lock) in _turns.items() if other.is_closed()]:
            del _turns[key]
        held = _turns[(id(loop), index)] = (loop, asyncio.Lock())
    return held[1]


def _end(lock: asyncio.Lock, reading: asyncio.Future) -> None:
    lock.release()
    if not reading.cancelled():
        reading.exception()  # read here, so a caller that stopped waiting leaves no unread error behind


# With no time left to wait, a turn nobody holds is still taken. This is how long that may take.
_FREE_TURN_SECONDS = 0.001


async def take_turn(index: str, start: Callable[[], Awaitable[T]], wait_seconds: float) -> T:
    """Run one nearest-neighbour read of `index` when no other read of it is running.

    `start` begins the read and is only called once the turn is held. Waiting
    for the turn is bounded: past `wait_seconds` this raises TimeoutError and
    the read is never started. With no time left to wait (zero or less) a turn
    that is free is still taken, and one that is held is refused at once. The
    turn ends when the read does, not when the caller stops waiting, so a
    request that is cancelled part-way never lets the next read start beside
    one the engine is still running.
    """
    lock = turn(index)
    if wait_seconds <= 0:
        if lock.locked():
            raise TimeoutError(f"no time left to wait for a turn at the {index} index")
        wait_seconds = _FREE_TURN_SECONDS
    await asyncio.wait_for(lock.acquire(), wait_seconds)
    try:
        reading = asyncio.ensure_future(start())
    except BaseException:
        lock.release()
        raise
    reading.add_done_callback(lambda done: _end(lock, done))
    return await asyncio.shield(reading)
