"""Record vector reads take turns.

On a real engine (2026-11-11) two nearest-neighbour reads of the record index
running at the same moment came back wrong about 1 to 3 times in 100: the
nearest record repeated in every row. One at a time, never. The engine's own
lookup does it, so the service makes its record vector reads take turns. It is
the only client of the database and runs as one process, so that closes it.
"""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services import graphiti_client, indexed_falkor, projection_records, vector_turns

ROW = ["r1", "a statement", "PROCESS_STEP", "src", "pv-1", "eng-1", "wf-1", "Receiving", 0.1, 1]


class WatchedGraph:
    """Counts how many reads of each kind are running at the same moment."""

    def __init__(self, read_seconds=0.03, vector_rows=True):
        self.read_seconds = read_seconds
        self.vector_rows = vector_rows
        self.lock = threading.Lock()
        self.running = {"vector": 0, "keyword": 0}
        self.most = {"vector": 0, "keyword": 0}
        self.reads: list[str] = []
        self.hold: threading.Event | None = None
        self.started = threading.Event()

    def query(self, statement, params=None, **_):
        return SimpleNamespace(result_set=[])

    def ro_query(self, statement, params=None, timeout=None):
        if "RETURN w.uuid LIMIT 1" in statement:
            return SimpleNamespace(result_set=[["wf-1"]])
        leg = "vector" if "db.idx.vector.queryNodes" in statement else "keyword"
        count = "count(n) AS seen" in statement
        with self.lock:
            self.running[leg] += 1
            self.most[leg] = max(self.most[leg], self.running[leg])
            self.reads.append(leg + ("-count" if count else ""))
        try:
            if leg == "vector":
                self.started.set()
                if self.hold is not None:
                    self.hold.wait(timeout=5)
            time.sleep(self.read_seconds)
            if count:
                return SimpleNamespace(result_set=[[0]])
            if leg == "vector" and not self.vector_rows:
                return SimpleNamespace(result_set=[])
            return SimpleNamespace(result_set=[ROW])
        finally:
            with self.lock:
                self.running[leg] -= 1


class Embedder:
    async def create(self, input_data):
        return [0.5, 0.5, 0.5, 0.5]


@pytest.fixture
def wired(monkeypatch):
    def wire(graph):
        monkeypatch.setattr(settings, "embedding_dim", 4, raising=False)
        monkeypatch.setattr(settings, "embedding_model", "unit-embedder", raising=False)
        monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: SimpleNamespace(select_graph=lambda name: graph))
        monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: Embedder())
        return graph

    return wire


def search(question="who signs the count sheet"):
    return projection_records.search_records("unit", ["eng-1"], question, 1)


def test_searches_at_once_never_run_two_vector_reads_together_and_keyword_reads_do_not_wait(wired):
    graph = wired(WatchedGraph())

    async def six_at_once():
        return await asyncio.gather(*[search(f"question {n}") for n in range(6)])

    answers = asyncio.run(six_at_once())
    assert all([record["record_id"] for record in found] == ["r1"] for found in answers)
    assert graph.reads.count("vector") == 6 and graph.reads.count("keyword") == 6
    assert graph.most["vector"] == 1, "one record vector read at a time"
    assert graph.most["keyword"] >= 2, "keyword reads run side by side as before"


def test_the_vector_legs_count_read_takes_its_turn_too(wired):
    # Nothing survives the filter, so each vector leg asks how many candidates its window held.
    graph = wired(WatchedGraph(vector_rows=False))

    async def four_at_once():
        return await asyncio.gather(*[search(f"question {n}") for n in range(4)])

    asyncio.run(four_at_once())
    assert graph.reads.count("vector") == 4 and graph.reads.count("vector-count") == 4
    assert graph.most["vector"] == 1


def test_a_vector_read_that_cannot_get_its_turn_in_time_is_skipped_and_the_keyword_half_still_answers(wired, monkeypatch, caplog):
    graph = wired(WatchedGraph(read_seconds=0.0))
    monkeypatch.setattr(settings, "search_vector_timeout_ms", 40, raising=False)

    async def held_out():
        turn = vector_turns.turn(vector_turns.RECORDS)
        await turn.acquire()  # another read has the turn and does not give it back
        try:
            began = time.monotonic()
            found = await search()
            return found, time.monotonic() - began
        finally:
            turn.release()

    with caplog.at_level(logging.WARNING):
        found, waited = asyncio.run(held_out())
    assert [record["record_id"] for record in found] == ["r1"], "the keyword half answers"
    assert "vector" not in graph.reads, "no vector read ran beside the one holding the turn"
    assert 0.03 <= waited < 0.25, f"waited {waited:.3f} s for a 40 ms limit"
    (skipped,) = [r for r in caplog.records if "record vector leg skipped" in r.getMessage()]
    assert "error_type=TimeoutError" in skipped.getMessage()


def test_a_search_that_is_cancelled_keeps_its_turn_until_its_read_is_done(wired):
    graph = wired(WatchedGraph(read_seconds=0.0))
    graph.hold = threading.Event()

    async def cancel_then_ask_again():
        first = asyncio.ensure_future(search("first question"))
        await asyncio.to_thread(graph.started.wait, 2)  # the first vector read is now inside the engine
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.ensure_future(search("second question"))
        await asyncio.sleep(0.15)
        beside = graph.reads.count("vector")  # the second must not have started its vector read
        graph.hold.set()  # the engine finishes the first read
        found = await second
        return beside, found

    beside, found = asyncio.run(cancel_then_ask_again())
    assert beside == 1, "the cancelled search's read was still running, so the next one waited"
    assert [record["record_id"] for record in found] == ["r1"]
    assert graph.reads.count("vector") == 2 and graph.most["vector"] == 1


def test_a_vector_read_that_fails_gives_the_turn_back(wired):
    graph = wired(WatchedGraph(read_seconds=0.0))
    plain = graph.ro_query
    failed = []

    def failing_once(statement, params=None, timeout=None):
        if "db.idx.vector.queryNodes" in statement and not failed:
            failed.append(1)
            raise RuntimeError("the engine refused this read")
        return plain(statement, params, timeout)

    graph.ro_query = failing_once

    async def twice():
        first = await search("first question")
        second = await search("second question")
        return first, second, vector_turns.turn(vector_turns.RECORDS).locked()

    first, second, still_held = asyncio.run(twice())
    assert [record["record_id"] for record in first] == ["r1"], "the keyword half answered the first"
    assert [record["record_id"] for record in second] == ["r1"]
    assert graph.reads.count("vector") == 1, "and the second search got its vector read"
    assert still_held is False


def test_each_event_loop_has_its_own_turn_and_a_closed_loop_is_forgotten():
    async def turn():
        return vector_turns.turn(vector_turns.RECORDS), vector_turns.turn(vector_turns.RECORDS), vector_turns.turn(vector_turns.ENTITIES)

    vector_turns._turns.clear()
    one, again, entities = asyncio.run(turn())
    assert one is again, "one loop and index, one turn"
    assert entities is not one, "each index has its own turn"
    other, _, _ = asyncio.run(turn())
    assert other is not one, "a lock is never carried from one loop to another"
    assert len(vector_turns._turns) == 2, "the closed loop's turns are gone"


def test_the_wait_for_a_turn_comes_out_of_the_legs_own_time(wired, monkeypatch):
    # The leg has 2 x its read limit in all. A leg whose first read found nothing asks for its count
    # read with what is left on that clock, never a fresh limit.
    graph = wired(WatchedGraph(read_seconds=0.0, vector_rows=False))
    monkeypatch.setattr(settings, "search_vector_timeout_ms", 60, raising=False)
    waits: list[float] = []
    plain = vector_turns.take_turn

    async def watching(index, start, wait_seconds):
        assert index == vector_turns.RECORDS
        waits.append(wait_seconds)
        return await plain(index, start, wait_seconds)

    monkeypatch.setattr(vector_turns, "take_turn", watching)
    asyncio.run(search())
    assert len(waits) == 2, "the leg read and its count read"
    assert all(0 < wait <= 0.06 for wait in waits), waits

    # Nothing left on the clock and the turn held: the read gives up at once (not after its own
    # 400 ms limit) and is never started.
    async def with_the_turn_held():
        turn = vector_turns.turn(vector_turns.RECORDS)
        await turn.acquire()
        try:
            began = time.monotonic()
            with pytest.raises(TimeoutError):
                await projection_records._leg_read(graph, "vector", "CALL db.idx.vector.queryNodes()", {}, 400, time.monotonic() - 1)
            return time.monotonic() - began
        finally:
            turn.release()

    before = graph.reads.count("vector")
    gave_up_after = asyncio.run(with_the_turn_held())
    assert gave_up_after < 0.15, f"gave up after {gave_up_after:.3f} s with no time left"
    assert graph.reads.count("vector") == before

    # A little left on the clock: the wait is that little, not the read's whole limit.
    async def a_little_left():
        turn = vector_turns.turn(vector_turns.RECORDS)
        await turn.acquire()
        try:
            began = time.monotonic()
            with pytest.raises(TimeoutError):
                await projection_records._leg_read(graph, "vector", "CALL db.idx.vector.queryNodes()", {}, 2000, time.monotonic() + 0.05)
            return time.monotonic() - began
        finally:
            turn.release()

    waited = asyncio.run(a_little_left())
    assert 0.04 <= waited < 0.5, f"waited {waited:.3f} s with 50 ms left and a 2 s read limit"


def test_a_read_that_fails_after_its_caller_stopped_waiting_leaves_no_unread_error(wired):
    graph = wired(WatchedGraph(read_seconds=0.0))
    graph.hold = threading.Event()
    plain = graph.ro_query

    def failing(statement, params=None, timeout=None):
        if "db.idx.vector.queryNodes" in statement:
            graph.started.set()
            graph.hold.wait(timeout=5)
            raise RuntimeError("the engine failed this read")
        return plain(statement, params, timeout)

    graph.ro_query = failing
    unread: list[dict] = []

    async def cancel_mid_read():
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unread.append(context))
        first = asyncio.ensure_future(search())
        await asyncio.to_thread(graph.started.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        graph.hold.set()
        await asyncio.sleep(0.1)
        import gc

        gc.collect()
        await asyncio.sleep(0)
        return vector_turns.turn(vector_turns.RECORDS).locked()

    assert asyncio.run(cancel_mid_read()) is False, "the turn was given back"
    assert unread == [], unread


def test_a_start_that_cannot_begin_gives_the_turn_back():
    async def run():
        def broken():
            raise RuntimeError("could not start")

        with pytest.raises(RuntimeError, match="could not start"):
            await vector_turns.take_turn(vector_turns.RECORDS, broken, 1.0)
        return vector_turns.turn(vector_turns.RECORDS).locked()

    assert asyncio.run(run()) is False


@pytest.mark.parametrize("wait", [0, 0.0, -5])
def test_with_no_time_left_a_free_turn_is_still_taken_and_a_held_one_is_refused_at_once(wait):
    ran = []

    async def read():
        ran.append(1)
        return "rows"

    async def free():
        return await vector_turns.take_turn(vector_turns.RECORDS, read, wait)

    assert asyncio.run(free()) == "rows" and ran == [1], "nobody held the turn, so the read ran"

    async def held():
        turn = vector_turns.turn(vector_turns.RECORDS)
        await turn.acquire()
        try:
            began = time.monotonic()
            with pytest.raises(TimeoutError):
                await vector_turns.take_turn(vector_turns.RECORDS, read, wait)
            return time.monotonic() - began
        finally:
            turn.release()

    assert asyncio.run(held()) < 0.1 and ran == [1], "refused at once, and the read was not started"


# --------------------------------------------------------------------------
# The entity lookup used when facts are ingested reads a node index too
# --------------------------------------------------------------------------


class EntityExecutor:
    """Stands in for the async driver: counts lookups running at the same moment."""

    def __init__(self, seconds=0.02):
        self.seconds, self.running, self.most, self.calls = seconds, 0, 0, []

    async def execute_query(self, cypher, **params):
        if "db.idx.vector.queryNodes" not in cypher:
            return [], None, None
        self.calls.append(cypher)
        self.running += 1
        self.most = max(self.most, self.running)
        try:
            await asyncio.sleep(self.seconds)
            return [], None, None
        finally:
            self.running -= 1


def entity_search(executor, monkeypatch):
    async def ensured(*_args, **_kwargs):
        return None

    monkeypatch.setattr(indexed_falkor, "ensure_node_vector_index_via", ensured)
    monkeypatch.setattr(indexed_falkor, "_has_filters", lambda search_filter: False)
    monkeypatch.setattr(settings, "embedding_dim", 4, raising=False)
    operations = indexed_falkor.IndexedFalkorSearchOperations()
    return lambda: operations.node_similarity_search(executor, [0.1, 0.2, 0.3, 0.4], None, ["client_unit"], 5, 0.6)


def test_entity_lookups_take_turns_with_each_other(monkeypatch):
    executor = EntityExecutor()
    lookup = entity_search(executor, monkeypatch)

    async def together():
        await asyncio.gather(*[lookup() for _ in range(6)])

    asyncio.run(together())
    assert len(executor.calls) == 6 and executor.most == 1, "one entity lookup at a time"


def test_record_reads_and_entity_lookups_each_have_their_own_turn_and_never_wait_on_each_other(wired, monkeypatch):
    # A lookup that hangs holds the entity index's turn until it ends. A record search must not notice.
    graph = wired(WatchedGraph(read_seconds=0.0))
    executor = EntityExecutor()
    lookup = entity_search(executor, monkeypatch)
    monkeypatch.setattr(indexed_falkor, "NODE_VECTOR_TURN_WAIT_SECONDS", 0.03)
    scanned = []

    async def scan(self, *args, **kwargs):
        scanned.append(1)
        return []

    monkeypatch.setattr(indexed_falkor.FalkorSearchOperations, "node_similarity_search", scan)

    async def entity_turn_held():
        turn = vector_turns.turn(vector_turns.ENTITIES)
        await turn.acquire()
        try:
            found = await search()  # goes straight through: the record index's turn is its own
            await lookup()  # waits out its bound and falls back to the scan
            return found
        finally:
            turn.release()

    found = asyncio.run(entity_turn_held())
    assert [record["record_id"] for record in found] == ["r1"] and graph.reads.count("vector") == 1
    assert scanned == [1] and executor.calls == []

    async def record_turn_held():
        turn = vector_turns.turn(vector_turns.RECORDS)
        await turn.acquire()
        try:
            await lookup()  # goes straight through
        finally:
            turn.release()

    asyncio.run(record_turn_held())
    assert len(executor.calls) == 1 and scanned == [1]


def test_an_entity_lookup_that_cannot_get_its_turn_falls_back_to_the_scan(monkeypatch):
    executor = EntityExecutor()
    lookup = entity_search(executor, monkeypatch)
    monkeypatch.setattr(indexed_falkor, "NODE_VECTOR_TURN_WAIT_SECONDS", 0.03)
    scanned = []

    async def scan(self, *args, **kwargs):
        scanned.append(1)
        return []

    monkeypatch.setattr(indexed_falkor.FalkorSearchOperations, "node_similarity_search", scan)

    async def held_out():
        turn = vector_turns.turn(vector_turns.ENTITIES)
        await turn.acquire()
        try:
            return await lookup()
        finally:
            turn.release()

    assert asyncio.run(held_out()) == []
    assert scanned == [1] and executor.calls == [], "the scan answered; the index was not read beside the read holding the turn"


def test_an_entity_lookup_waits_two_seconds_for_its_turn_at_most():
    assert indexed_falkor.NODE_VECTOR_TURN_WAIT_SECONDS == 2.0


def test_a_free_turn_with_a_waiter_ahead_is_given_up_on_within_a_millisecond():
    # The turn has just been released and a waiter ahead has been woken but not yet run: the lock reads as
    # free. A caller with no time left takes its place in the queue for one millisecond and no longer.
    assert vector_turns._FREE_TURN_SECONDS == 0.001

    async def run():
        turn = vector_turns.turn(vector_turns.RECORDS)
        await turn.acquire()
        started = asyncio.Event()

        async def waiter():
            started.set()
            await turn.acquire()
            await asyncio.sleep(0.2)  # the waiter's read
            turn.release()

        ahead = asyncio.ensure_future(waiter())
        await started.wait()
        await asyncio.sleep(0)
        turn.release()  # the waiter is woken; until it runs the lock is unlocked
        assert turn.locked() is False

        async def read():
            return "rows"

        began = time.monotonic()
        with pytest.raises(TimeoutError):
            await vector_turns.take_turn(vector_turns.RECORDS, read, 0)
        waited = time.monotonic() - began
        await ahead
        return waited

    assert asyncio.run(run()) < 0.1
