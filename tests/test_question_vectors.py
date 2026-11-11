"""A question's vector is made once and kept.

The embedding provider does not give the same vector for the same text twice
(measured 2026-11-11: up to seven different vectors in twelve asks of one
question, about 1e-3 apart in cosine distance), so a search could rank what it
found differently from one ask to the next. These tests hold the store to what
it promises: the same question gets the same vector however the provider
answers, a store in trouble never costs a search its answer, and what is kept
is bounded and carries no question text.
"""

import asyncio
import logging
import math
import struct
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services import graphiti_client, projection_records, question_vectors

# Taken at import, before the shared fixture swaps the store off for every test.
REAL_CLIENT = question_vectors._client
QUESTION = "who signs the weekly count sheet"
GRAPH = "client_unit"


class JitterEmbedder:
    """A provider that never says the same thing twice."""

    def __init__(self, dim=4):
        self.dim = dim
        self.asked: list[list[str]] = []

    async def create(self, input_data):
        self.asked.append(list(input_data))
        nudge = len(self.asked) * 1e-3
        return [0.1 + nudge + place * 0.01 for place in range(self.dim)]


class FakePipe:
    def __init__(self, store):
        self.store, self.steps = store, []

    def zremrangebyscore(self, name, low, high):
        self.steps.append(("zremrangebyscore", name, low, high))

    def zadd(self, name, mapping):
        self.steps.append(("zadd", name, dict(mapping)))

    def zcard(self, name):
        self.steps.append(("zcard", name))

    def execute(self):
        out = []
        for step in self.steps:
            self.store.calls.append(step)
            if "pipeline" in self.store.fail:
                raise TimeoutError("timed out")
            if step[0] == "zremrangebyscore":
                gone = [key for key, score in self.store.index.items() if step[2] <= score <= step[3]]
                for key in gone:
                    del self.store.index[key]
                out.append(len(gone))
            elif step[0] == "zadd":
                self.store.index.update(step[2])
                out.append(len(step[2]))
            else:
                out.append(len(self.store.index))
        return out


class FakeStore:
    """The few commands the store uses, with the meaning the database gives them."""

    def __init__(self):
        self.values: dict[str, str] = {}
        self.ttl: dict[str, int | None] = {}
        self.index: dict[str, float] = {}
        self.calls: list[tuple] = []
        self.fail: set[str] = set()

    def get(self, key):
        self.calls.append(("get", key))
        if "get" in self.fail:
            raise ConnectionError("connection refused")
        return self.values.get(key)

    def set(self, key, value, ex=None, nx=False):
        self.calls.append(("set", key, ex, nx))
        if "set" in self.fail:
            raise TimeoutError("timed out")
        if nx and key in self.values:
            return None
        self.values[key] = value
        self.ttl[key] = ex
        return True

    def pipeline(self, transaction=True):
        self.calls.append(("pipeline", transaction))
        return FakePipe(self)

    def zpopmin(self, name, count):
        self.calls.append(("zpopmin", name, count))
        oldest = sorted(self.index.items(), key=lambda item: item[1])[:count]
        for key, _score in oldest:
            del self.index[key]
        return oldest

    def delete(self, *keys):
        self.calls.append(("delete", *keys))
        for key in keys:
            self.values.pop(key, None)
        return len(keys)


@pytest.fixture
def kept(monkeypatch):
    """A store in place of the real one, with a small model so vectors are readable."""
    monkeypatch.setattr(settings, "embedding_dim", 4, raising=False)
    monkeypatch.setattr(settings, "embedding_model", "unit-embedder", raising=False)
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", False, raising=False)
    store = FakeStore()
    monkeypatch.setattr(question_vectors, "_client", lambda: store)
    return store


def ask(embedder, question=QUESTION, graph=GRAPH):
    return asyncio.run(question_vectors.question_vector(embedder, graph, question))


def as_stored(vector):
    return list(struct.unpack(f"<{len(vector)}f", struct.pack(f"<{len(vector)}f", *vector)))


# --------------------------------------------------------------------------
# The promise
# --------------------------------------------------------------------------


def test_the_same_question_gets_the_same_vector_however_the_provider_answers(kept):
    embedder = JitterEmbedder()
    first, second, third = ask(embedder), ask(embedder), ask(embedder)
    assert first == second == third
    assert len(embedder.asked) == 1, "the provider is asked once; after that the kept vector answers"
    assert embedder.asked[0] == [QUESTION]


def test_without_the_store_the_same_question_gets_a_different_vector_each_time(kept, monkeypatch):
    # What the store is for: this is the behaviour before it existed.
    monkeypatch.setattr(question_vectors, "_client", lambda: None)
    embedder = JitterEmbedder()
    assert ask(embedder) != ask(embedder)
    assert len(embedder.asked) == 2 and kept.calls == []


def test_the_first_ask_hands_back_the_vector_as_stored_so_it_matches_every_later_ask_bit_for_bit(kept):
    embedder = JitterEmbedder()
    made = [0.1 + 1e-3 + place * 0.01 for place in range(4)]
    first = ask(embedder)
    assert first == as_stored(made), "32-bit numbers, which is what the index compares"
    assert first != made, "the provider's 64-bit numbers are not what is handed on"
    assert ask(embedder) == first
    (value,) = kept.values.values()
    assert question_vectors.decode(value) == first


def test_line_breaks_are_spaces_as_they_were_before_and_both_spellings_are_one_question(kept):
    embedder = JitterEmbedder()
    first = ask(embedder, "who signs\nthe weekly count sheet")
    assert embedder.asked == [["who signs the weekly count sheet"]]
    assert ask(embedder, "who signs the weekly count sheet") == first
    assert len(embedder.asked) == 1


def test_the_key_is_a_hash_of_model_graph_and_question_and_holds_no_question_text(kept, monkeypatch):
    key = question_vectors.question_key(GRAPH, QUESTION)
    assert key.startswith(question_vectors.KEY_PREFIX) and len(key) == len(question_vectors.KEY_PREFIX) + 64
    for word in QUESTION.split() + [GRAPH]:
        assert word not in key
    assert question_vectors.question_key("client_other", QUESTION) != key, "one tenant's key is never another's"
    assert question_vectors.question_key(GRAPH, QUESTION + "?") != key
    monkeypatch.setattr(settings, "embedding_model", "another-embedder", raising=False)
    assert question_vectors.question_key(GRAPH, QUESTION) != key, "a new model never reads the old model's vectors"
    monkeypatch.setattr(settings, "embedding_model", "unit-embedder", raising=False)
    monkeypatch.setattr(settings, "embedding_dim", 8, raising=False)
    assert question_vectors.question_key(GRAPH, QUESTION) != key, "nor does a new size"


def test_each_graph_and_each_question_keeps_its_own_vector(kept):
    embedder = JitterEmbedder()
    here, there, other = ask(embedder), ask(embedder, graph="client_other"), ask(embedder, question="a different question")
    assert here != there and here != other
    assert len(kept.values) == 3
    assert ask(embedder) == here and ask(embedder, graph="client_other") == there
    assert len(embedder.asked) == 3


def test_two_asks_that_raced_end_on_the_same_vector(kept):
    # Another ask stored its vector between this one's read and its write.
    others = question_vectors.encode([0.9, 0.8, 0.7, 0.6])
    key = question_vectors.question_key(GRAPH, QUESTION)
    plain_get = kept.get
    seen = {"reads": 0}

    def get(name):
        seen["reads"] += 1
        if seen["reads"] == 1:
            plain_get(name)
            kept.values[key] = others  # the other ask lands now
            return None
        return plain_get(name)

    kept.get = get
    embedder = JitterEmbedder()
    assert ask(embedder) == question_vectors.decode(others), "the first writer's vector is the question's"
    assert kept.values[key] == others, "and it is not written over"
    assert ("set", key, question_vectors.KEEP_SECONDS, True) in kept.calls


# --------------------------------------------------------------------------
# A store in trouble never costs a search its answer
# --------------------------------------------------------------------------


def test_a_store_that_cannot_be_read_asks_the_provider_and_is_left_alone_for_a_minute(kept, monkeypatch, caplog):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(question_vectors.time, "monotonic", lambda: clock.now)
    kept.fail.add("get")
    embedder = JitterEmbedder()
    with caplog.at_level(logging.WARNING):
        first = ask(embedder)
    assert first == [0.1 + 1e-3 + place * 0.01 for place in range(4)], "the provider's own answer, as before"
    assert [call[0] for call in kept.calls] == ["get"], "nothing is written to a store that just failed"
    (warning,) = [r for r in caplog.records if "question vector store failed" in r.getMessage()]
    assert "error_type=ConnectionError" in warning.getMessage()
    assert QUESTION not in caplog.text and GRAPH not in caplog.text

    # While it rests the store is not asked at all, and nothing more is logged.
    clock.now += question_vectors.STORE_REST_SECONDS - 0.01
    with caplog.at_level(logging.WARNING):
        ask(embedder)
    assert [call[0] for call in kept.calls] == ["get"]
    assert len([r for r in caplog.records if "question vector store failed" in r.getMessage()]) == 1

    # After the rest it is tried again, and from then on the question is kept.
    clock.now += 0.02
    kept.fail.clear()
    third = ask(embedder)
    assert ask(embedder) == third and len(embedder.asked) == 3


@pytest.mark.parametrize("broken", ["set", "pipeline"])
def test_a_store_that_cannot_be_written_still_answers_and_rests(kept, caplog, broken):
    kept.fail.add(broken)
    embedder = JitterEmbedder()
    with caplog.at_level(logging.WARNING):
        answer = ask(embedder)
    assert answer == as_stored([0.1 + 1e-3 + place * 0.01 for place in range(4)])
    assert len([r for r in caplog.records if "question vector store failed" in r.getMessage()]) == 1
    before = len(kept.calls)
    ask(embedder)
    assert len(kept.calls) == before, "resting"


@pytest.mark.parametrize("held", ["not base64 !!", "", "AAAA", question_vectors.encode([float("nan"), 0.1, 0.2, 0.3]), b"\xff\xfe"])
def test_a_value_that_is_not_this_models_vector_is_passed_over_and_written_over(kept, held):
    key = question_vectors.question_key(GRAPH, QUESTION)
    kept.values[key] = held
    embedder = JitterEmbedder()
    answer = ask(embedder)
    assert len(embedder.asked) == 1, "the provider is asked"
    assert ("set", key, question_vectors.KEEP_SECONDS, False) in kept.calls, "written over, not left in the way"
    assert question_vectors.decode(kept.values[key]) == answer
    assert ask(embedder) == answer and len(embedder.asked) == 1


@pytest.mark.parametrize(
    "made",
    [
        [0.1, 0.2, 0.3],  # too short
        [0.1, 0.2, 0.3, 0.4, 0.5],  # too long
        [0.1, float("nan"), 0.3, 0.4],
        [0.1, float("inf"), 0.3, 0.4],
        [0.1, 1e39, 0.3, 0.4],  # does not fit a 32-bit number
        [0.1, "0.2", 0.3, None],
        [0.1, "0.2", 0.3, 0.4],  # a number written as text
        [0.1, True, 0.3, 0.4],
        (0.1, 0.2, 0.3, 0.4),  # not a list
        None,
    ],
)
def test_only_a_whole_vector_of_this_models_size_is_kept(kept, made):
    class Odd:
        async def create(self, input_data):
            return made

    answer = ask(Odd())
    assert answer is made or answer == made or (isinstance(made, list) and all(a is b or a == b or (isinstance(a, float) and math.isnan(a)) for a, b in zip(answer, made)))
    assert kept.values == {} and [call[0] for call in kept.calls] == ["get"]


def test_the_providers_failure_is_the_callers_as_before_and_nothing_is_kept(kept):
    class Down:
        async def create(self, input_data):
            raise RuntimeError("the provider is away")

    with pytest.raises(RuntimeError, match="the provider is away"):
        ask(Down())
    assert kept.values == {} and kept.index == {}


def test_a_probe_process_keeps_nothing_and_never_opens_the_store(kept, monkeypatch):
    monkeypatch.setattr(settings, "graphiti_acceptance_probe_mode", True, raising=False)
    opened = []
    monkeypatch.setattr(question_vectors, "_client", lambda: opened.append(1) or kept)
    embedder = JitterEmbedder()
    assert ask(embedder) != ask(embedder), "a probe asks the provider every time"
    assert opened == [] and kept.calls == []


# --------------------------------------------------------------------------
# What is kept is bounded
# --------------------------------------------------------------------------


def test_past_the_cap_the_oldest_vectors_go_first(kept, monkeypatch):
    monkeypatch.setattr(question_vectors, "MAX_STORED", 3)
    clock = SimpleNamespace(now=5_000_000.0)
    monkeypatch.setattr(question_vectors.time, "time", lambda: clock.now)
    embedder = JitterEmbedder()
    keys = []
    for number in range(5):
        clock.now += 1
        ask(embedder, question=f"question {number}")
        keys.append(question_vectors.question_key(GRAPH, f"question {number}"))
    assert set(kept.values) == set(keys[2:]) == set(kept.index), "the two oldest are gone, value and index entry both"
    asked = len(embedder.asked)
    ask(embedder, question="question 4")
    assert len(embedder.asked) == asked, "a kept question is still kept"
    ask(embedder, question="question 0")
    assert len(embedder.asked) == asked + 1, "a dropped one is made again"


def test_a_vector_is_kept_for_a_year_and_an_index_entry_older_than_that_is_cleared(kept, monkeypatch):
    clock = SimpleNamespace(now=5_000_000_000.0)
    monkeypatch.setattr(question_vectors.time, "time", lambda: clock.now)
    kept.index["sdi:question-vector:v1:gone-long-ago"] = clock.now - question_vectors.KEEP_SECONDS - 1
    kept.index["sdi:question-vector:v1:still-good"] = clock.now - question_vectors.KEEP_SECONDS + 60
    ask(JitterEmbedder())
    key = question_vectors.question_key(GRAPH, QUESTION)
    assert kept.ttl[key] == question_vectors.KEEP_SECONDS == 365 * 24 * 3600
    assert set(kept.index) == {"sdi:question-vector:v1:still-good", key}
    assert kept.index[key] == clock.now


def test_the_bounds_are_the_ones_the_design_was_sized_for():
    assert question_vectors.MAX_STORED == 20_000
    assert question_vectors.STORE_SOCKET_SECONDS == 0.5
    assert question_vectors.STORE_REST_SECONDS == 60.0
    assert question_vectors.INDEX_KEY.startswith(question_vectors.KEY_PREFIX)
    # 1,024 numbers as stored: the size the cap's memory figure rests on.
    assert len(question_vectors.encode([0.0] * 1024)) == 5464


def test_the_stores_own_connection_gives_up_in_half_a_second_and_is_made_once(monkeypatch):
    import redis

    built = []

    def fake_redis(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(kind="store")

    monkeypatch.setattr(redis, "Redis", fake_redis)
    monkeypatch.setattr(question_vectors, "_client", REAL_CLIENT)
    monkeypatch.setattr(settings, "falkordb_host", "graph.internal", raising=False)
    monkeypatch.setattr(settings, "falkordb_port", 6380, raising=False)
    monkeypatch.setattr(settings, "falkordb_password", "", raising=False)
    question_vectors.reset()
    first, second = question_vectors._client(), question_vectors._client()
    assert first is second and len(built) == 1
    assert built[0] == {
        "host": "graph.internal",
        "port": 6380,
        "password": None,
        "socket_connect_timeout": 0.5,
        "socket_timeout": 0.5,
        "decode_responses": True,
        "max_connections": 8,
    }
    question_vectors.reset()
    question_vectors._client()
    assert len(built) == 2, "reset() forgets the connection"


# --------------------------------------------------------------------------
# Both searches use it
# --------------------------------------------------------------------------


class RecordGraph:
    """Enough of a graph for one record search: it says records are held and keeps what each leg was sent."""

    def __init__(self):
        self.sent: list[dict] = []

    def query(self, statement, params=None, **_):
        return SimpleNamespace(result_set=[])

    def ro_query(self, statement, params=None, timeout=None):
        if "RETURN w.uuid LIMIT 1" in statement:
            return SimpleNamespace(result_set=[["wf-1"]])
        if "db.idx.vector.queryNodes" in statement:
            self.sent.append(dict(params or {}))
        row = ["r1", "a statement", "PROCESS_STEP", "src", "pv-1", "eng-1", "wf-1", "Receiving", 0.1, 1]
        return SimpleNamespace(result_set=[row])


def test_record_search_sends_the_index_the_same_vector_for_the_same_question(kept, monkeypatch):
    graph, embedder = RecordGraph(), JitterEmbedder()
    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: SimpleNamespace(select_graph=lambda name: graph))
    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: embedder)
    for _ in range(3):
        found = asyncio.run(projection_records.search_records("unit", ["eng-1"], QUESTION, 1))
        assert [record["record_id"] for record in found] == ["r1"]
    assert len(graph.sent) == 3 and graph.sent[0]["q"] == graph.sent[1]["q"] == graph.sent[2]["q"]
    assert len(embedder.asked) == 1
    (key,) = kept.values
    assert key == question_vectors.question_key(graphiti_client._graph_name_for_client("unit"), QUESTION)


def test_fact_search_sends_the_index_the_same_vector_for_the_same_question(kept, monkeypatch):
    sent: list[dict] = []
    row = ["e1", "One team owns the monthly close.", "OWNS", "s", "One Team", "t", "Monthly Close", "['ep']", None, None, None]

    class FactGraph:
        def query(self, q, params=None, timeout=None):
            if "db.idx.vector" in q:
                sent.append(dict(params or {}))
            return SimpleNamespace(result_set=[row])

        ro_query = query

    graph, embedder = FactGraph(), JitterEmbedder()
    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: SimpleNamespace(select_graph=lambda name: graph))
    monkeypatch.setattr(graphiti_client, "_ensure_edge_vector_index", lambda g, n: None)
    monkeypatch.setattr(graphiti_client, "_create_embedder", lambda: embedder)
    for _ in range(3):
        edges = asyncio.run(graphiti_client._search_fast("unit", QUESTION, 5))
        assert len(edges) == 1
    assert len(sent) == 3 and sent[0]["q"] == sent[1]["q"] == sent[2]["q"]
    assert len(embedder.asked) == 1
    assert set(kept.values) == {question_vectors.question_key(graphiti_client._graph_name_for_client("unit"), QUESTION)}


def test_the_search_timing_route_still_asks_the_provider_itself():
    # The profile route measures how long the provider takes, so it must not read a kept vector.
    import inspect

    source = inspect.getsource(graphiti_client)
    assert source.count("question_vectors.question_vector(") == 1, "the fact search, and only it, in this module"
    assert source.count('embedder.create(input_data=[query.replace("\\n", " ")])') == 1, "the timing route"
