"""POST /law/sections and POST /law/sources: exact reads of law sections by identity, with no search.

These run without an engine: a fake graph records every statement and its
parameters and hands back rows a test chooses. They pin the route's contract
(auth, scope, the cap, trimming, when it is open, how rows become answers, and
that request values never reach the Cypher text). What only a real engine can
show (text byte for byte through the driver, hostile values in the engine's own
matching, scope in its own matching) is in tests/test_law_sections_engine.py.
"""

import json
import time
import uuid as uuidlib

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.auth import build_signature
from app.config import settings
from app.main import app
from app.models.law import MAX_LAW_SECTION_ITEMS
from app.services import graphiti_client, law_sections

CLIENT = "unit-tenant"
GRAPH = "client_unit_tenant"
SEARCH_SECRET = "search-secret-that-is-at-least-32-characters"
INGEST_SECRET = "ingest-secret-that-is-at-least-32-characters"
ADMIN_SECRET = "admin-secret-that-is-at-least-32-charactersx"


class FakeResult:
    def __init__(self, rows):
        self.result_set = rows


class FakeGraph:
    def __init__(self, fake, name):
        self.fake, self.name = fake, name

    def ro_query(self, statement, params=None, timeout=None):
        self.fake.reads.append((self.name, statement, params, timeout))
        if self.fake.error is not None:
            raise self.fake.error
        return FakeResult(self.fake.rows)

    def query(self, statement, params=None, timeout=None):
        # The route reads with ro_query only. (The service's startup census also reaches this fake; it is not the route.)
        self.fake.other.append(statement)
        return FakeResult([])


class FakeDB:
    def __init__(self):
        self.reads: list = []
        self.other: list = []
        self.rows: list = []
        self.error: Exception | None = None
        self.graphs = [GRAPH]

    def select_graph(self, name):
        return FakeGraph(self, name)

    def list_graphs(self):
        return list(self.graphs)


@pytest.fixture
def fake(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(graphiti_client, "get_falkor_db", lambda: db)
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required", raising=False)
    monkeypatch.setattr(settings, "graphiti_search_secret", SEARCH_SECRET, raising=False)
    monkeypatch.setattr(settings, "graphiti_ingest_secret", INGEST_SECRET, raising=False)
    monkeypatch.setattr(settings, "graphiti_admin_secret", ADMIN_SECRET, raising=False)
    monkeypatch.setattr(settings, "projection_v2_allow_law", True, raising=False)
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", False, raising=False)
    consumed: set[str] = set()

    async def consume(scope, nonce):
        if nonce in consumed:
            return False
        consumed.add(nonce)
        return True

    monkeypatch.setattr(auth, "_consume_nonce", consume)
    return db


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def post(client, path, body, *, scope="search", secret=SEARCH_SECRET, header_client=None):
    raw = json.dumps(body).encode("utf-8")
    claimed = header_client if header_client is not None else body.get("client_slug", CLIENT)
    timestamp, nonce = str(int(time.time())), uuidlib.uuid4().hex
    signature = build_signature(
        secret=secret, timestamp=timestamp, nonce=nonce, method="POST", path=path, scope=scope,
        client_slug=claimed, body=raw,
    )
    return client.post(
        path,
        content=raw,
        headers={
            "content-type": "application/json",
            auth.AUTH_TIMESTAMP_HEADER: timestamp,
            auth.AUTH_SCOPE_HEADER: scope,
            auth.AUTH_CLIENT_HEADER: claimed,
            auth.AUTH_NONCE_HEADER: nonce,
            auth.AUTH_SIGNATURE_HEADER: signature,
        },
    )


def ask(items, **extra):
    return {"client_slug": CLIENT, "engagement_id": "eng-1", "items": items, **extra}


def row(item, *, engagement="eng-1", record="law-1@v1:1.01:1", part=1, parts=1, text="Section 1.01 Example heading.",
        heading="Example heading", version="v1", current="v1"):
    return [item, engagement, "Example Rules", version, record, heading, "", "Example Rules", part, parts, text,
            f"src-{record}", "ab" * 32, current]


# -- the contract --------------------------------------------------------


def test_a_section_comes_back_exactly_as_stored_with_its_identity(fake, client):
    fake.rows = [row(0, text='Section 1.01 Example heading. "A" \\ b\nc')]
    response = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01"}]))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["contract_version"] == "graphiti_law_sections_v1" and body["graph_name"] == GRAPH
    assert body["sections"] == [{
        "law": "law-1", "section": "1.01", "version": None, "status": "found", "engagement_id": "eng-1",
        "law_title": "Example Rules", "heading": "Example heading", "chapter": "", "version_id": "v1",
        "current_version_id": "v1", "in_force": True, "parts_expected": 1,
        "parts": [{"record_id": "law-1@v1:1.01:1", "part": 1, "text": 'Section 1.01 Example heading. "A" \\ b\nc',
                   "source_id": "src-law-1@v1:1.01:1", "content_hash": "ab" * 32}],
    }]


def test_one_answer_per_item_in_the_order_asked_with_not_found_where_nothing_is_stored(fake, client):
    fake.rows = [row(2, record="r-2"), row(0, record="r-0")]
    items = [{"law": "law-1", "section": "1.01"}, {"law": "law-1", "section": "9.99"}, {"law": "law-2", "section": "1.01"}]
    body = post(client, "/law/sections", ask(items)).json()
    assert [(e["law"], e["section"], e["status"]) for e in body["sections"]] == [
        ("law-1", "1.01", "found"), ("law-1", "9.99", "not_found"), ("law-2", "1.01", "found"),
    ]
    assert body["sections"][1] == {
        "law": "law-1", "section": "9.99", "version": None, "status": "not_found", "engagement_id": "",
        "law_title": "", "heading": "", "chapter": "", "version_id": "", "current_version_id": "", "in_force": False,
        "parts_expected": None, "parts": [],
    }


def test_parts_are_ordered_and_a_gap_a_repeat_or_a_disagreement_is_incomplete_never_whole(fake, client):
    def answer(rows):
        fake.rows = rows
        return post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01"}])).json()["sections"][0]

    whole = answer([row(0, record="r3", part=3, parts=3, text="c"), row(0, record="r1", part=1, parts=3, text="a"),
                    row(0, record="r2", part=2, parts=3, text="b")])
    assert whole["status"] == "found" and [p["text"] for p in whole["parts"]] == ["a", "b", "c"]
    assert whole["parts_expected"] == 3

    gap = answer([row(0, record="r1", part=1, parts=3), row(0, record="r3", part=3, parts=3)])
    assert gap["status"] == "incomplete" and [p["part"] for p in gap["parts"]] == [1, 3] and gap["parts_expected"] == 3
    twice = answer([row(0, record="r1", part=1, parts=2), row(0, record="r1b", part=1, parts=2)])
    assert twice["status"] == "incomplete"
    disagree = answer([row(0, record="r1", part=1, parts=2), row(0, record="r2", part=2, parts=3)])
    assert disagree["status"] == "incomplete" and disagree["parts_expected"] is None
    no_words = answer([row(0, text=None)])
    assert no_words["status"] == "incomplete" and no_words["parts"][0]["text"] is None
    not_words = answer([row(0, text=42)])
    assert not_words["status"] == "incomplete" and not_words["parts"][0]["text"] is None
    # A section kept in one record that states neither its part nor a count is whole.
    single = answer([row(0, part=None, parts=None)])
    assert single["status"] == "found" and single["parts_expected"] == 1
    blank = answer([row(0, text=" \n ")])
    assert blank["status"] == "incomplete" and blank["parts"][0]["text"] == " \n "
    # A stored count is never used to size anything: an absurd one is answered at once, as incomplete.
    began = time.monotonic()
    assert answer([row(0, part=1, parts=10**12)])["status"] == "incomplete"
    assert time.monotonic() - began < 2
    # A part number written as text or as a boolean is not a part number.
    assert answer([row(0, part="1", parts=1)])["status"] == "incomplete"
    assert answer([row(0, part=True, parts=True)])["status"] == "incomplete"


def law_only(item, engagement, current="v1"):
    """The row the read answers for an engagement that holds the law but no section of that number at that version."""
    return [item, engagement, "Example Rules", None, None, None, None, None, None, None, None, None, None, current]


def test_an_item_is_read_from_the_first_engagement_holding_the_law_even_when_it_lacks_the_section(fake, client):
    # The requesting engagement holds the law, but its version in force has no such section; a readable engagement
    # holds an older load that does. The answer is not found: never another load's section passed off as in force.
    fake.rows = [law_only(0, "eng-1", current="v2"), row(0, engagement="eng-2", record="b1", text="old")]
    entry = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01"}], readable_engagement_ids=["eng-2"])).json()["sections"][0]
    assert entry["status"] == "not_found" and entry["parts"] == []
    # An engagement that does not hold the law at all is passed over.
    fake.rows = [row(0, engagement="eng-2", record="b1", text="from eng-2")]
    entry = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01"}], readable_engagement_ids=["eng-2"])).json()["sections"][0]
    assert entry["status"] == "found" and entry["engagement_id"] == "eng-2"


def test_a_law_in_two_engagements_is_read_from_the_first_named_and_never_mixed(fake, client):
    fake.rows = [row(0, engagement="eng-2", record="b1", part=1, parts=1, text="B"),
                 row(0, engagement="eng-1", record="a1", part=1, parts=2, text="A1"),
                 row(0, engagement="eng-1", record="a2", part=2, parts=2, text="A2")]
    entry = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01"}], readable_engagement_ids=["eng-2"])).json()["sections"][0]
    assert entry["engagement_id"] == "eng-1" and [p["text"] for p in entry["parts"]] == ["A1", "A2"]
    entry = post(client, "/law/sections", {**ask([{"law": "law-1", "section": "1.01"}]), "engagement_id": "eng-3",
                                           "readable_engagement_ids": ["eng-2", "eng-1"]}).json()["sections"][0]
    assert entry["engagement_id"] == "eng-2" and [p["text"] for p in entry["parts"]] == ["B"]
    assert fake.reads[-1][2]["engagement_ids"] == ["eng-3", "eng-2", "eng-1"]


# -- values are parameters, never Cypher ---------------------------------


def test_request_values_travel_only_as_parameters_and_the_statement_never_changes(fake, client):
    hostile = ['1.01"}) MATCH (n) DETACH DELETE n //', "x' OR '1'='1", "$evil", "\\", "`n`", "{a: 1}"]
    items = [{"law": h, "section": h, "version": h} for h in hostile]
    response = post(client, "/law/sections", ask(items, engagement_id='e"]) RETURN 1 //'))
    assert response.status_code == 200, response.text
    (name, statement, params, timeout), = fake.reads
    assert name == GRAPH and statement == law_sections.SECTIONS_QUERY
    for value in hostile + ['e"]) RETURN 1 //']:
        assert value not in statement
    assert params == {
        "group_id": GRAPH,
        "engagement_ids": ['e"]) RETURN 1 //'],
        "items": [{"index": i, "law": h, "section": h, "version": h} for i, h in enumerate(hostile)],
    }
    assert timeout == law_sections.LAW_READ_TIMEOUT_MS
    post(client, "/law/sources", {"client_slug": CLIENT, "engagement_id": "eng-1"})
    assert fake.reads[-1][1] == law_sections.SOURCES_QUERY


def test_the_statements_keep_to_the_in_force_rule_and_the_tenant_and_engagement():
    q = law_sections.SECTIONS_QUERY
    for clause in [
        "MATCH (w:ProjectionNode {uuid: item.law})",
        "w.group_id = $group_id AND w.engagement_id IN $engagement_ids",
        "w.projection_source_kind = 'LAW_SOURCE'",
        "OPTIONAL MATCH (n:LawRecordNode)",
        "n.group_id = $group_id AND n.engagement_id = w.engagement_id",
        "n.workflow_node_id = w.uuid",
        "n.projection_source_kind IN ['LAW_SECTION']",
        "n.projection_source_version_id = coalesce(item.version, w.current_law_version_id)",
        "n.section = item.section",
        "n.law_text AS text",
    ]:
        assert clause in q, clause
    # Equality only: nothing that would match a near or partial number.
    for loose in ["CONTAINS", "STARTS WITH", "ENDS WITH", "=~", "toLower", "toUpper", "trim(", "toFloat", "toInteger",
                  "queryNodes", "vec."]:
        assert loose not in q and loose not in law_sections.SOURCES_QUERY, loose
    s = law_sections.SOURCES_QUERY
    assert "w.group_id = $group_id AND w.engagement_id IN $engagement_ids" in s
    assert "n.group_id = $group_id AND n.engagement_id = w.engagement_id" in s
    assert "n.projection_source_version_id = w.current_law_version_id" in s
    for statement in (q, s):
        assert "MERGE" not in statement and "SET " not in statement and "DELETE" not in statement
        assert "CREATE" not in statement


# -- the request ---------------------------------------------------------


def test_the_section_and_law_are_trimmed_of_whitespace_and_nothing_else(fake, client):
    items = [{"law": " law-1\t", "section": "\n 1.01 "}, {"law": "law-1", "section": "1. 01"}, {"law": "Law-1", "section": "01.01"}]
    body = post(client, "/law/sections", ask(items)).json()
    assert [(e["law"], e["section"]) for e in body["sections"]] == [("law-1", "1.01"), ("law-1", "1. 01"), ("Law-1", "01.01")]
    assert [(i["law"], i["section"], i["version"]) for i in fake.reads[-1][2]["items"]] == [
        ("law-1", "1.01", None), ("law-1", "1. 01", None), ("Law-1", "01.01", None)]
    post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01", "version": " v1 \n"}]))
    assert fake.reads[-1][2]["items"][0]["version"] == "v1"


def test_a_pinned_version_is_echoed_and_the_answer_says_whether_it_is_still_in_force(fake, client):
    fake.rows = [row(0, version="v1", current="v2")]
    entry = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01", "version": "v1"}])).json()["sections"][0]
    assert (entry["version"], entry["version_id"], entry["current_version_id"], entry["in_force"]) == ("v1", "v1", "v2", False)
    fake.rows = [row(0, version="v1", current=None)]
    entry = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01", "version": "v1"}])).json()["sections"][0]
    assert entry["in_force"] is False and entry["current_version_id"] == ""
    fake.rows = []
    entry = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01", "version": "v9"}])).json()["sections"][0]
    assert entry["status"] == "not_found" and entry["version"] == "v9"


@pytest.mark.parametrize("item", [{"law": "law-1", "section": "   "}, {"law": " ", "section": "1.01"},
                                  {"law": "law-1", "section": ""}, {"law": "law-1"}, {"section": "1.01"},
                                  {"law": "law-1", "section": "1.01", "extra": 1}, {"law": "law-1", "section": 1.01},
                                  {"law": "law-1", "section": "1.01", "version": " "},
                                  {"law": "law-1", "section": "1.01", "version": ""},
                                  {"law": "law-1", "section": "1.\u000001"}, {"law": "law\u0000", "section": "1.01"},
                                  {"law": "law-1", "section": "1.01", "version": "v\u00001"}])
def test_a_malformed_item_is_refused(fake, client, item):
    assert post(client, "/law/sections", ask([item])).status_code == 422
    assert fake.reads == []


def test_the_number_of_items_is_capped_with_a_clear_error(fake, client):
    at_cap = [{"law": "law-1", "section": f"1.{i:02d}"} for i in range(MAX_LAW_SECTION_ITEMS)]
    assert post(client, "/law/sections", ask(at_cap)).status_code == 200
    over = post(client, "/law/sections", ask(at_cap + [{"law": "law-1", "section": "9.99"}]))
    assert over.status_code == 422
    assert f"at most {MAX_LAW_SECTION_ITEMS} sections may be asked for in one request (51 were)" in over.text
    assert post(client, "/law/sections", ask([])).status_code == 422
    assert len(fake.reads) == 1


def test_an_unknown_field_and_an_invalid_tenant_are_refused(fake, client):
    assert post(client, "/law/sections", ask([{"law": "l", "section": "1"}], query="x")).status_code == 422
    assert post(client, "/law/sources", {"client_slug": CLIENT, "engagement_id": "eng-1", "max_results": 3}).status_code == 422
    assert post(client, "/law/sections", {**ask([{"law": "l", "section": "1"}]), "client_slug": "Bad_Slug"}).status_code == 422
    assert fake.reads == []


# -- the perimeter -------------------------------------------------------


def test_the_routes_take_the_signed_search_scope_and_the_tenant_claim_must_match(fake, client):
    body = ask([{"law": "law-1", "section": "1.01"}])
    assert client.post("/law/sections", json=body).status_code == 401
    assert post(client, "/law/sections", body, scope="ingest", secret=INGEST_SECRET).status_code == 403
    assert post(client, "/law/sections", body, header_client="other-tenant").status_code == 403
    assert post(client, "/law/sections", body, secret="wrong-secret-that-is-at-least-32-chars").status_code == 401
    listing = {"client_slug": CLIENT, "engagement_id": "eng-1"}
    assert client.post("/law/sources", json=listing).status_code == 401
    assert post(client, "/law/sources", listing, header_client="other-tenant").status_code == 403
    assert fake.reads == []


@pytest.mark.parametrize(
    "process, law, mode, open_",
    [
        (False, True, "required", True),
        (True, False, "required", True),
        (True, True, "required", True),
        (False, False, "required", False),
        (True, True, "optional", False),
    ],
)
def test_reads_are_open_on_the_same_terms_as_the_record_search_and_refused_out_loud_otherwise(
    fake, client, monkeypatch, process, law, mode, open_
):
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", process, raising=False)
    monkeypatch.setattr(settings, "projection_v2_allow_law", law, raising=False)
    monkeypatch.setattr(settings, "graphiti_auth_mode", mode, raising=False)
    sections = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01"}]))
    listing = post(client, "/law/sources", {"client_slug": CLIENT, "engagement_id": "eng-1"})
    if open_:
        assert sections.status_code == 200 and listing.status_code == 200
    else:
        assert sections.status_code == 503 and listing.status_code == 503
        assert sections.json() == {"detail": "Law reads are not enabled on this service"}
        assert fake.reads == []


# -- failures ------------------------------------------------------------


def test_a_read_that_fails_is_an_error_never_a_not_found(fake, client):
    fake.error = RuntimeError("engine said something internal: password=hunter2")
    response = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01"}]))
    assert response.status_code == 500 and response.json() == {"detail": "Law read failed"}
    assert "hunter2" not in response.text
    listing = post(client, "/law/sources", {"client_slug": CLIENT, "engagement_id": "eng-1"})
    assert listing.status_code == 500 and listing.json() == {"detail": "Law read failed"}


def test_a_tenant_with_no_graph_has_no_law(fake, client):
    fake.graphs = []
    fake.error = RuntimeError("Invalid graph operation on empty key")
    body = post(client, "/law/sections", ask([{"law": "law-1", "section": "1.01"}])).json()
    assert [e["status"] for e in body["sections"]] == ["not_found"]
    assert post(client, "/law/sources", {"client_slug": CLIENT, "engagement_id": "eng-1"}).json()["sources"] == []


def test_the_listing_reports_what_is_stored_and_says_when_it_is_cut(fake, client, monkeypatch):
    fake.rows = [["law-1", "eng-1", "Example Rules", "v2", 3, 4], ["law-2", "eng-1", None, "v1", 0, 0]]
    body = post(client, "/law/sources", {"client_slug": CLIENT, "engagement_id": "eng-1"}).json()
    assert body == {
        "contract_version": "graphiti_law_sources_v1",
        "sources": [
            {"law": "law-1", "engagement_id": "eng-1", "title": "Example Rules", "version_id": "v2", "sections": 3, "parts": 4},
            {"law": "law-2", "engagement_id": "eng-1", "title": "", "version_id": "v1", "sections": 0, "parts": 0},
        ],
        "truncated": False,
        "graph_name": GRAPH,
    }
    monkeypatch.setattr(law_sections, "MAX_LAW_SOURCES", 1)
    body = post(client, "/law/sources", {"client_slug": CLIENT, "engagement_id": "eng-1"}).json()
    assert [s["law"] for s in body["sources"]] == ["law-1"] and body["truncated"] is True


def test_the_record_search_is_unchanged_by_the_new_routes():
    """POST /search/records keeps its contract: the same fields, and law only when asked for by law_results."""
    from app.models.search import RecordSearchRequest, RecordSearchResponse

    assert set(RecordSearchRequest.model_fields) == {
        "client_slug", "engagement_id", "readable_engagement_ids", "query", "max_results", "law_results",
    }
    assert set(RecordSearchResponse.model_fields) == {"contract_version", "records", "law", "graph_name", "search_time_ms"}
