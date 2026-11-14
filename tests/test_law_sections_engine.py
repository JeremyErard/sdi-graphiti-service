"""Exact law-section reads on a real FalkorDB engine: written through the projection route, read through /law.

What a fake cannot show is what these check: that text survives the driver
(which writes query parameters into the query text itself) and the engine byte
for byte, that a hostile value in a request is data and never Cypher, and that
the tenant and engagement scope holds in the engine's own matching.

Needs a FalkorDB the tests may write to and delete graphs from. Skipped unless
GRAPH_TEST_FALKORDB_PORT names one (GRAPH_TEST_FALKORDB_HOST defaults to
127.0.0.1). Every graph a test writes is named after a fresh random tenant and
deleted afterwards. Never point this at a database that holds anything else.

Run: GRAPH_TEST_FALKORDB_PORT=<port> python -m pytest tests/test_law_sections_engine.py -q
"""

import hashlib
import json
import os
import time
import uuid as uuidlib

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.auth import build_signature
from app.config import settings
from app.graph_names import graph_name_for_client
from app.main import app
from app.routers import projection
from app.services import graphiti_client

PORT = os.getenv("GRAPH_TEST_FALKORDB_PORT")
HOST = os.getenv("GRAPH_TEST_FALKORDB_HOST", "127.0.0.1")

pytestmark = pytest.mark.skipif(not PORT, reason="needs GRAPH_TEST_FALKORDB_PORT: a FalkorDB the tests may write to")

INGEST_SECRET = "ingest-secret-for-engine-tests-0123456789"
SEARCH_SECRET = "search-secret-for-engine-tests-0123456789"
ADMIN_SECRET = "admin-secret-for-engine-tests-01234567890"

# Text no reader should ever see altered: quotes of both kinds, backslashes (alone, doubled, before a quote and
# before letters a parser might read as an escape), line breaks of three kinds, a tab, text outside the Basic
# Multilingual Plane, right-to-left text, combining-free accented letters (NFC, as the write path requires),
# zero-width and non-breaking spaces, and strings that would be Cypher if they were ever spliced into a query.
HOSTILE_TEXT = (
    'Section 1.01 Example heading. "Quoted" and \'single\' and `backtick`.\n'
    "Backslashes: \\ \\\\ \\\" \\' \\n \\t \\u0041 end\\\n"
    "Windows line\r\nold Mac line\rtab\there\n"
    "Unicode: café naïve Ωμέγα 中文 日本語 עברית العربية 😀 𝔘𝔫𝔦𝔠𝔬𝔡𝔢​   .\n"
    '"}) MATCH (n) DETACH DELETE n //\n'
    "'}) RETURN 1 UNION MATCH (m) RETURN m //\n"
    "$group_id $items ${x} {{law_text}} CYPHER items=[] \n"
    "  leading and trailing spaces are kept  "
)


@pytest.fixture
def engine(monkeypatch):
    from falkordb import FalkorDB

    try:
        db = FalkorDB(host=HOST, port=int(PORT), socket_connect_timeout=2, socket_timeout=10)
        db.connection.ping()
    except Exception as error:  # pragma: no cover - only when the named engine is not there
        pytest.skip(f"no FalkorDB at the configured test port ({type(error).__name__})")

    monkeypatch.setattr(settings, "falkordb_host", HOST, raising=False)
    monkeypatch.setattr(settings, "falkordb_port", int(PORT), raising=False)
    monkeypatch.setattr(settings, "falkordb_password", "", raising=False)
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required", raising=False)
    monkeypatch.setattr(settings, "graphiti_ingest_secret", INGEST_SECRET, raising=False)
    monkeypatch.setattr(settings, "graphiti_search_secret", SEARCH_SECRET, raising=False)
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
    graphiti_client.reset_falkor_db()
    projection._reset_falkor_db()
    projection._INDEXED_GRAPHS.clear()

    tenants: list[str] = []

    def tenant() -> str:
        slug = f"lawtest-{uuidlib.uuid4().hex[:12]}"
        tenants.append(slug)
        return slug

    yield db, tenant

    for slug in tenants:
        name = graph_name_for_client(slug)
        if name in (db.list_graphs() or []):
            db.select_graph(name).delete()
    graphiti_client.reset_falkor_db()
    projection._reset_falkor_db()
    projection._INDEXED_GRAPHS.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def signed_post(client, path, body, *, scope, secret):
    raw = json.dumps(body).encode("utf-8")
    timestamp, nonce = str(int(time.time())), uuidlib.uuid4().hex
    signature = build_signature(
        secret=secret, timestamp=timestamp, nonce=nonce, method="POST", path=path, scope=scope,
        client_slug=body["client_slug"], body=raw,
    )
    return client.post(
        path,
        content=raw,
        headers={
            "content-type": "application/json",
            auth.AUTH_TIMESTAMP_HEADER: timestamp,
            auth.AUTH_SCOPE_HEADER: scope,
            auth.AUTH_CLIENT_HEADER: body["client_slug"],
            auth.AUTH_NONCE_HEADER: nonce,
            auth.AUTH_SIGNATURE_HEADER: signature,
        },
    )


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def write_law(client, slug, engagement, law_id, version, sections, *, title="Example Rules"):
    """Write one law through POST /ingest/projection/v2, as the other service does.

    `sections` is a list of (section number, heading, [part texts]). Each part is its own record, numbered from 1,
    stating how many parts its section has, as the other service writes them.
    """
    node_ops, edge_ops = [], []
    for number, heading, parts in sections:
        for index, text in enumerate(parts, start=1):
            node_id = f"{law_id}@{version}:{number}:{index}"
            properties = {
                "text": text, "law_title": title, "section": number, "title": heading, "chapter": "",
                "part": index, "parts": len(parts), "workflow_node_id": law_id,
            }
            node_ops.append({
                "node_id": node_id,
                "source": {"kind": "LAW_SECTION", "id": f"{law_id}:{number}:{index}", "immutable_version_id": version,
                           "content_hash": _sha(properties)},
                "properties": properties,
            })
            edge_ops.append({"edge_id": f"{node_id}|SECTION_OF", "relation": "SECTION_OF", "from_node_id": node_id,
                             "to_node_id": law_id})
    law_properties = {"name": title, "current_law_version_id": version}
    node_ops.append({
        "node_id": law_id,
        "source": {"kind": "LAW_SOURCE", "id": law_id, "immutable_version_id": f"{law_id}@{version}",
                   "content_hash": _sha(law_properties)},
        "properties": law_properties,
    })
    body = {
        "schema_version": "projection.v2",
        "client_slug": slug,
        "namespace": graph_name_for_client(slug),
        "engagement_id": engagement,
        "origin": "DIRECT_IMPORT",
        "cursor": {"import_manifest_hash": uuidlib.uuid4().hex * 2, "batch_ordinal": 0},
        "operation_id": f"op-{uuidlib.uuid4().hex}",
        "cutoff": {"cutoff_id": f"cutoff-{uuidlib.uuid4().hex}"},
        "node_ops": node_ops,
        "edge_ops": edge_ops,
    }
    response = signed_post(client, "/ingest/projection/v2", body, scope="ingest", secret=INGEST_SECRET)
    assert response.status_code == 200, response.text


def fetch(client, slug, engagement, items, readable=None):
    body = {"client_slug": slug, "engagement_id": engagement, "items": items}
    if readable is not None:
        body["readable_engagement_ids"] = readable
    return signed_post(client, "/law/sections", body, scope="search", secret=SEARCH_SECRET)


def sources(client, slug, engagement, readable=None):
    body = {"client_slug": slug, "engagement_id": engagement}
    if readable is not None:
        body["readable_engagement_ids"] = readable
    return signed_post(client, "/law/sources", body, scope="search", secret=SEARCH_SECRET)


def one(response):
    assert response.status_code == 200, response.text
    (entry,) = response.json()["sections"]
    return entry


def test_hostile_text_round_trips_byte_for_byte_through_write_and_fetch(engine, client):
    _, tenant = engine
    slug = tenant()
    write_law(client, slug, "eng-a", "law-1", "v1", [("1.01", 'Example "heading" \\ one', [HOSTILE_TEXT])])
    entry = one(fetch(client, slug, "eng-a", [{"law": "law-1", "section": "1.01"}]))
    assert entry["status"] == "found"
    assert entry["heading"] == 'Example "heading" \\ one'
    (part,) = entry["parts"]
    assert part["text"] == HOSTILE_TEXT
    assert part["text"].encode("utf-8") == HOSTILE_TEXT.encode("utf-8")
    assert entry["version_id"] == "v1" and entry["law_title"] == "Example Rules" and entry["engagement_id"] == "eng-a"


def test_a_section_in_parts_comes_back_in_order_and_whole(engine, client):
    _, tenant = engine
    slug = tenant()
    parts = ["Section 2.01 Example heading, part one.", "Part two, \"quoted\".", "Part three.\nLast line."]
    write_law(client, slug, "eng-a", "law-1", "v1", [("2.01", "Example heading", parts), ("2.02", "Other", ["x."])])
    entry = one(fetch(client, slug, "eng-a", [{"law": "law-1", "section": "2.01"}]))
    assert entry["status"] == "found" and entry["parts_expected"] == 3
    assert [p["part"] for p in entry["parts"]] == [1, 2, 3]
    assert [p["text"] for p in entry["parts"]] == parts


def test_a_section_written_for_one_scope_is_never_returned_for_another(engine, client):
    db, tenant = engine
    first, second = tenant(), tenant()
    write_law(client, first, "eng-a", "law-1", "v1", [("1.01", "Example heading", ["Text of the first tenant."])])
    write_law(client, second, "eng-b", "law-1", "v1", [("1.01", "Example heading", ["Text of the second tenant."])])
    ask = [{"law": "law-1", "section": "1.01"}]

    # Another engagement of the same tenant does not see it, unless the caller says it may read that engagement.
    assert one(fetch(client, first, "eng-b", ask))["status"] == "not_found"
    assert one(fetch(client, first, "eng-other", ask, readable=["eng-c"]))["status"] == "not_found"
    readable = one(fetch(client, first, "eng-other", ask, readable=["eng-a"]))
    assert readable["status"] == "found" and readable["parts"][0]["text"] == "Text of the first tenant."

    # Another tenant never does, whatever engagement it names: its own law with the same ids answers its own text.
    assert one(fetch(client, second, "eng-a", ask))["status"] == "not_found"
    assert one(fetch(client, second, "eng-b", ask))["parts"][0]["text"] == "Text of the second tenant."
    assert one(fetch(client, first, "eng-a", ask))["parts"][0]["text"] == "Text of the first tenant."

    # A tenant with no graph at all has no law: not found, not an error, and no graph is made by asking.
    nobody = tenant()
    assert one(fetch(client, nobody, "eng-a", ask))["status"] == "not_found"
    assert graph_name_for_client(nobody) not in (db.list_graphs() or [])

    # Listing keeps to its scope too.
    assert [s["law"] for s in sources(client, first, "eng-a").json()["sources"]] == ["law-1"]
    assert sources(client, first, "eng-b").json()["sources"] == []
    assert sources(client, nobody, "eng-a").json()["sources"] == []


def test_one_law_loaded_in_two_engagements_is_read_from_one_and_never_mixed(engine, client):
    _, tenant = engine
    slug = tenant()
    write_law(client, slug, "eng-a", "law-1", "v1", [("1.01", "Example heading", ["A, part one.", "A, part two."])])
    # The same law id and the same version id in both (a version id is a digest of the text, so two loads of one
    # document share it); the words differ here so a mix would show.
    write_law(client, slug, "eng-b", "law-1", "v1", [("1.01", "Example heading", ["B, the only part."])])
    ask = [{"law": "law-1", "section": "1.01"}]

    def texts(entry):
        return entry["engagement_id"], entry["version_id"], [p["text"] for p in entry["parts"]]

    assert texts(one(fetch(client, slug, "eng-a", ask))) == ("eng-a", "v1", ["A, part one.", "A, part two."])
    assert texts(one(fetch(client, slug, "eng-b", ask))) == ("eng-b", "v1", ["B, the only part."])
    # Reading both, the requesting engagement comes first, then the readable ones in the order named.
    assert texts(one(fetch(client, slug, "eng-b", ask, readable=["eng-a"])))[0] == "eng-b"
    assert texts(one(fetch(client, slug, "eng-a", ask, readable=["eng-b"])))[0] == "eng-a"
    assert texts(one(fetch(client, slug, "eng-x", ask, readable=["eng-b", "eng-a"])))[0] == "eng-b"
    assert texts(one(fetch(client, slug, "eng-x", ask, readable=["eng-a", "eng-b"])))[0] == "eng-a"
    listed = sources(client, slug, "eng-a", readable=["eng-b"]).json()["sources"]
    assert [(s["law"], s["engagement_id"], s["parts"]) for s in listed] == [("law-1", "eng-a", 2), ("law-1", "eng-b", 1)]


def test_a_section_the_asking_engagements_law_no_longer_has_is_not_served_from_another(engine, client):
    _, tenant = engine
    slug = tenant()
    write_law(client, slug, "eng-a", "law-1", "v1", [("1.01", "Example heading", ["Old A."]), ("1.02", "Other", ["A."])])
    write_law(client, slug, "eng-a", "law-1", "v2", [("1.02", "Other", ["A, amended."])])
    write_law(client, slug, "eng-b", "law-1", "v1", [("1.01", "Example heading", ["Old B."]), ("1.02", "Other", ["B."])])
    ask = [{"law": "law-1", "section": "1.01"}]
    assert one(fetch(client, slug, "eng-a", ask, readable=["eng-b"]))["status"] == "not_found"
    # An engagement that does not hold the law is passed over, so the readable one answers.
    found = one(fetch(client, slug, "eng-c", ask, readable=["eng-b"]))
    assert found["status"] == "found" and found["engagement_id"] == "eng-b" and found["parts"][0]["text"] == "Old B."


def test_the_section_number_matches_exactly_after_trimming_whitespace_only(engine, client):
    _, tenant = engine
    slug = tenant()
    write_law(client, slug, "eng-a", "law-1", "v1", [
        ("1.01", "Example heading", ["One."]), ("1.10", "Tenth", ["Ten."]), ("history", "History", ["Past."]),
    ])
    asked = [" 1.01 ", "\t1.01\n", "1.01", "1.1", "1.010", "01.01", "1.0", "1,01", "1.01a", "1. 01", "1.10",
             "History", "HISTORY", "history"]
    response = fetch(client, slug, "eng-a", [{"law": "law-1", "section": s} for s in asked])
    assert response.status_code == 200, response.text
    got = {(e["section"], e["status"]) for e in response.json()["sections"]}
    assert [e["status"] for e in response.json()["sections"]] == [
        "found", "found", "found", "not_found", "not_found", "not_found", "not_found", "not_found", "not_found",
        "not_found", "found", "not_found", "not_found", "found",
    ], got
    assert [e["section"] for e in response.json()["sections"]][:2] == ["1.01", "1.01"]
    texts = {e["section"]: e["parts"][0]["text"] for e in response.json()["sections"] if e["status"] == "found"}
    assert texts == {"1.01": "One.", "1.10": "Ten.", "history": "Past."}
    # The law id is matched exactly too: no prefix, no case folding.
    for law in ["law-", "LAW-1", "law-10", "law"]:
        assert one(fetch(client, slug, "eng-a", [{"law": law, "section": "1.01"}]))["status"] == "not_found"


def test_only_the_version_in_force_is_returned(engine, client):
    _, tenant = engine
    slug = tenant()
    write_law(client, slug, "eng-a", "law-1", "v1", [("1.01", "Example heading", ["Old text."]), ("1.02", "Gone", ["Dropped."])])
    write_law(client, slug, "eng-a", "law-1", "v2", [("1.01", "Example heading", ["New text."])])
    entry = one(fetch(client, slug, "eng-a", [{"law": "law-1", "section": "1.01"}]))
    assert entry["version_id"] == "v2" and [p["text"] for p in entry["parts"]] == ["New text."]
    assert entry["in_force"] is True and entry["current_version_id"] == "v2" and entry["version"] is None
    # A section only the old version had is not in force, though it is still on record.
    assert one(fetch(client, slug, "eng-a", [{"law": "law-1", "section": "1.02"}]))["status"] == "not_found"
    # A caller that recorded the section at v1 gets v1's words back, exactly, told that v1 is no longer in force.
    pinned = one(fetch(client, slug, "eng-a", [{"law": "law-1", "section": "1.01", "version": "v1"}]))
    assert pinned["status"] == "found" and [p["text"] for p in pinned["parts"]] == ["Old text."]
    assert (pinned["version"], pinned["version_id"], pinned["current_version_id"], pinned["in_force"]) == ("v1", "v1", "v2", False)
    assert one(fetch(client, slug, "eng-a", [{"law": "law-1", "section": "1.02", "version": "v1"}]))["status"] == "found"
    # Pinned versions match exactly too.
    for version in ["v", "V1", "v1 x", "v3"]:
        assert one(fetch(client, slug, "eng-a", [{"law": "law-1", "section": "1.01", "version": version}]))["status"] == "not_found"
    # A pinned version of one engagement's law is not read for another engagement.
    assert one(fetch(client, slug, "eng-b", [{"law": "law-1", "section": "1.01", "version": "v1"}]))["status"] == "not_found"
    (listed,) = sources(client, slug, "eng-a").json()["sources"]
    assert listed == {"law": "law-1", "engagement_id": "eng-a", "title": "Example Rules", "version_id": "v2",
                      "sections": 1, "parts": 1}


def test_a_hostile_request_is_data_never_cypher_and_changes_nothing(engine, client):
    db, tenant = engine
    slug = tenant()
    write_law(client, slug, "eng-a", "law-1", "v1", [("1.01", "Example heading", ["One."])])
    graph = db.select_graph(graph_name_for_client(slug))
    before = graph.ro_query("MATCH (n) RETURN count(n)").result_set[0][0]
    hostile = [
        '1.01"}) MATCH (n) DETACH DELETE n //',
        "1.01' OR '1'='1",
        '" OR 1=1 OR "',
        "\\",
        '\\"',
        "1.01\\",
        "$section",
        "*",
        ".*",
        "1.0%",
        "x\ny",
    ]
    items = ([{"law": value, "section": "1.01"} for value in hostile] + [{"law": "law-1", "section": v} for v in hostile]
             + [{"law": "law-1", "section": "1.01", "version": v} for v in hostile])
    response = fetch(client, slug, "eng-a", items)
    assert response.status_code == 200, response.text
    assert {e["status"] for e in response.json()["sections"]} == {"not_found"}
    assert [(e["law"], e["section"]) for e in response.json()["sections"]] == [
        (i["law"].strip(), i["section"].strip()) for i in items
    ]
    hostile_scope = fetch(client, slug, "eng-a\"]) MATCH (n) DETACH DELETE n //", [{"law": "law-1", "section": "1.01"}])
    assert one(hostile_scope)["status"] == "not_found"
    assert graph.ro_query("MATCH (n) RETURN count(n)").result_set[0][0] == before
    assert one(fetch(client, slug, "eng-a", [{"law": "law-1", "section": "1.01"}]))["status"] == "found"


def test_reads_do_not_depend_on_the_law_write_switch(engine, client, monkeypatch):
    _, tenant = engine
    slug = tenant()
    write_law(client, slug, "eng-a", "law-1", "v1", [("1.01", "Example heading", ["One."])])
    monkeypatch.setattr(settings, "projection_v2_allow_law", False, raising=False)
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", True, raising=False)
    assert one(fetch(client, slug, "eng-a", [{"law": "law-1", "section": "1.01"}]))["status"] == "found"
