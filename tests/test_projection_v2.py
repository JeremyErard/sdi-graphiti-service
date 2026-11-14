"""Contract tests for the governed exact-ID projection endpoint.

Every clause of the ratified severability contract has at least one test here
that fails if the behavior is removed. The FalkorDB fake below is not a stub that
agrees with the router: it parses the Cypher the router actually emits and
derives its behavior from that text, so dropping a key from a MERGE, swapping
SET n = for SET n +=, or removing a WHERE clause changes what the fake stores and
is observed as state rather than asserted as intention.

Nothing in this module touches a live database, a live Redis, or a network.

Run: .venv/bin/python -m pytest tests/test_projection_v2.py -q
"""

import copy
import hashlib
import json
import re
import time
import unicodedata
import uuid as uuidlib

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.auth import build_signature
from app.config import settings
from app.main import app
from app.models import projection as pm
from app.models.projection import (
    MAX_OPERATIONS_PER_ENVELOPE,
    MAX_PROPERTIES_PER_OPERATION,
    ProjectionEnvelopeV2,
)
from app.routers import projection

PATH = "/ingest/projection/v2"
RECEIPTS_PATH = "/ingest/projection/v2/receipts"

CLIENT = "unit-tenant"
GRAPH = "client_unit_tenant"
ENGAGEMENT = "eng-1"
MANIFEST_HASH = "1f" * 32
CONTENT_HASH = "b3" * 32
DRIVER_HASH = "c4" * 32
RECEIPT_HASH = "d5" * 32
INGEST_SECRET = "ingest-secret-that-is-at-least-32-characters"
SEARCH_SECRET = "search-secret-that-is-at-least-32-characters"
ADMIN_SECRET = "admin-secret-that-is-at-least-32-charactersx"


# ==========================================================================
# A FalkorDB fake that obeys the query text
# ==========================================================================

_LABEL_KEY = re.compile(r"\(\s*(\w*)\s*:\s*(\w+)\s*\{([^}]*)\}\s*\)")
_SET_LABELS = re.compile(r"SET\s+(\w+)((?::\w+)+)\s")
_SET_MAP = re.compile(r"SET\s+(\w+)\s*(\+?=)\s*(row\.props|\$props)")
_MERGE_EDGE = re.compile(r"MERGE\s*\(\w+\)-\[\s*(\w+)\s*:\s*(\w+)\s*\{([^}]*)\}\s*\]->\(\w+\)")
_EDGE_MATCH = re.compile(r"-\[\s*(\w+)\s*\{([^}]*)\}\s*\]->")


def _parse_map(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in body.split(","):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        out[key.strip()] = value.strip()
    return out


def _resolve(token: str, params: dict, row: dict | None, wanted=None):
    if token.startswith("$"):
        return params.get(token[1:])
    if token.startswith("row."):
        return (row or {}).get(token[4:])
    if token == "wanted":
        return wanted
    return token.strip('"\'')


class FakeResult:
    def __init__(self, rows):
        self.result_set = rows


class FakeStore:
    def __init__(self):
        # Node keys are whatever the router's MERGE clause actually keys on. If
        # the router drops group_id, or keys on a type label instead of the
        # constant projection label, these keys change shape and the isolation
        # and convergence tests see it.
        self.nodes: dict[tuple, dict] = {}
        self.edges: dict[tuple, dict] = {}
        self.receipts: dict[tuple, dict] = {}
        self.queries: list[tuple[str, dict]] = []
        self.indices: list[str] = []
        self.selected: list[str] = []
        self.memory_probes: list[int] = []
        # Failure injection.
        self.fail_node_apply_on: int | None = None
        self.node_apply_calls = 0
        self.drop_after_resolve: str | None = None
        self.competing_receipt: dict | None = None
        self.skip_node_row: str | None = None
        self.drop_receipt_before_finalize = False

    def snapshot(self):
        return copy.deepcopy((self.nodes, self.edges, self.receipts))

    def mutating_queries(self):
        return [
            q
            for q, _ in self.queries
            if ("MERGE" in q or "SET " in q or "DELETE" in q) and not q.startswith("CREATE INDEX")
        ]


class FakeGraph:
    def __init__(self, store: FakeStore, name: str):
        self.store = store
        self.name = name

    # -- helpers ---------------------------------------------------------
    def _node_key(self, label: str, keymap: dict, params: dict, row: dict | None, wanted=None):
        return (label,) + tuple(
            sorted((k, _resolve(v, params, row, wanted)) for k, v in keymap.items())
        )

    def _match_nodes(self, label: str, keymap: dict, params: dict, row=None, wanted=None):
        key = self._node_key(label, keymap, params, row, wanted)
        found = self.store.nodes.get(key)
        return [found] if found is not None else []

    # -- dispatcher ------------------------------------------------------
    def query(self, q: str, params: dict | None = None, timeout=None):
        params = params or {}
        self.store.queries.append((q, copy.deepcopy(params)))

        if q.startswith("CREATE INDEX"):
            self.store.indices.append(q)
            return FakeResult([])
        if "ProjectionReceipt" in q:
            return self._receipt_query(q, params)
        if q.startswith("UNWIND $node_ids"):
            return self._resolve_nodes(q, params)
        if q.startswith("UNWIND $edge_ids"):
            return self._resolve_edges(q, params)
        if q.startswith("UNWIND $rows") and "MERGE" in q and "]->" in q:
            return self._apply_edges(q, params)
        if q.startswith("UNWIND $rows") and "MERGE" in q:
            return self._apply_nodes(q, params)
        if q.startswith("UNWIND $rows") and "]->" in q:
            return self._verify_edges(q, params)
        if q.startswith("UNWIND $rows"):
            return self._verify_nodes(q, params)
        raise AssertionError(f"the fake was asked to run an unrecognized query: {q}")

    # -- receipts --------------------------------------------------------
    def _receipt_query(self, q: str, params: dict):
        match = _LABEL_KEY.search(q)
        assert match, q
        _, label, body = match.groups()
        keymap = _parse_map(body)
        key = self._node_key(label, keymap, params, None)

        if q.startswith("MERGE"):
            existing = self.store.receipts.get(key)
            if existing is None and self.store.competing_receipt is not None:
                # Another writer created the receipt between the first read and
                # this MERGE, so the MERGE binds their node. Whether our property
                # write then lands is decided by the guard in the query text.
                self.store.receipts[key] = dict(self.store.competing_receipt)
                self.store.competing_receipt = None
                existing = self.store.receipts[key]
            if existing is None:
                self.store.receipts[key] = {}
                existing = self.store.receipts[key]
            # WITH r WHERE r.envelope_hash IS NULL guards the property write.
            if "r.envelope_hash IS NULL" in q and existing.get("envelope_hash"):
                return FakeResult([])
            setter = _SET_MAP.search(q)
            assert setter, q
            payload = dict(params[setter.group(3)[1:]])
            if setter.group(2) == "=":
                existing.clear()
            existing.update(payload)
            return FakeResult([])

        found = self.store.receipts.get(key)
        if "SET r." in q:
            if self.store.drop_receipt_before_finalize and "RETURN properties(r)" in q:
                self.store.receipts.pop(key, None)
                return FakeResult([])
            if found is None:
                return FakeResult([])
            for field, param in re.findall(r"r\.(\w+)\s*=\s*\$(\w+)", q):
                found[field] = params.get(param)
            if "RETURN properties(r)" in q:
                return FakeResult([[dict(found)]])
            return FakeResult([])

        # Read paths keyed on a partial map (cursor claim, manifest cutoff, list).
        if "RETURN r.operation_id" in q or "RETURN DISTINCT r.cutoff_id" in q:
            wanted = {k: _resolve(v, params, None) for k, v in keymap.items()}
            field = "operation_id" if "RETURN r.operation_id" in q else "cutoff_id"
            rows = [
                [stored.get(field)]
                for stored in self.store.receipts.values()
                if all(stored.get(k) == v for k, v in wanted.items())
            ]
            if "DISTINCT" in q:
                seen, deduped = set(), []
                for row in rows:
                    if row[0] not in seen:
                        seen.add(row[0])
                        deduped.append(row)
                rows = deduped
            return FakeResult(rows)

        if "RETURN properties(r) AS props ORDER BY" in q:
            wanted = {k: _resolve(v, params, None) for k, v in keymap.items()}
            rows = []
            for stored in self.store.receipts.values():
                if not all(stored.get(k) == v for k, v in wanted.items()):
                    continue
                if all(
                    stored.get(field) == params.get(param)
                    for field, param in re.findall(r"r\.(\w+)\s*=\s*\$(\w+)", q)
                ):
                    rows.append([dict(stored)])
            rows.sort(key=lambda r: str(r[0].get("cursor_key", "")))
            return FakeResult(rows)

        return FakeResult([[dict(found)]] if found is not None else [])

    # -- resolve ---------------------------------------------------------
    def _resolve_nodes(self, q: str, params: dict):
        match = _LABEL_KEY.search(q)
        assert match, q
        _, label, body = match.groups()
        keymap = _parse_map(body)
        rows = []
        for wanted in params["node_ids"]:
            if wanted == self.store.drop_after_resolve:
                continue
            for node in self._match_nodes(label, keymap, params, wanted=wanted):
                rows.append([wanted, node["props"].get("projection_source_kind")])
        return FakeResult(rows)

    def _resolve_edges(self, q: str, params: dict):
        # The identity check is one typed, index-answered lookup per relation,
        # with the endpoints read off the edge and held to the projection label
        # space (the base labels that relation's endpoint kinds carry) and the
        # tenant scope in the WHERE clause.
        typed = re.search(r"MATCH \(\)-\[e:(\w+) \{uuid: wanted\}\]->\(\)", q)
        assert typed, q
        start, end = pm.RELATION_ENDPOINT_BASE_LABELS[pm.ProjectionRelation(typed.group(1))]
        assert f"a:{start} AND b:{end}" in q, q
        scope = {key: params[name] for key, name in re.findall(r"\ba\.(\w+) = \$(\w+)", q)}
        assert scope and scope == {key: params[name] for key, name in re.findall(r"\bb\.(\w+) = \$(\w+)", q)}, q
        rows = []
        for wanted in params["edge_ids"]:
            for (_, edge_id, from_id, to_id), edge in self.store.edges.items():
                if edge_id != wanted or edge["relation"] != typed.group(1):
                    continue
                if not all(edge["props"].get(k) == v for k, v in scope.items()):
                    continue
                rows.append([wanted, edge["relation"], from_id, to_id])
        return FakeResult(rows)

    # -- apply -----------------------------------------------------------
    def _apply_nodes(self, q: str, params: dict):
        self.store.node_apply_calls += 1
        if self.store.fail_node_apply_on == self.store.node_apply_calls:
            raise RuntimeError("simulated FalkorDB failure mid-apply")

        match = re.search(r"MERGE \(n:(\w+) \{([^}]*)\}\)", q)
        assert match, q
        merge_label, body = match.groups()
        keymap = _parse_map(body)
        # The write is held off a row whose uuid a node of another base label already holds, in the same statement.
        guards = [(label, _parse_map(guard)) for label, guard in re.findall(r"OPTIONAL MATCH \(other:(\w+) \{([^}]*)\}\) WITH row WHERE other IS NULL ", q)]
        label_set = _SET_LABELS.search(q)
        extra_labels = [lbl for lbl in (label_set.group(2).split(":") if label_set else []) if lbl]
        setter = _SET_MAP.search(q)
        assert setter, q

        applied = 0
        for row in params["rows"]:
            if row.get("node_id") == self.store.skip_node_row:
                applied += 1
                continue
            # A lookup hidden from the preflight sees the graph again once writing starts.
            self.store.drop_after_resolve = None
            if any(self._match_nodes(label, guard, params, row) for label, guard in guards):
                continue
            key = self._node_key(merge_label, keymap, params, row)
            node = self.store.nodes.setdefault(key, {"labels": {merge_label}, "props": {}})
            node["labels"].update(extra_labels)
            payload = copy.deepcopy(row["props"])
            if setter.group(2) == "=":
                node["props"] = payload
            else:
                node["props"].update(payload)
            applied += 1
        return FakeResult([[applied]])

    def _apply_edges(self, q: str, params: dict):
        endpoints = _LABEL_KEY.findall(q)
        assert len(endpoints) >= 2, q
        merge = _MERGE_EDGE.search(q)
        assert merge, q
        relation = merge.group(2)
        setter = _SET_MAP.search(q)
        assert setter, q

        applied = 0
        for row in params["rows"]:
            bound = [
                self._match_nodes(label, _parse_map(body), params, row)
                for _, label, body in endpoints[:2]
            ]
            if not bound[0] or not bound[1]:
                continue
            key = (relation, row["edge_id"], row["from_node_id"], row["to_node_id"])
            edge = self.store.edges.setdefault(key, {"relation": relation, "props": {}})
            payload = copy.deepcopy(row["props"])
            if setter.group(2) == "=":
                edge["props"] = payload
            else:
                edge["props"].update(payload)
            applied += 1
        return FakeResult([[applied]])

    # -- verify ----------------------------------------------------------
    def _verify_nodes(self, q: str, params: dict):
        match = _LABEL_KEY.search(q)
        assert match, q
        _, label, body = match.groups()
        keymap = _parse_map(body)
        checks_hash = "projection_op_hash = row.op_hash" in q
        verified = set()
        for row in params["rows"]:
            for node in self._match_nodes(label, keymap, params, row):
                if checks_hash and node["props"].get("projection_op_hash") != row["op_hash"]:
                    continue
                verified.add(row["node_id"])
        return FakeResult([[len(verified)]])

    def _verify_edges(self, q: str, params: dict):
        endpoints = _LABEL_KEY.findall(q)
        assert endpoints, q
        scope = {
            k: _resolve(v, params, None)
            for k, v in _parse_map(endpoints[0][2]).items()
            if k != "uuid"
        }
        assert _EDGE_MATCH.search(q), q
        checks_hash = "projection_op_hash = row.op_hash" in q
        verified = set()
        for row in params["rows"]:
            for (_, edge_id, from_id, to_id), edge in self.store.edges.items():
                if edge_id != row["edge_id"]:
                    continue
                # Anchored on the endpoints the operation declares.
                if "uuid: row.from_node_id" in q and from_id != row["from_node_id"]:
                    continue
                if "uuid: row.to_node_id" in q and to_id != row["to_node_id"]:
                    continue
                if not all(edge["props"].get(k) == v for k, v in scope.items()):
                    continue
                if checks_hash and edge["props"].get("projection_op_hash") != row["op_hash"]:
                    continue
                verified.add(row["edge_id"])
        return FakeResult([[len(verified)]])


class FakeDB:
    def __init__(self, store: FakeStore):
        self.store = store

    def select_graph(self, name: str):
        self.store.selected.append(name)
        return FakeGraph(self.store, name)


# ==========================================================================
# Fixtures
# ==========================================================================

MEMORY_SERIES = [
    {"used_memory": 1_000, "used_memory_rss": 2_000, "maxmemory": 10_000},
    {"used_memory": 3_000, "used_memory_rss": 4_000, "maxmemory": 10_000},
    {"used_memory": 5_000, "used_memory_rss": 6_000, "maxmemory": 10_000},
]


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(projection, "_open_graph", lambda name: FakeDB(fake).select_graph(name))
    projection._INDEXED_GRAPHS.clear()

    # The perimeter is closed for every test, because the lane refuses to run in
    # any other posture.
    monkeypatch.setattr(settings, "graphiti_auth_mode", "required", raising=False)
    monkeypatch.setattr(settings, "graphiti_ingest_secret", INGEST_SECRET, raising=False)
    monkeypatch.setattr(settings, "graphiti_search_secret", SEARCH_SECRET, raising=False)
    monkeypatch.setattr(settings, "graphiti_admin_secret", ADMIN_SECRET, raising=False)
    monkeypatch.setattr(settings, "projection_v2_allow_outcome_event", False, raising=False)
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", False, raising=False)

    consumed: set[str] = set()

    async def fake_consume(scope, nonce):
        # Real nonce semantics. If the perimeter check ran twice for one request
        # the second call would return False and the request would 409.
        if nonce in consumed:
            return False
        consumed.add(nonce)
        return True

    monkeypatch.setattr(auth, "_consume_nonce", fake_consume)

    async def fake_memory():
        index = min(len(fake.memory_probes), len(MEMORY_SERIES) - 1)
        fake.memory_probes.append(index)
        return dict(MEMORY_SERIES[index])

    monkeypatch.setattr(projection, "_read_memory_info", fake_memory)
    yield fake
    projection._INDEXED_GRAPHS.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def post(client, path, body, *, scope="ingest", secret=INGEST_SECRET, header_client=None):
    raw = json.dumps(body).encode("utf-8")
    header_slug = header_client if header_client is not None else body.get("client_slug", CLIENT)
    timestamp = str(int(time.time()))
    nonce = uuidlib.uuid4().hex
    signature = build_signature(
        secret=secret,
        timestamp=timestamp,
        nonce=nonce,
        method="POST",
        path=path,
        scope=scope,
        client_slug=header_slug,
        body=raw,
    )
    return client.post(
        path,
        content=raw,
        headers={
            "content-type": "application/json",
            auth.AUTH_TIMESTAMP_HEADER: timestamp,
            auth.AUTH_SCOPE_HEADER: scope,
            auth.AUTH_CLIENT_HEADER: header_slug,
            auth.AUTH_NONCE_HEADER: nonce,
            auth.AUTH_SIGNATURE_HEADER: signature,
        },
    )


# ==========================================================================
# Envelope builders
# ==========================================================================


def finding_source(index=1, content_hash=CONTENT_HASH):
    return {
        "kind": "FINDING",
        "id": f"f-{index}",
        "immutable_version_id": f"f-{index}@v1",
        "content_hash": content_hash,
    }


def unreviewed():
    return {"state": "RETAINED_IMPORT_UNREVIEWED"}


def node_op(node_id="finding-1", *, source=None, properties=None, lifecycle="default"):
    op = {"node_id": node_id, "source": source or finding_source()}
    if properties is not None:
        op["properties"] = properties
    if lifecycle == "default":
        op["lifecycle"] = unreviewed()
    elif lifecycle is not None:
        op["lifecycle"] = lifecycle
    return op


def driver_op(node_id="driver-1", index=1):
    return {
        "node_id": node_id,
        "source": {
            "kind": "FINDING_DRIVER",
            "id": f"d-{index}",
            "immutable_version_id": f"d-{index}@v1",
            "content_hash": DRIVER_HASH,
        },
    }


def receipt_op(node_id="receipt-1", index=1):
    return {
        "node_id": node_id,
        "source": {
            "kind": "FINDING_RECEIPT",
            "id": f"r-{index}",
            "immutable_version_id": f"r-{index}@v1",
            "content_hash": RECEIPT_HASH,
        },
    }


def edge_op(
    edge_id="e-1", relation="DERIVED_FROM", from_id="finding-1", to_id="driver-1", properties=None
):
    op = {
        "edge_id": edge_id,
        "relation": relation,
        "from_node_id": from_id,
        "to_node_id": to_id,
    }
    if properties is not None:
        op["properties"] = properties
    return op


def envelope(
    *,
    operation_id="op-1",
    ordinal=0,
    node_ops=None,
    edge_ops=None,
    client_slug=CLIENT,
    namespace=GRAPH,
    engagement_id=ENGAGEMENT,
    origin="DIRECT_IMPORT",
    cursor=None,
    cutoff_id="cutoff-1",
    manifest=MANIFEST_HASH,
    **extra,
):
    body = {
        "schema_version": "projection.v2",
        "client_slug": client_slug,
        "namespace": namespace,
        "engagement_id": engagement_id,
        "origin": origin,
        "cursor": cursor
        if cursor is not None
        else {"import_manifest_hash": manifest, "batch_ordinal": ordinal},
        "operation_id": operation_id,
        "cutoff": {"cutoff_id": cutoff_id},
        "node_ops": node_ops if node_ops is not None else [node_op()],
        "edge_ops": edge_ops or [],
    }
    body.update(extra)
    return body


def apply(client, body, **kwargs):
    return post(client, PATH, body, **kwargs)


# ==========================================================================
# Canonical hash: an independent oracle
# ==========================================================================

# Written by hand from the recipe documented in app/models/projection.py, not
# produced by calling the code under test. A non-Python emitter that follows that
# recipe produces exactly these bytes. If the implementation drifts from the
# documentation, this fails.
GOLDEN_CANONICAL = (
    '{"client_slug":"unit-tenant",'
    '"cursor":{"batch_ordinal":0,"event_id":null,"import_manifest_hash":"' + MANIFEST_HASH + '"},'
    '"cutoff":{"cutoff_at":null,"cutoff_id":"cutoff-1"},'
    '"edge_ops":[],'
    '"engagement_id":"eng-1",'
    '"namespace":"client_unit_tenant",'
    '"node_ops":[{'
    '"lifecycle":{"is_rejected":false,"legacy_validated_flag":false,'
    '"merged_into_id":null,"state":"RETAINED_IMPORT_UNREVIEWED","superseded_by_id":null,'
    '"validated_at":null,"validated_by":null,"validated_content_hash":null,'
    '"validated_version_id":null,"validator_subject_id":null},'
    '"node_id":"finding-1","op":"MERGE_NODE","properties":{"importance":"critical"},'
    '"source":{"content_hash":"' + CONTENT_HASH + '","id":"f-1",'
    '"immutable_version_id":"f-1@v1","kind":"FINDING"}}],'
    '"operation_id":"op-golden","origin":"DIRECT_IMPORT","schema_version":"projection.v2"}'
)


def golden_envelope():
    return envelope(
        operation_id="op-golden",
        node_ops=[node_op(properties={"importance": "critical"})],
    )


def test_the_documented_canonical_recipe_reproduces_the_service_digest():
    """A non-Python emitter following the docstring gets the same bytes."""
    model = ProjectionEnvelopeV2.model_validate(golden_envelope())
    assert json.loads(GOLDEN_CANONICAL) == model.canonical_payload()
    expected = hashlib.sha256(GOLDEN_CANONICAL.encode("utf-8")).hexdigest()
    assert model.canonical_hash() == expected


# Non-ASCII is emitted as UTF-8, not as a \u escape. Without a case containing a
# non-ASCII character, ensure_ascii could flip and no digest would move.
GOLDEN_UNICODE_CANONICAL = GOLDEN_CANONICAL.replace(
    '"properties":{"importance":"critical"}',
    '"properties":{"statement":"café naïve über ß"}',
).replace('"operation_id":"op-golden"', '"operation_id":"op-unicode"')


def test_the_canonical_form_emits_non_ascii_as_utf8_rather_than_escaping_it():
    model = ProjectionEnvelopeV2.model_validate(
        envelope(
            operation_id="op-unicode",
            node_ops=[node_op(properties={"statement": "café naïve über ß"})],
        )
    )
    assert json.loads(GOLDEN_UNICODE_CANONICAL) == model.canonical_payload()
    assert "\\u" not in GOLDEN_UNICODE_CANONICAL
    expected = hashlib.sha256(GOLDEN_UNICODE_CANONICAL.encode("utf-8")).hexdigest()
    assert model.canonical_hash() == expected


def test_canonical_hash_ignores_key_order_and_insignificant_whitespace(store, client):
    body = golden_envelope()
    first = apply(client, body)
    assert first.status_code == 200, first.text

    def shuffle(value):
        if isinstance(value, dict):
            return {k: shuffle(value[k]) for k in reversed(list(value))}
        if isinstance(value, list):
            return [shuffle(v) for v in value]
        return value

    reordered = json.loads(json.dumps(shuffle(body), indent=4))
    second = apply(client, reordered)
    assert second.status_code == 200, second.text
    assert second.json()["replay"] is True


def test_canonical_hash_changes_on_every_class_of_semantic_difference():
    base = golden_envelope()
    variants = {
        "baseline": base,
        "client": {**base, "client_slug": "other-tenant", "namespace": "client_other_tenant"},
        "engagement": {**base, "engagement_id": "eng-2"},
        "namespace": {**base, "namespace": "client_unit_tenant_other"},
        "operation": {**base, "operation_id": "op-2"},
        "ordinal": {**base, "cursor": {"import_manifest_hash": MANIFEST_HASH, "batch_ordinal": 1}},
        "manifest": {**base, "cursor": {"import_manifest_hash": "2f" * 32, "batch_ordinal": 0}},
        "cutoff": {**base, "cutoff": {"cutoff_id": "cutoff-2"}},
        "node_id": {
            **base,
            "node_ops": [node_op("finding-2", properties={"importance": "critical"})],
        },
        "properties": {**base, "node_ops": [node_op(properties={"importance": "high"})]},
        "source_version": {
            **base,
            "node_ops": [
                node_op(
                    source={**finding_source(), "immutable_version_id": "f-1@v2"},
                    properties={"importance": "critical"},
                )
            ],
        },
        "content_hash": {
            **base,
            "node_ops": [
                node_op(
                    source=finding_source(content_hash="a1" * 32),
                    properties={"importance": "critical"},
                )
            ],
        },
        "lifecycle": {
            **base,
            "node_ops": [
                node_op(
                    properties={"importance": "critical"},
                    lifecycle={"state": "REJECTED", "is_rejected": True},
                )
            ],
        },
        "extra_op": {
            **base,
            "node_ops": [node_op(properties={"importance": "critical"}), driver_op()],
        },
    }
    digests = {
        name: ProjectionEnvelopeV2.model_validate(body).canonical_hash()
        for name, body in variants.items()
    }
    assert len(set(digests.values())) == len(digests), digests


def test_non_semantic_fields_are_excluded_from_the_hash():
    base = golden_envelope()
    plain = ProjectionEnvelopeV2.model_validate(base).canonical_hash()
    timed = ProjectionEnvelopeV2.model_validate(
        {**base, "emitted_at": "2026-07-24T10:00:00+00:00"}
    ).canonical_hash()
    assert plain == timed
    stated = ProjectionEnvelopeV2.model_validate({**base, "envelope_hash": plain})
    assert stated.canonical_hash() == plain


def test_a_supplied_envelope_hash_that_does_not_match_is_refused(store, client):
    response = apply(client, {**golden_envelope(), "envelope_hash": "0" * 64})
    assert response.status_code == 422
    assert store.mutating_queries() == []


def test_operation_order_is_semantic():
    a = ProjectionEnvelopeV2.model_validate(
        envelope(node_ops=[node_op("finding-1"), node_op("finding-2", source=finding_source(2))])
    )
    b = ProjectionEnvelopeV2.model_validate(
        envelope(node_ops=[node_op("finding-2", source=finding_source(2)), node_op("finding-1")])
    )
    assert a.canonical_hash() != b.canonical_hash()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_is_refused_rather_than_hashing_as_null(value):
    """Pydantic serializes NaN and Infinity to null in JSON mode while the raw
    value would still be written, so three different envelopes would share one
    digest and a correction would be misread as an exact replay."""
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op(properties={"severity": value})])
        )
    assert "NaN or Infinity" in str(excinfo.value)


def test_an_integer_a_javascript_emitter_could_not_reproduce_is_refused():
    ok = ProjectionEnvelopeV2.model_validate(
        envelope(node_ops=[node_op(properties={"n": pm.MAX_EXACT_INTEGER})])
    )
    assert ok.canonical_hash()
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op(properties={"n": pm.MAX_EXACT_INTEGER + 1})])
        )
    assert "exactly representable" in str(excinfo.value)


def test_a_string_that_is_not_unicode_nfc_is_refused_not_rewritten():
    """One envelope, two emitters, one digest: an NFD source column would
    otherwise hash differently from the identical NFC text."""
    nfd = unicodedata.normalize("NFD", "café")
    assert nfd != "café"
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op(properties={"statement": nfd})])
        )
    assert "NFC" in str(excinfo.value)
    ProjectionEnvelopeV2.model_validate(
        envelope(node_ops=[node_op(properties={"statement": "café"})])
    )


# ==========================================================================
# The perimeter and tenant binding
# ==========================================================================


def test_the_lane_refuses_to_run_unless_the_signed_perimeter_is_closed(store, client, monkeypatch):
    for mode in ("off", "optional"):
        monkeypatch.setattr(settings, "graphiti_auth_mode", mode, raising=False)
        response = client.post(PATH, json=envelope())
        assert response.status_code == 503, (mode, response.text)
        assert "GRAPHITI_AUTH_MODE=required" in response.json()["detail"]
    assert store.mutating_queries() == []


def test_an_unsigned_request_is_refused_and_a_retargeted_one_is_too(store, client):
    assert client.post(PATH, json=envelope()).status_code == 401
    assert apply(client, envelope(), header_client="other-tenant").status_code == 403
    assert store.mutating_queries() == []


def test_a_signed_request_passes_the_perimeter_exactly_once(store, client):
    """Two perimeter checks would consume the request nonce twice and 409."""
    response = apply(client, envelope())
    assert response.status_code == 200, response.text


def test_the_route_reads_the_principal_and_not_only_the_body(store, client, monkeypatch):
    async def foreign_principal(request, expected_scope):
        return auth.GraphPrincipal(scope=expected_scope, client_slug="someone-else")

    monkeypatch.setattr(auth, "verify_request", foreign_principal)
    response = client.post(PATH, json=envelope())
    assert response.status_code == 403
    assert store.mutating_queries() == []


def test_only_the_derived_tenant_graph_is_touched_and_every_row_carries_the_tenant(store, client):
    assert apply(client, envelope()).status_code == 200
    assert set(store.selected) == {GRAPH}
    for node in store.nodes.values():
        assert node["props"]["group_id"] == GRAPH
        assert node["props"]["engagement_id"] == ENGAGEMENT


def test_a_namespace_that_disagrees_with_the_derived_graph_is_refused_not_coerced(store, client):
    response = apply(client, envelope(namespace="client_somewhere_else"))
    assert response.status_code == 422
    assert "refuses a namespace mismatch" in response.json()["detail"]
    assert store.mutating_queries() == []


def test_the_same_operation_id_in_two_tenants_stays_two_independent_receipts(store, client):
    assert apply(client, envelope(operation_id="shared")).status_code == 200
    other = envelope(
        operation_id="shared", client_slug="other-tenant", namespace="client_other_tenant"
    )
    assert apply(client, other).status_code == 200
    assert len(store.receipts) == 2
    assert len(store.nodes) == 2


def test_the_same_operation_id_in_two_engagements_stays_two_independent_receipts(store, client):
    assert apply(client, envelope(operation_id="shared")).status_code == 200
    assert apply(client, envelope(operation_id="shared", engagement_id="eng-2")).status_code == 200
    assert len(store.receipts) == 2


def test_one_engagement_cannot_overwrite_another_engagements_row(store, client):
    """Finding IDs are per-engagement in the relational store, so identity that
    ignored the engagement would let one engagement replace another's row."""
    first = envelope(node_ops=[node_op("finding-1", properties={"owner": "a"})])
    assert apply(client, first).status_code == 200
    second = envelope(
        operation_id="op-2",
        ordinal=1,
        engagement_id="eng-2",
        node_ops=[node_op("finding-1", properties={"owner": "b"})],
    )
    assert apply(client, second).status_code == 200
    assert sorted(node["props"]["owner"] for node in store.nodes.values()) == ["a", "b"]


def test_a_foreign_schema_version_is_refused(store, client):
    assert apply(client, envelope(schema_version="projection.v3")).status_code == 422
    assert store.mutating_queries() == []


def test_the_receipt_read_surface_is_read_only_and_tenant_scoped(store, client):
    assert apply(client, envelope(operation_id="op-a", ordinal=0)).status_code == 200
    assert apply(client, envelope(operation_id="op-b", ordinal=1)).status_code == 200
    before = store.snapshot()

    response = post(
        client,
        RECEIPTS_PATH,
        {
            "client_slug": CLIENT,
            "engagement_id": ENGAGEMENT,
            "import_manifest_hash": MANIFEST_HASH,
        },
    )
    assert response.status_code == 200, response.text
    receipts = response.json()["receipts"]
    assert sorted(r["operation_id"] for r in receipts) == ["op-a", "op-b"]
    assert {r["batch_ordinal"] for r in receipts} == {0, 1}
    assert all(r["status"] == "APPLIED" for r in receipts)
    assert store.snapshot() == before

    scoped = post(client, RECEIPTS_PATH, {"client_slug": CLIENT, "engagement_id": "eng-other"})
    assert scoped.json()["receipts"] == []


# ==========================================================================
# Origin and cursor
# ==========================================================================


def test_direct_import_and_outcome_event_cursor_spaces_cannot_alias():
    direct = ProjectionEnvelopeV2.model_validate(envelope(ordinal=4))
    event = ProjectionEnvelopeV2.model_validate(
        envelope(origin="OUTCOME_EVENT", cursor={"event_id": f"{MANIFEST_HASH}:4"})
    )
    assert direct.cursor_key() != event.cursor_key()
    assert direct.cursor_key().startswith("DIRECT_IMPORT|")
    assert event.cursor_key().startswith("OUTCOME_EVENT|")
    smuggled = ProjectionEnvelopeV2.model_validate(
        envelope(origin="OUTCOME_EVENT", cursor={"event_id": 'DIRECT_IMPORT|{"batch_ordinal":4}'})
    )
    assert smuggled.cursor_key() != direct.cursor_key()


@pytest.mark.parametrize(
    "origin,cursor",
    [
        (
            "DIRECT_IMPORT",
            {"import_manifest_hash": MANIFEST_HASH, "batch_ordinal": 0, "event_id": "e"},
        ),
        ("DIRECT_IMPORT", {"event_id": "e"}),
        ("DIRECT_IMPORT", {"import_manifest_hash": MANIFEST_HASH}),
        ("OUTCOME_EVENT", {"event_id": "e", "batch_ordinal": 0}),
        ("OUTCOME_EVENT", {"import_manifest_hash": MANIFEST_HASH, "batch_ordinal": 0}),
        ("OUTCOME_EVENT", {}),
    ],
)
def test_the_wrong_cursor_fields_for_the_declared_origin_are_refused(origin, cursor):
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(envelope(origin=origin, cursor=cursor))


def test_outcome_event_lane_is_refused_in_this_phase_but_keeps_its_own_cursor_space(store, client):
    response = apply(client, envelope(origin="OUTCOME_EVENT", cursor={"event_id": "evt-1"}))
    assert response.status_code == 422
    assert "OUTCOME_EVENT" in response.json()["detail"]
    assert store.mutating_queries() == []


def test_one_cursor_position_is_claimed_by_at_most_one_operation(store, client):
    """A re-cut batch under a fresh operation ID would otherwise apply the same
    ordinal of a frozen manifest twice with no signal."""
    assert apply(client, envelope(operation_id="op-a", ordinal=7)).status_code == 200
    before = store.snapshot()

    second = envelope(
        operation_id="op-b", ordinal=7, node_ops=[node_op("finding-9", source=finding_source(9))]
    )
    response = apply(client, second)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == projection.CONFLICT_CODE_CURSOR
    assert response.json()["detail"]["claimed_by"] == ["op-a"]
    assert store.snapshot() == before


def test_the_same_manifest_cannot_be_projected_against_two_frozen_cutoffs(store, client):
    assert apply(client, envelope(operation_id="op-a", ordinal=0)).status_code == 200
    before = store.snapshot()
    response = apply(client, envelope(operation_id="op-b", ordinal=1, cutoff_id="cutoff-2"))
    assert response.status_code == 422
    assert "frozen cutoff" in response.json()["detail"]
    assert store.snapshot() == before


def test_a_different_manifest_may_carry_a_different_cutoff(store, client):
    assert apply(client, envelope(operation_id="op-a")).status_code == 200
    other = envelope(operation_id="op-b", manifest="2f" * 32, cutoff_id="cutoff-2", ordinal=0)
    assert apply(client, other).status_code == 200


def test_the_cursor_is_recorded_on_the_receipt_in_a_readable_form(store, client):
    receipt = apply(client, envelope(ordinal=3)).json()["receipt"]
    assert receipt["cursor_space"] == "DIRECT_IMPORT"
    assert receipt["import_manifest_hash"] == MANIFEST_HASH
    assert receipt["batch_ordinal"] == 3
    assert receipt["event_id"] is None


# ==========================================================================
# Phase scope: P2-prime admits, P3 refuses
# ==========================================================================


@pytest.mark.parametrize("kind", ["RACI_MATRIX", "RACI"])
def test_a_responsibility_chart_is_refused_until_its_lane_is_switched_on(kind, store, client):
    """The model admits the shape; the deployment has to authorize the write."""
    body = envelope(node_ops=[structure_op("chart-row", kind, version="rv-1", properties={"text": "Who does it."})])
    response = apply(client, body)
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_RACI" in json.dumps(response.json())
    assert store.mutating_queries() == []


@pytest.mark.parametrize("kind", ["WORKFLOW", "ROLE", "SYSTEM"])
def test_process_structure_is_refused_until_its_lane_is_switched_on(kind, store, client):
    """The model admits the shape; the deployment has to authorize the write."""
    body = envelope(node_ops=[node_op(source={**finding_source(), "kind": kind}, lifecycle=None)])
    response = apply(client, body)
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_PROCESS_STRUCTURE" in json.dumps(response.json())
    assert store.mutating_queries() == []


def test_a_p3_shaped_payload_cannot_ride_in_under_an_admitted_source_kind(store, client):
    """The phase gate cannot be a self-declared metadata field: what actually
    lands in the graph is what has to be constrained."""
    labelled = envelope(node_ops=[node_op()])
    labelled["node_ops"][0]["labels"] = ["SOP", "Workflow"]
    assert apply(client, labelled).status_code == 422
    assert store.nodes == {}

    still_p3 = envelope(
        node_ops=[node_op(), driver_op()],
        edge_ops=[edge_op(relation="OWNED_BY")],
    )
    response = apply(client, still_p3)
    assert response.status_code == 422
    assert "P3" in json.dumps(response.json())
    assert store.nodes == {}

    # A chart edge between two Finding rows is a chart edge all the same: it
    # meets the chart lane's switch, not the Finding lane's open door.
    raci = envelope(
        node_ops=[node_op(), driver_op()],
        edge_ops=[edge_op(relation="ACCOUNTABLE_FOR")],
    )
    response = apply(client, raci)
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_RACI" in json.dumps(response.json())
    assert store.nodes == {}


def test_the_node_type_label_is_a_function_of_the_declared_kind_not_of_caller_text(store, client):
    assert apply(client, envelope(node_ops=[node_op(), driver_op(), receipt_op()])).status_code == 200
    labels = sorted(tuple(sorted(node["labels"])) for node in store.nodes.values())
    assert labels == [
        ("Finding", pm.PROJECTION_NODE_LABEL),
        ("FindingDriver", pm.PROJECTION_NODE_LABEL),
        ("FindingReceipt", pm.PROJECTION_NODE_LABEL),
    ]
    assert "labels" not in pm.ProjectionNodeOp.model_fields


@pytest.mark.parametrize("relation", sorted(pm.P3_EDGE_RELATIONS | pm.P3_DEPENDENT_EDGE_RELATIONS))
def test_p3_relationship_types_are_refused_today(relation):
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op(), driver_op()], edge_ops=[edge_op(relation=relation)])
        )
    assert "phase" in str(excinfo.value)


def test_a_relation_outside_the_allowlist_is_refused_even_when_it_is_not_a_p3_relation():
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(
            envelope(
                node_ops=[node_op(), driver_op()],
                edge_ops=[edge_op(relation="INVENTED_BY_A_CALLER")],
            )
        )
    assert "projection allowlist" in str(excinfo.value)


def test_graphiti_owned_labels_cannot_be_written_because_labels_are_not_caller_supplied(store, client):
    """graphiti-core owns Entity, Episodic, Community and Saga on this driver. An
    allowlist rather than a denylist means a future graphiti release adding a
    label cannot silently widen this write surface."""
    assert apply(client, envelope()).status_code == 200
    written = set()
    for node in store.nodes.values():
        written.update(node["labels"])
    assert written == {pm.PROJECTION_NODE_LABEL, "Finding"}
    assert written.isdisjoint({"Entity", "Episodic", "Community", "Saga", "ProjectionReceipt"})


def test_every_declared_source_kind_is_classified_into_exactly_one_phase():
    kinds = set(pm.ProjectionSourceKind)
    phases = [pm.P2_PRIME_SOURCE_KINDS, pm.PROCESS_STRUCTURE_SOURCE_KINDS, pm.P3_SOURCE_KINDS, pm.SUMMARY_SOURCE_KINDS, pm.LAW_SOURCE_KINDS]
    assert frozenset().union(*phases) == kinds
    assert sum(len(phase) for phase in phases) == len(kinds)
    assert set(pm.SOURCE_KIND_NODE_LABEL) == kinds
    assert len(set(pm.SOURCE_KIND_NODE_LABEL.values())) == len(kinds)
    assert {pm.SOURCE_KIND_NODE_LABEL[k] for k in pm.P3_SOURCE_KINDS} <= pm.P3_NODE_LABELS
    assert pm.RECORD_RULE_SOURCE_KINDS == pm.PROCESS_STRUCTURE_SOURCE_KINDS | pm.P3_SOURCE_KINDS | pm.SUMMARY_SOURCE_KINDS | pm.LAW_SOURCE_KINDS
    assert pm.VERSIONED_RECORD_KINDS < pm.RECORD_RULE_SOURCE_KINDS
    assert pm.P3_SOURCE_KINDS < pm.VERSIONED_RECORD_KINDS
    # Every summary is a record of a version; the record set that points at them is not.
    assert pm.SUMMARY_SOURCE_KINDS - pm.VERSIONED_RECORD_KINDS == {pm.ProjectionSourceKind.RECORD_SET}
    # Every section of a law is a record of a version of that law; the law that points at them is not.
    assert pm.LAW_SOURCE_KINDS - pm.VERSIONED_RECORD_KINDS == {pm.ProjectionSourceKind.LAW_SOURCE}
    assert pm.POINTER_ANCHOR_KIND["current_law_version_id"] is pm.ProjectionSourceKind.LAW_SOURCE
    # Every pointer has its records and the one kind of node that carries it, and every
    # versioned record kind is put in force by exactly one pointer.
    assert set(pm.POINTER_RECORD_KINDS) == set(pm.POINTER_ANCHOR_KIND) == set(pm.CURRENT_VERSION_PROPERTIES)
    served = [kind for kinds_ in pm.POINTER_RECORD_KINDS.values() for kind in kinds_]
    assert sorted(served, key=lambda k: k.value) == sorted(pm.VERSIONED_RECORD_KINDS, key=lambda k: k.value)
    assert {pointer for pointer, anchor in pm.POINTER_ANCHOR_KIND.items() if anchor is pm.ProjectionSourceKind.RECORD_SET} == {"current_summary_version_id"}
    assert set(pm.RELATION_ENDPOINT_KINDS) == set(pm.ProjectionRelation)
    # Every relation belongs to exactly one lane, so each lane's switch covers its own edges.
    finding_relations = {pm.ProjectionRelation.DERIVED_FROM, pm.ProjectionRelation.ATTESTED_BY}
    lanes = [finding_relations, set(pm.PROCESS_STRUCTURE_RELATIONS), set(pm.RACI_RELATIONS), set(pm.SUMMARY_RELATIONS), set(pm.LAW_RELATIONS)]
    assert set().union(*lanes) == set(pm.ProjectionRelation)
    assert sum(len(lane) for lane in lanes) == len(pm.ProjectionRelation)
    # What is still refused by name is not also admitted.
    assert not (pm.P3_EDGE_RELATIONS | pm.P3_DEPENDENT_EDGE_RELATIONS) & {r.value for r in pm.ProjectionRelation}


def test_an_edge_must_join_the_endpoint_kinds_its_relation_declares(store, client):
    wrong_target = envelope(
        node_ops=[node_op("finding-1"), node_op("finding-2", source=finding_source(2))],
        edge_ops=[edge_op(to_id="finding-2")],
    )
    response = apply(client, wrong_target)
    assert response.status_code == 422
    assert "must end at a FINDING_DRIVER node" in response.json()["detail"]
    assert store.nodes == {}

    wrong_origin = envelope(
        node_ops=[driver_op("driver-1", 1), driver_op("driver-2", 2)],
        edge_ops=[edge_op(from_id="driver-1", to_id="driver-2")],
    )
    second = apply(client, wrong_origin)
    assert second.status_code == 422
    assert "must start at a FINDING node" in second.json()["detail"]
    assert store.nodes == {}


def test_the_p2_prime_allowlist_refuses_a_kind_that_no_phase_has_classified(monkeypatch):
    """The allowlist is not made redundant by the P3 denial: a source kind added
    to the enum later, and classified nowhere, must still be refused."""
    monkeypatch.setattr(pm, "P3_SOURCE_KINDS", frozenset())
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op(source={**finding_source(), "kind": "RACI"}, lifecycle=None)])
        )
    assert "projection allowlist" in str(excinfo.value)


def test_the_admitted_p2_prime_shape_applies(store, client):
    body = envelope(
        node_ops=[node_op(), driver_op(), receipt_op()],
        edge_ops=[
            edge_op("e-drv"),
            edge_op("e-rec", relation="ATTESTED_BY", to_id="receipt-1"),
        ],
    )
    response = apply(client, body)
    assert response.status_code == 200, response.text
    assert response.json()["nodes_applied"] == 3
    assert response.json()["edges_applied"] == 2


# ==========================================================================
# A2 lifecycle
# ==========================================================================


@pytest.mark.parametrize(
    "lifecycle",
    [
        {"state": "REJECTED", "merged_into_id": "f-9", "is_rejected": True},
        {"state": "RETAINED_HUMAN", "merged_into_id": "f-9"},
        {"state": "RETAINED_IMPORT_UNREVIEWED", "is_rejected": True},
        {"state": "MERGED"},
        {"state": "REJECTED"},
        {"state": "SUPERSEDED"},
    ],
)
def test_a_lifecycle_that_contradicts_the_ratified_precedence_is_refused(lifecycle):
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(envelope(node_ops=[node_op(lifecycle=lifecycle)]))


def test_named_human_validation_requires_actor_identity_and_a_content_binding():
    """A2 backfills to RETAINED_HUMAN only where actor identity and version and
    content binding are provable, so a free-text name is not enough."""
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(
            envelope(
                node_ops=[
                    node_op(
                        lifecycle={"state": "RETAINED_HUMAN", "validated_by": "batch importer v3"}
                    )
                ]
            )
        )
    message = str(excinfo.value)
    for required in (
        "validator_subject_id",
        "validated_at",
        "validated_content_hash",
        "validated_version_id",
    ):
        assert required in message


def test_a_validation_cannot_adjudicate_a_version_or_statement_it_did_not_see():
    good = {
        "state": "RETAINED_HUMAN",
        "validated_by": "A. Reviewer",
        "validator_subject_id": "subject-77",
        "validated_at": "2026-07-01T12:00:00+00:00",
        "validated_version_id": "f-1@v1",
        "validated_content_hash": CONTENT_HASH,
    }
    ProjectionEnvelopeV2.model_validate(envelope(node_ops=[node_op(lifecycle=good)]))

    with pytest.raises(Exception) as version_error:
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op(lifecycle={**good, "validated_version_id": "f-1@v0"})])
        )
    assert "validated_version_id" in str(version_error.value)

    with pytest.raises(Exception) as content_error:
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op(lifecycle={**good, "validated_content_hash": "0e" * 32})])
        )
    assert "did not see" in str(content_error.value)


def test_a_provable_validation_is_recorded_with_its_actor_and_binding(store, client):
    good = {
        "state": "RETAINED_HUMAN",
        "validated_by": "A. Reviewer",
        "validator_subject_id": "subject-77",
        "validated_at": "2026-07-01T12:00:00+00:00",
        "validated_version_id": "f-1@v1",
        "validated_content_hash": CONTENT_HASH,
    }
    assert apply(client, envelope(node_ops=[node_op(lifecycle=good)])).status_code == 200
    props = next(iter(store.nodes.values()))["props"]
    assert props["lifecycle_human_validated"] is True
    assert props["lifecycle_validator_subject_id"] == "subject-77"
    assert props["lifecycle_validated_content_hash"] == CONTENT_HASH
    assert props["lifecycle_legacy_validated_flag"] is False


def test_an_unreviewed_import_is_never_recorded_as_named_human_validation(store, client):
    assert apply(client, envelope()).status_code == 200
    props = next(iter(store.nodes.values()))["props"]
    assert props["lifecycle_state"] == "RETAINED_IMPORT_UNREVIEWED"
    assert props["lifecycle_retained"] is True
    assert props["lifecycle_human_validated"] is False
    assert not any(key.startswith("lifecycle_validat") for key in props)


def test_an_unprovable_legacy_validation_flag_survives_as_audit_metadata_only(store, client):
    body = envelope(
        node_ops=[
            node_op(
                lifecycle={"state": "RETAINED_IMPORT_UNREVIEWED", "legacy_validated_flag": True}
            )
        ]
    )
    assert apply(client, body).status_code == 200
    props = next(iter(store.nodes.values()))["props"]
    assert props["lifecycle_legacy_validated_flag"] is True
    assert props["lifecycle_human_validated"] is False
    assert props["lifecycle_state"] == "RETAINED_IMPORT_UNREVIEWED"


def test_a_non_retained_state_may_not_carry_named_human_validation():
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(
            envelope(
                node_ops=[
                    node_op(
                        lifecycle={
                            "state": "REJECTED",
                            "is_rejected": True,
                            "validated_by": "someone",
                        }
                    )
                ]
            )
        )


def test_only_declared_retained_states_are_marked_retained(store, client):
    cases = {
        "MERGED": {"state": "MERGED", "merged_into_id": "f-2"},
        "REJECTED": {"state": "REJECTED", "is_rejected": True},
        "SUPERSEDED": {"state": "SUPERSEDED", "superseded_by_id": "f-3"},
        "RETAINED_IMPORT_UNREVIEWED": {"state": "RETAINED_IMPORT_UNREVIEWED"},
    }
    for index, (state, lifecycle) in enumerate(cases.items()):
        body = envelope(
            operation_id=f"op-{index}",
            ordinal=index,
            node_ops=[
                node_op(f"finding-{index}", source=finding_source(index), lifecycle=lifecycle)
            ],
        )
        assert apply(client, body).status_code == 200, state
    retained = {
        node["props"]["lifecycle_state"]: node["props"]["lifecycle_retained"]
        for node in store.nodes.values()
    }
    assert retained == {
        "MERGED": False,
        "REJECTED": False,
        "SUPERSEDED": False,
        "RETAINED_IMPORT_UNREVIEWED": True,
    }


def test_a_finding_must_carry_a_lifecycle_and_a_non_finding_must_not():
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(envelope(node_ops=[node_op(lifecycle=None)]))
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[{**driver_op(), "lifecycle": unreviewed()}])
        )


def test_a_projected_row_cannot_lose_its_lifecycle_by_being_reprojected_as_another_kind(store, client):
    """SET n:Label only adds, so a retyped node would keep the Finding label
    while SET n = wiped every lifecycle property off it."""
    assert apply(client, envelope(node_ops=[node_op("shared-id")])).status_code == 200
    retyped = envelope(operation_id="op-2", ordinal=1, node_ops=[driver_op("shared-id")])
    response = apply(client, retyped)
    assert response.status_code == 422
    assert "different source kind" in response.json()["detail"]
    assert next(iter(store.nodes.values()))["props"]["lifecycle_state"] == "RETAINED_IMPORT_UNREVIEWED"


def test_a_projection_row_is_not_reachable_from_the_knowledge_map_query(store, client, monkeypatch):
    """Only declared retained states feed retrieval, and the existing map query
    is label agnostic, so it would otherwise return a REJECTED projection row,
    and the ledger, the moment this lane is used."""
    import falkordb

    captured: list[str] = []

    class _Result:
        result_set: list = []

    class _Graph:
        def query(self, statement, params=None, timeout=None):
            captured.append(statement)
            return _Result()

    class _DB:
        def __init__(self, **kwargs):
            pass

        def select_graph(self, name):
            return _Graph()

    monkeypatch.setattr(falkordb, "FalkorDB", _DB)
    response = post(
        client,
        "/graph/nodes-and-edges",
        {"client_slug": CLIENT},
        scope="search",
        secret=SEARCH_SECRET,
    )
    assert response.status_code == 200, response.text

    node_query = next(q for q in captured if "MATCH (n)" in q)
    assert f"NOT n:{pm.PROJECTION_NODE_LABEL}" in node_query
    assert f"NOT n:{pm.PROJECTION_RECEIPT_LABEL}" in node_query
    edge_query = next(q for q in captured if "-[r]->" in q)
    assert f"NOT a:{pm.PROJECTION_NODE_LABEL}" in edge_query
    assert f"NOT b:{pm.PROJECTION_NODE_LABEL}" in edge_query
    # A section of a law is a projected row under a base label of its own, and is kept off the map the same way.
    assert f"NOT n:{pm.LAW_SECTION_BASE_LABEL}" in node_query
    assert f"NOT a:{pm.LAW_SECTION_BASE_LABEL}" in edge_query
    assert f"NOT b:{pm.LAW_SECTION_BASE_LABEL}" in edge_query


# ==========================================================================
# Receipts: replay, conflict, resume, convergence
# ==========================================================================


def test_a_fresh_operation_applies_and_stores_a_durable_receipt(store, client):
    response = apply(client, envelope())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert (payload["applied"], payload["replay"], payload["resumed"]) == (True, False, False)
    assert payload["nodes_applied"] == 1
    receipt = payload["receipt"]
    assert receipt["status"] == "APPLIED"
    assert receipt["group_id"] == GRAPH
    assert receipt["cutoff_id"] == "cutoff-1"
    assert receipt["node_ops_requested"] == 1
    assert len(store.receipts) == 1


def test_the_receipt_identity_is_derived_deterministically_not_generated(store, client):
    apply(client, envelope())
    stored = next(iter(store.receipts.values()))
    expected = hashlib.sha256(f"{GRAPH}\x1f{ENGAGEMENT}\x1fop-1".encode("utf-8")).hexdigest()
    assert stored["receipt_id"] == expected


def test_the_receipt_is_written_pending_before_the_operations_are_applied(store, client):
    store.fail_node_apply_on = 1
    response = apply(client, envelope())
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "projection_failed"
    assert next(iter(store.receipts.values()))["status"] == "PENDING"
    assert store.nodes == {}


def test_an_exact_replay_returns_the_prior_receipt_and_applies_nothing(store, client):
    body = envelope()
    first = apply(client, body)
    assert first.status_code == 200
    before = store.snapshot()
    marker = len(store.queries)

    second = apply(client, body)
    assert second.status_code == 200
    payload = second.json()
    assert (payload["applied"], payload["replay"]) == (False, True)
    assert payload["receipt"] == first.json()["receipt"]
    assert store.snapshot() == before
    replayed = [q for q, _ in store.queries[marker:]]
    assert not [q for q in replayed if "MERGE" in q or "SET " in q]


def test_the_same_operation_id_with_a_different_envelope_hash_conflicts_and_applies_nothing(store, client):
    assert apply(client, envelope()).status_code == 200
    before = store.snapshot()
    response = apply(client, envelope(node_ops=[node_op(properties={"importance": "changed"})]))
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == projection.CONFLICT_CODE_OPERATION
    assert detail["stored_envelope_hash"] != detail["submitted_envelope_hash"]
    assert store.snapshot() == before


def test_an_operation_conflict_is_distinguishable_from_a_replayed_signature(store, client):
    """Both are 409. A retrying client must be able to tell 're-sign and resend'
    from 'your envelope is wrong'."""
    assert apply(client, envelope()).status_code == 200
    conflict = apply(client, envelope(node_ops=[node_op(properties={"x": "y"})]))
    assert conflict.status_code == 409
    assert isinstance(conflict.json()["detail"], dict)
    assert conflict.json()["detail"]["code"] == projection.CONFLICT_CODE_OPERATION


def test_a_receipt_claimed_concurrently_between_the_guarded_write_and_the_read(store, client):
    """This exercises the re-read branch, not the first read: the competing
    receipt appears during the guarded create, after the first read said absent."""
    store.competing_receipt = {
        "group_id": GRAPH,
        "engagement_id": ENGAGEMENT,
        "operation_id": "op-1",
        "envelope_hash": "9" * 64,
        "status": "PENDING",
    }
    response = apply(client, envelope())
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == projection.CONFLICT_CODE_OPERATION
    assert store.nodes == {}


@pytest.mark.parametrize("stored_hash", [None, "absent"])
def test_a_half_written_receipt_is_adopted_rather_than_poisoning_its_operation_id(
    stored_hash, store, client
):
    """A receipt with no envelope hash must not 409 forever: the operation ID is
    deterministic, so 'use a new one' is not available to the caller. A driver
    may report the missing property as absent or as an explicit null."""
    key = (
        "ProjectionReceipt",
        ("engagement_id", ENGAGEMENT),
        ("group_id", GRAPH),
        ("operation_id", "op-1"),
    )
    receipt = {"group_id": GRAPH, "engagement_id": ENGAGEMENT, "operation_id": "op-1"}
    if stored_hash is None:
        receipt["envelope_hash"] = None
    store.receipts[key] = receipt
    response = apply(client, envelope())
    assert response.status_code == 200, response.text
    assert response.json()["receipt"]["status"] == "APPLIED"


def test_the_guarded_create_never_overwrites_a_concurrent_writers_receipt(store, client):
    """The guarded property write is the concurrency primitive the whole ledger
    rests on: without it the loser of the race stamps its own hash over the
    winner's and both writers apply."""
    store.competing_receipt = {
        "group_id": GRAPH,
        "engagement_id": ENGAGEMENT,
        "operation_id": "op-1",
        "envelope_hash": "9" * 64,
        "status": "PENDING",
        "nodes_applied": 0,
    }
    response = apply(client, envelope())
    assert response.status_code == 409
    stored = next(iter(store.receipts.values()))
    assert stored["envelope_hash"] == "9" * 64
    assert store.nodes == {}


def test_a_receipt_that_vanishes_before_finalization_fails_loudly(store, client):
    store.drop_receipt_before_finalize = True
    response = apply(client, envelope())
    assert response.status_code == 500
    assert response.json()["detail"] == "projection receipt could not be finalized after applying"


def test_an_apply_interrupted_mid_flight_resumes_and_converges(store, client):
    body = envelope(node_ops=[node_op(), driver_op()])
    store.fail_node_apply_on = 2
    assert apply(client, body).status_code == 500
    assert next(iter(store.receipts.values()))["status"] == "PENDING"
    assert len(store.nodes) == 1

    store.fail_node_apply_on = None
    second = apply(client, body)
    assert second.status_code == 200, second.text
    assert second.json()["resumed"] is True
    assert len(store.nodes) == 2
    assert next(iter(store.receipts.values()))["status"] == "APPLIED"


def test_a_pending_receipt_records_progress_so_the_ledger_is_not_silent(store, client):
    store.fail_node_apply_on = 2
    assert apply(client, envelope(node_ops=[node_op(), driver_op()])).status_code == 500
    stored = next(iter(store.receipts.values()))
    assert stored["status"] == "PENDING"
    assert stored["nodes_applied"] == 1


def test_reprojecting_the_same_immutable_source_version_converges(store, client):
    ops = [node_op(), driver_op()]
    assert apply(client, envelope(operation_id="op-a", ordinal=0, node_ops=ops)).status_code == 200
    snapshot = copy.deepcopy(store.nodes)

    assert apply(client, envelope(operation_id="op-b", ordinal=1, node_ops=ops)).status_code == 200
    assert len(store.nodes) == len(snapshot)
    # No exclusion list. The row is a pure function of the source version and the
    # envelope's tenant scope, so a second projection is byte identical.
    assert store.nodes == snapshot


def test_a_reprojection_replaces_the_property_map_rather_than_merging_into_it(store, client):
    """The projection restates the whole row. A property dropped upstream must
    disappear from the graph, otherwise the disposable projection accumulates
    values the authoritative store no longer holds."""
    first = envelope(node_ops=[node_op(properties={"importance": "critical", "owner": "a"})])
    assert apply(client, first).status_code == 200
    assert next(iter(store.nodes.values()))["props"]["owner"] == "a"

    second = envelope(
        operation_id="op-2", ordinal=1, node_ops=[node_op(properties={"importance": "critical"})]
    )
    assert apply(client, second).status_code == 200
    props = next(iter(store.nodes.values()))["props"]
    assert "owner" not in props
    assert props["importance"] == "critical"


def test_a_reprojected_edge_also_replaces_its_property_map(store, client):
    nodes = [node_op(), driver_op()]
    first = envelope(node_ops=nodes, edge_ops=[edge_op(properties={"weight": 1, "note": "x"})])
    assert apply(client, first).status_code == 200
    assert next(iter(store.edges.values()))["props"]["note"] == "x"

    second = envelope(
        operation_id="op-2",
        ordinal=1,
        node_ops=nodes,
        edge_ops=[edge_op(properties={"weight": 1})],
    )
    assert apply(client, second).status_code == 200
    assert len(store.edges) == 1
    assert "note" not in next(iter(store.edges.values()))["props"]


def test_projected_rows_carry_no_operation_scoped_or_wall_clock_value(store, client):
    apply(client, envelope())
    props = next(iter(store.nodes.values()))["props"]
    forbidden = {
        "projection_operation_id",
        "projection_envelope_hash",
        "projection_cursor_key",
        "projection_cursor_space",
    }
    assert forbidden.isdisjoint(props)
    receipt = next(iter(store.receipts.values()))
    assert receipt["operation_id"] and receipt["envelope_hash"] and receipt["cursor_key"]


def test_a_new_source_version_updates_the_row_in_place_instead_of_duplicating(store, client):
    assert apply(client, envelope(operation_id="op-a")).status_code == 200
    updated = envelope(
        operation_id="op-b",
        ordinal=1,
        node_ops=[
            node_op(
                source={
                    **finding_source(),
                    "immutable_version_id": "f-1@v2",
                    "content_hash": "0a" * 32,
                }
            )
        ],
    )
    assert apply(client, updated).status_code == 200
    assert len(store.nodes) == 1
    assert next(iter(store.nodes.values()))["props"]["projection_source_version_id"] == "f-1@v2"


def test_every_write_is_a_merge_on_the_caller_supplied_deterministic_identity(store, client):
    apply(client, envelope(node_ops=[node_op(), driver_op()], edge_ops=[edge_op()]))
    applies = [q for q, _ in store.queries if q.startswith("UNWIND $rows") and "MERGE" in q]
    assert applies
    for statement in applies:
        assert "CREATE (" not in statement
        assert "uuid" in statement
        assert "group_id: $group_id" in statement
        assert "engagement_id: $engagement_id" in statement
    for statement in [q for q in applies if "]->" not in q]:
        # Identity is the constant projection label, never the type label, so a
        # node whose type changes cannot split into two rows sharing one uuid.
        assert f"MERGE (n:{pm.PROJECTION_NODE_LABEL} {{uuid: row.node_id" in statement


def test_an_edge_identity_cannot_be_repointed_leaving_a_contradictory_pair(store, client):
    nodes = [node_op(), driver_op("driver-1", 1), driver_op("driver-2", 2)]
    assert apply(client, envelope(node_ops=nodes, edge_ops=[edge_op()])).status_code == 200
    assert len(store.edges) == 1

    repointed = envelope(
        operation_id="op-2",
        ordinal=1,
        node_ops=nodes,
        edge_ops=[edge_op(to_id="driver-2")],
    )
    response = apply(client, repointed)
    assert response.status_code == 422
    assert "re-pointing an edge identity" in response.json()["detail"]
    assert len(store.edges) == 1


def test_the_edge_lane_cannot_bind_a_node_outside_the_projection_label_space(store, client):
    """graphiti hands episode uuids back to callers, so an unscoped MATCH would
    let a projection graft invented content onto extracted content."""
    apply(client, envelope(node_ops=[node_op(), driver_op()], edge_ops=[edge_op()]))
    for statement, _ in store.queries:
        if "]->" in statement and "MATCH" in statement:
            # Either both endpoints are matched as projected nodes, or (the
            # identity check, which starts from the edge index) both are read
            # off the edge and then held to the projection label space: the
            # base labels the relation's endpoint kinds carry.
            typed = re.search(r"\[e:(\w+) ", statement)
            if typed:
                start, end = pm.RELATION_ENDPOINT_BASE_LABELS[pm.ProjectionRelation(typed.group(1))]
                assert set((start, end)) <= set(pm.PROJECTION_BASE_LABELS)
                if statement.startswith("UNWIND $edge_ids"):
                    assert f"a:{start} AND b:{end}" in statement, statement
                else:
                    assert f"(a:{start} {{" in statement and f"(b:{end} {{" in statement, statement
            else:
                assert f":{pm.PROJECTION_NODE_LABEL} {{" in statement, statement
    resolves = [q for q, _ in store.queries if q.startswith("UNWIND $node_ids")]
    assert resolves
    for statement in resolves:
        assert any(f"MATCH (n:{label} {{uuid: wanted" in statement for label in pm.PROJECTION_BASE_LABELS), statement


def test_an_edge_endpoint_that_does_not_exist_is_refused_before_any_write(store, client):
    body = envelope(node_ops=[node_op()], edge_ops=[edge_op(to_id="driver-absent")])
    response = apply(client, body)
    assert response.status_code == 422
    assert "driver-absent" in response.json()["detail"]
    assert store.nodes == {}
    assert store.receipts == {}


def test_a_row_that_did_not_land_fails_the_read_back_and_leaves_the_receipt_pending(store, client):
    """The reconciliation reads the graph back and only counts rows carrying this
    operation's op hash, so it is not a restatement of the request."""
    store.skip_node_row = "driver-1"
    response = apply(client, envelope(node_ops=[node_op(), driver_op()]))
    assert response.status_code == 500
    assert "1/2 nodes" in response.json()["detail"]
    assert next(iter(store.receipts.values()))["status"] == "PENDING"


def test_the_read_back_actually_checks_the_operation_hash(store, client):
    apply(client, envelope(node_ops=[node_op(), driver_op()], edge_ops=[edge_op()]))
    verifies = [q for q, _ in store.queries if q.startswith("UNWIND $rows") and "MERGE" not in q]
    assert len(verifies) >= 2
    for statement in verifies:
        assert "projection_op_hash = row.op_hash" in statement


# ==========================================================================
# Source identity, per row
# ==========================================================================


def test_every_projected_row_carries_its_own_source_identity_not_the_batchs(store, client):
    """A bulk envelope carries many source records. One envelope-level source
    would stamp one record's ID, version and content hash onto every other row,
    and the spot probe would then verify nothing."""
    body = envelope(
        node_ops=[
            node_op("finding-1", source=finding_source(1, content_hash="11" * 32)),
            node_op("finding-2", source=finding_source(2, content_hash="22" * 32)),
            driver_op("driver-1", 1),
        ]
    )
    assert apply(client, body).status_code == 200
    by_uuid = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert by_uuid["finding-1"]["projection_source_id"] == "f-1"
    assert by_uuid["finding-1"]["projection_source_content_hash"] == "11" * 32
    assert by_uuid["finding-2"]["projection_source_id"] == "f-2"
    assert by_uuid["finding-2"]["projection_source_content_hash"] == "22" * 32
    assert by_uuid["driver-1"]["projection_source_id"] == "d-1"
    assert by_uuid["driver-1"]["projection_source_kind"] == "FINDING_DRIVER"
    for props in by_uuid.values():
        assert props["projection_cutoff_id"] == "cutoff-1"
        assert props["projection_schema_version"] == "projection.v2"


def test_the_receipt_records_a_digest_of_every_source_identity_not_one_of_them(store, client):
    body = envelope(node_ops=[node_op("finding-1"), node_op("finding-2", source=finding_source(2))])
    digest = apply(client, body).json()["receipt"]["source_identity_digest"]
    assert len(digest) == 64

    other = envelope(
        operation_id="op-2",
        ordinal=1,
        node_ops=[node_op("finding-1"), node_op("finding-3", source=finding_source(3))],
    )
    assert apply(client, other).json()["receipt"]["source_identity_digest"] != digest


def test_the_response_carries_per_operation_detail_for_spot_probes(store, client):
    body = envelope(node_ops=[node_op("finding-1"), node_op("finding-2", source=finding_source(2))])
    payload = apply(client, body).json()
    assert payload["operations_total"] == 2
    assert len(payload["operations_digest"]) == 64
    probes = {p["id"]: p for p in payload["operations"]}
    assert probes["finding-1"]["source_id"] == "f-1"
    assert probes["finding-2"]["source_content_hash"] == CONTENT_HASH
    samples = {s["node_id"]: s for s in payload["spot_probe"]["samples"]}
    assert samples["finding-2"]["source_immutable_version_id"] == "f-2@v1"
    assert pm.PROJECTION_NODE_LABEL in payload["spot_probe"]["node_cypher"]


def test_the_echoed_operation_list_is_bounded_but_still_covers_the_whole_batch(store, client):
    count = projection.MAX_ECHOED_OPERATIONS * 3
    ops = [node_op(f"finding-{i}", source=finding_source(i)) for i in range(count)]
    payload = apply(client, envelope(node_ops=ops)).json()
    assert payload["operations_total"] == count
    assert len(payload["operations"]) <= projection.MAX_ECHOED_OPERATIONS
    # The sample spans the batch, so a systematic error in the tail is visible.
    assert payload["operations"][-1]["id"] == f"finding-{count - 1}"
    assert payload["spot_probe"]["samples"][-1]["node_id"] == f"finding-{count - 1}"
    model = ProjectionEnvelopeV2.model_validate(envelope(node_ops=ops))
    assert payload["operations_digest"] == projection._operations_digest(
        projection._all_operation_hashes(model)
    )


# ==========================================================================
# P2 gate elements and operational posture
# ==========================================================================


def test_memory_headroom_is_probed_before_and_after_the_apply(store, client):
    payload = apply(client, envelope()).json()
    assert payload["memory_before"]["available"] is True
    assert payload["memory_after"]["available"] is True
    # Two distinct probes, and the second is the later reading.
    assert len(store.memory_probes) == 2
    assert payload["memory_before"]["used_memory_bytes"] == MEMORY_SERIES[0]["used_memory"]
    assert payload["memory_after"]["used_memory_bytes"] == MEMORY_SERIES[1]["used_memory"]
    assert payload["memory_after"]["used_memory_bytes"] != payload["memory_before"]["used_memory_bytes"]
    assert payload["memory_before"]["headroom_bytes"] == 9_000
    assert payload["memory_after"]["headroom_bytes"] == 7_000


def test_an_absent_memory_surface_is_tolerated_without_failing_the_request(store, client, monkeypatch):
    async def unavailable():
        raise RuntimeError("no redis here")

    monkeypatch.setattr(projection, "_read_memory_info", unavailable)
    payload = apply(client, envelope()).json()
    assert payload["memory_before"]["available"] is False
    assert payload["memory_after"]["available"] is False
    assert payload["applied"] is True


def test_projection_indices_are_created_once_per_graph(store, client):
    assert apply(client, envelope(operation_id="op-a", ordinal=0)).status_code == 200
    first_pass = list(store.indices)
    assert any(pm.PROJECTION_NODE_LABEL in q and "n.uuid" in q for q in first_pass)
    assert any(pm.PROJECTION_RECEIPT_LABEL in q and "cursor_key" in q for q in first_pass)
    assert apply(client, envelope(operation_id="op-b", ordinal=1)).status_code == 200
    assert store.indices == first_pass


def test_index_creation_failure_never_fails_the_request(store, client, monkeypatch):
    real_query = projection._query

    async def flaky(graph, statement, params=None):
        if statement.startswith("CREATE INDEX"):
            raise RuntimeError("this server does not accept that index syntax")
        return await real_query(graph, statement, params)

    monkeypatch.setattr(projection, "_query", flaky)
    assert apply(client, envelope()).status_code == 200


def test_a_driver_failure_is_not_echoed_back_to_the_caller(store, client, monkeypatch):
    async def leaky(graph, statement, params=None):
        raise RuntimeError("Cypher error near 'statement with tenant content'")

    monkeypatch.setattr(projection, "_query", leaky)
    response = apply(client, envelope())
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "projection_failed"
    assert len(detail["correlation_id"]) == 32
    assert "tenant content" not in json.dumps(detail)


def test_the_apply_is_chunked_rather_than_one_oversized_statement(store, client):
    count = projection.APPLY_CHUNK_ROWS + 25
    ops = [node_op(f"finding-{i}", source=finding_source(i)) for i in range(count)]
    assert apply(client, envelope(node_ops=ops)).status_code == 200
    applies = [
        params for q, params in store.queries if q.startswith("UNWIND $rows") and "MERGE" in q
    ]
    assert len(applies) == 2
    assert all(len(params["rows"]) <= projection.APPLY_CHUNK_ROWS for params in applies)


def test_the_property_budgets_are_enforced():
    too_many = {f"p{i}": i for i in range(MAX_PROPERTIES_PER_OPERATION + 1)}
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(envelope(node_ops=[node_op(properties=too_many)]))
    assert "at most" in str(excinfo.value)

    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(
            envelope(
                node_ops=[node_op(properties={"s": "x" * (pm.MAX_PROPERTY_STRING_LENGTH + 1)})]
            )
        )
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op(properties={"l": [1] * (pm.MAX_PROPERTY_LIST_LENGTH + 1)})])
        )


@pytest.mark.parametrize("value", [{"nested": 1}, [[1]], object()])
def test_a_property_value_that_is_not_a_flat_scalar_is_refused(value):
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(envelope(node_ops=[node_op(properties={"p": value})]))


def test_a_property_name_outside_the_identifier_grammar_is_refused():
    """The FalkorDB client interpolates parameters into the Cypher text and emits
    map keys unquoted, so this grammar is the injection boundary."""
    for bad in ["has space", "1leading", 'quote"key', "a" * 65, "semi;colon"]:
        with pytest.raises(Exception):
            ProjectionEnvelopeV2.model_validate(envelope(node_ops=[node_op(properties={bad: 1})]))


def test_hostile_property_values_survive_the_real_falkordb_parameter_serializer():
    """The escaping layer is client side in falkordb-py, so drive the real one
    rather than trusting the fake."""
    from falkordb.graph import Graph

    hostile = {
        "quote": 'a" RETURN 1 //',
        "backslash": "back\\slash",
        "brace": "}{",
    }
    ProjectionEnvelopeV2.model_validate(envelope(node_ops=[node_op(properties=hostile)]))
    header = Graph._build_params_header(None, {"rows": [{"props": hostile}]})
    assert header.startswith("CYPHER ")
    assert 'a\\" RETURN 1 //' in header
    assert "back\\\\slash" in header
    # Every caller-supplied quote is escaped, so no value can close its literal.
    assert re.search(r'(?<!\\)"', header[len("CYPHER rows=[{quote:") :]) is not None


def test_service_owned_properties_cannot_be_restated_by_the_caller():
    for owned in ["group_id", "uuid", "lifecycle_state", "projection_op_hash", "name_embedding"]:
        with pytest.raises(Exception):
            ProjectionEnvelopeV2.model_validate(envelope(node_ops=[node_op(properties={owned: "x"})]))


def test_the_segment_channel_and_any_unknown_field_are_refused():
    for extra in [{"include_segment": True}, {"segment_namespace": "x"}, {"unknown": 1}]:
        with pytest.raises(Exception):
            ProjectionEnvelopeV2.model_validate(envelope(**extra))
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[{**node_op(), "include_segment": True}])
        )


def test_the_operation_budget_bounds_one_request():
    ops = [
        node_op(f"n-{i}", source=finding_source(i)) for i in range(MAX_OPERATIONS_PER_ENVELOPE + 1)
    ]
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(envelope(node_ops=ops))
    assert "batch ordinals" in str(excinfo.value)
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(envelope(node_ops=[]))


def test_a_repeated_identity_inside_one_envelope_is_refused():
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(envelope(node_ops=[node_op(), node_op()]))
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(
            envelope(
                node_ops=[node_op(), driver_op()],
                edge_ops=[edge_op(), edge_op(relation="ATTESTED_BY")],
            )
        )


def test_an_edge_cannot_join_a_node_to_itself():
    with pytest.raises(Exception):
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op()], edge_ops=[edge_op(to_id="finding-1")])
        )


def test_the_canonical_envelope_size_is_bounded(monkeypatch):
    monkeypatch.setattr(pm, "MAX_CANONICAL_ENVELOPE_BYTES", 2048)
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(
            envelope(node_ops=[node_op(properties={"statement": "x" * 4096})])
        )
    assert "canonical envelope" in str(excinfo.value)


# ==========================================================================
# Isolation and dormancy
# ==========================================================================


def test_the_projection_module_opens_no_path_back_to_a_system_of_record():
    import inspect

    source = inspect.getsource(projection)
    for forbidden in [
        "graphiti_core",
        "graphiti_client",
        "add_episode",
        "psycopg",
        "sqlalchemy",
        "pg_boss",
        "httpx",
        "import requests",
        "anthropic",
        "include_segment",
        "segment_graph_name",
    ]:
        assert forbidden not in source, forbidden
    # The module's own top-level imports reach only the graph driver and this
    # service, so there is no reverse-write path one import hop away either.
    module_imports = set(re.findall(r"^(?:from|import)\s+([\w.]+)", source, re.MULTILINE))
    assert module_imports <= {
        "hashlib",
        "logging",
        "time",
        "uuid",
        "datetime",
        "typing",
        "fastapi",
        "fastapi.concurrency",
        "pydantic",
        "app.auth",
        "app.config",
        "app.graph_names",
        "app.models.projection",
    }, module_imports


def _walk_routes(node):
    """Yield every route, flattening whatever container shape the installed
    Starlette uses. Newer versions wrap an included router in a `_IncludedRouter`
    that carries no `.path` of its own, so walking `app.routes` one level deep is
    version-dependent and silently finds nothing."""
    for route in getattr(node, "routes", []) or []:
        yield route
        if getattr(route, "routes", None):
            yield from _walk_routes(route)


def _mounted_paths():
    """Prefer the OpenAPI document, which is public and stable across versions,
    and fall back to walking the route tree."""
    documented = set(app.openapi().get("paths", {}))
    return documented | {
        path for route in _walk_routes(app) if (path := getattr(route, "path", None))
    }


def test_adding_the_route_did_not_change_any_existing_route(client):
    paths = _mounted_paths()
    assert PATH in paths
    assert RECEIPTS_PATH in paths
    for existing in ["/ingest/episode", "/search/context", "/graph/nodes-and-edges", "/health"]:
        assert existing in paths


def test_p2_prime_is_a_separate_lane_from_the_source_anchored_compatibility_endpoint():
    """A8 names this endpoint `/ingest/projection/v2` precisely so that
    `/ingest/structured/v2`, a P1 artifact, can never be mistaken for
    authoritative exact-ID projection, and bars structured/v2 from P2-prime.

    An earlier version of this test asserted that `/ingest/structured/v2` did not
    exist at all. That was the wrong property and it broke the moment the P1 lane
    landed: structured/v2 is a legitimate sibling. What must hold is that the two
    are distinct paths and that this module registers only its own."""
    own_paths = {getattr(route, "path", None) for route in projection.router.routes}
    assert own_paths == {"/projection/v2", "/projection/v2/receipts"}
    assert "/structured/v2" not in own_paths
    assert PATH != "/ingest/structured/v2"


def test_the_route_is_mounted_under_the_ingest_scope_with_one_shared_dependency():
    """Asserted against this module's own APIRouter rather than the composed
    application. How FastAPI composes `app.routes` is a private detail that
    differs by version: on newer Starlette an included router becomes an opaque
    wrapper that exposes no resolved route object, so reaching into the app is
    not portable. `projection.router` is the object this module owns and its
    `.routes` are ordinary APIRoutes on every supported version.

    The behavioural counterpart, that one shared principal means verify_request
    runs once and the request nonce is consumed once, is proven end to end by
    test_a_signed_request_passes_the_perimeter_exactly_once. If the mount-level
    guard and the handler principal were ever distinct objects, the perimeter
    would run twice and the second call would answer 409 Replayed."""
    from app import main

    assert PATH in _mounted_paths()
    assert RECEIPTS_PATH in _mounted_paths()

    own = [r for r in projection.router.routes if getattr(r, "path", None) == "/projection/v2"]
    assert own, [getattr(r, "path", None) for r in projection.router.routes]
    calls = [d.call for d in own[0].dependant.dependencies]
    assert calls
    assert set(calls) == {projection.INGEST_PRINCIPAL}

    # main mounts the very same principal object, so the perimeter check is
    # declared once for the request rather than twice.
    assert main.projection.INGEST_PRINCIPAL is projection.INGEST_PRINCIPAL


def test_no_client_identity_or_planning_snapshot_is_hardcoded():
    from pathlib import Path

    root = Path(projection.__file__).resolve().parents[2]
    for relative in ["app/models/projection.py", "app/routers/projection.py"]:
        text = (root / relative).read_text(encoding="utf-8")
        assert "22,361" not in text
        assert "22361" not in text
        for slug in ["pbgc", "pokagon", "michigan"]:
            assert slug not in text.lower()


# ==========================================================================
# Process structure: the approved operating record, behind its own switch
# ==========================================================================

PV_HASH = "e6" * 32
# The canonical hash of the fixed envelope in the golden test below, computed on the code as it stood before the
# responsibility-chart vocabulary was added.
GOLDEN_STRUCTURE_HASH = "14c8116697124c437451a033edaf6b374b408ac830b20ae83c8c92971f0b1d49"


def structure_source(kind, source_id, version="pv-3"):
    return {"kind": kind, "id": source_id, "immutable_version_id": version, "content_hash": PV_HASH}


def structure_op(node_id, kind, *, version="pv-3", workflow="wf-1", properties=None, **extra):
    props = dict(properties or {})
    if workflow is not None:
        props.setdefault("workflow_node_id", workflow)
    props.update(extra)
    return {"node_id": node_id, "source": structure_source(kind, node_id, version), "properties": props}


def structure_envelope(**overrides):
    node_ops = overrides.pop(
        "node_ops",
        [
            structure_op(
                "wf-1", "WORKFLOW", version="wf-1@a", workflow=None,
                properties={"name": "Receiving", "current_map_version_id": "pv-3"},
            ),
            structure_op("pv-3", "PROCESS_VERSION", properties={"text": "Version 3 was approved."}),
            structure_op("pv-3:s1", "PROCESS_STEP", properties={"text": "Step 1 is a task.", "lane_label": "Clerk"}),
            structure_op("pv-3:s2", "PROCESS_STEP", properties={"text": "Step 2 is a decision."}),
            structure_op("role:clerk", "ROLE", version="clerk@a", workflow=None, properties={"name": "Clerk"}),
        ],
    )
    edge_ops = overrides.pop(
        "edge_ops",
        [
            edge_op("e-version", relation="VERSION_OF", from_id="pv-3", to_id="wf-1"),
            edge_op("e-has-1", relation="HAS_STEP", from_id="pv-3", to_id="pv-3:s1"),
            edge_op("e-has-2", relation="HAS_STEP", from_id="pv-3", to_id="pv-3:s2"),
            edge_op("e-next", relation="NEXT", from_id="pv-3:s1", to_id="pv-3:s2", properties={"condition": "the count matches"}),
            edge_op("e-who", relation="PERFORMED_BY", from_id="pv-3:s1", to_id="role:clerk"),
        ],
    )
    return envelope(node_ops=node_ops, edge_ops=edge_ops, **overrides)


@pytest.fixture
def lane_on(monkeypatch):
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", True, raising=False)


@pytest.fixture
def raci_on(monkeypatch):
    monkeypatch.setattr(settings, "projection_v2_allow_raci", True, raising=False)


def raci_envelope(**overrides):
    """An approved chart of two rows for workflow wf-1, the roles it names, and the workflow pointing at it."""
    node_ops = overrides.pop(
        "node_ops",
        [
            structure_op("chart-1", "RACI_MATRIX", version="rv-1", properties={"text": "Version 1 of the chart was approved."}),
            structure_op("chart-1:r1", "RACI", version="rv-1", properties={"text": "Step 6.1: Clerk is Responsible; Lead is Accountable.", "row_label": "6.1"}),
            structure_op("chart-1:r2", "RACI", version="rv-1", properties={"text": "Step 6.2: Clerk is Responsible and Accountable; Lead is Informed.", "sop_step_numbers": ["6.2"]}),
            structure_op("role:clerk", "ROLE", version="clerk@a", workflow=None, properties={"name": "Clerk"}),
            structure_op("role:lead", "ROLE", version="lead@a", workflow=None, properties={"name": "Lead"}),
            structure_op(
                "wf-1", "WORKFLOW", version="wf-1@b", workflow=None,
                properties={"name": "Receiving", "current_raci_version_id": "rv-1"},
            ),
        ],
    )
    edge_ops = overrides.pop(
        "edge_ops",
        [
            edge_op("e-chart", relation="CHART_OF", from_id="chart-1", to_id="wf-1"),
            edge_op("e-row-1", relation="HAS_ASSIGNMENT", from_id="chart-1", to_id="chart-1:r1"),
            edge_op("e-row-2", relation="HAS_ASSIGNMENT", from_id="chart-1", to_id="chart-1:r2"),
            edge_op("e-r1-r", relation="RESPONSIBLE_FOR", from_id="role:clerk", to_id="chart-1:r1"),
            edge_op("e-r1-a", relation="ACCOUNTABLE_FOR", from_id="role:lead", to_id="chart-1:r1"),
            edge_op("e-r2-r", relation="RESPONSIBLE_FOR", from_id="role:clerk", to_id="chart-1:r2"),
            edge_op("e-r2-a", relation="ACCOUNTABLE_FOR", from_id="role:clerk", to_id="chart-1:r2"),
            edge_op("e-r2-i", relation="INFORMED", from_id="role:lead", to_id="chart-1:r2"),
            edge_op("e-r1-c", relation="CONSULTED", from_id="role:lead", to_id="chart-1:r2"),
        ],
    )
    return envelope(node_ops=node_ops, edge_ops=edge_ops, **overrides)


def test_the_process_structure_shape_applies_when_its_lane_is_on(store, client, lane_on):
    response = apply(client, structure_envelope())
    assert response.status_code == 200, response.text
    assert response.json()["nodes_applied"] == 5
    assert response.json()["edges_applied"] == 5
    labels = sorted(tuple(sorted(node["labels"])) for node in store.nodes.values())
    assert labels == [
        ("ProcessStep", pm.PROJECTION_NODE_LABEL),
        ("ProcessStep", pm.PROJECTION_NODE_LABEL),
        ("ProcessVersion", pm.PROJECTION_NODE_LABEL),
        (pm.PROJECTION_NODE_LABEL, "Role"),
        (pm.PROJECTION_NODE_LABEL, "Workflow"),
    ]


def test_which_version_is_in_force_is_a_pointer_on_the_workflow_not_a_flag_on_every_row(store, client, lane_on):
    assert apply(client, structure_envelope()).status_code == 200
    by_id = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert by_id["wf-1"]["current_map_version_id"] == "pv-3"
    assert {uuid: props.get("workflow_node_id") for uuid, props in by_id.items()} == {
        "wf-1": None, "pv-3": "wf-1", "pv-3:s1": "wf-1", "pv-3:s2": "wf-1", "role:clerk": None,
    }
    # The version a record belongs to is the service's own stamp from its source.
    assert {by_id[uuid]["projection_source_version_id"] for uuid in ("pv-3", "pv-3:s1", "pv-3:s2")} == {"pv-3"}
    assert not [props for props in by_id.values() if "record_current" in props or "record_state" in props]


def test_replacing_a_version_is_one_write_to_the_workflow_node(store, client, lane_on):
    assert apply(client, structure_envelope()).status_code == 200
    rows_before = {key: copy.deepcopy(node) for key, node in store.nodes.items() if node["props"]["uuid"] != "wf-1"}
    flip = structure_envelope(
        operation_id="op-2",
        ordinal=1,
        node_ops=[
            structure_op(
                "wf-1", "WORKFLOW", version="wf-1@b", workflow=None,
                properties={"name": "Receiving", "current_map_version_id": "pv-4"},
            )
        ],
        edge_ops=[],
    )
    assert apply(client, flip).status_code == 200
    by_id = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert by_id["wf-1"]["current_map_version_id"] == "pv-4"
    # Not one row of the old version was touched to take it out of force.
    assert {key: node for key, node in store.nodes.items() if node["props"]["uuid"] != "wf-1"} == rows_before


@pytest.mark.parametrize(
    "op, fragment",
    [
        (structure_op("pv-3:s1", "PROCESS_STEP", workflow=None), "must name its workflow node"),
        (structure_op("pv-3:s1", "PROCESS_STEP", workflow="   "), "must name its workflow node"),
        (structure_op("role:clerk", "ROLE", version="clerk@a", workflow="wf-1"), "only carried by records of an approved version"),
        (structure_op("pv-3:s1", "PROCESS_STEP", current_map_version_id="pv-3"), "only carried by a WORKFLOW node"),
        (
            structure_op("wf-1", "WORKFLOW", version="wf-1@a", workflow=None, current_sop_version_id=" "),
            "must be a non-empty version id",
        ),
        (structure_op("pv-3:s1", "PROCESS_STEP", properties={"text": "   "}), "must be a non-empty string"),
        (structure_op("pv-3:s1", "PROCESS_STEP", properties={"workflow_name": 12345}), "'workflow_name' must be a string"),
        (structure_op("pv-3:s1", "PROCESS_STEP", properties={"text_embedding": [0.1]}), "written by the projection service"),
        (structure_op("pv-3:s1", "PROCESS_STEP", properties={"text_emb_failed": "m:4"}), "written by the projection service"),
        (structure_op("pv-3:s1", "PROCESS_STEP", properties={"text_emb_failed_at": 5}), "written by the projection service"),
        (structure_op("pv-3:s1", "PROCESS_STEP", properties={"projection_source_version_id": "pv-9"}), "written by the projection service"),
    ],
)
def test_a_record_that_misstates_where_it_belongs_or_what_it_says_is_refused(op, fragment):
    with pytest.raises(Exception) as excinfo:
        ProjectionEnvelopeV2.model_validate(structure_envelope(node_ops=[op], edge_ops=[]))
    assert fragment in str(excinfo.value)


def test_the_new_vocabulary_adds_no_envelope_field_so_older_envelopes_hash_as_they_always_did():
    """A new envelope field would change the canonical form of every envelope
    already applied, and a resend of one would then conflict instead of replay.
    What a record needs to say rides in its property map."""
    assert "currency" not in pm.ProjectionNodeOp.model_fields
    assert set(pm.ProjectionNodeOp.model_fields) == {"op", "node_id", "source", "properties", "lifecycle"}


def test_a_structure_edge_must_join_the_kinds_its_relation_declares(store, client, lane_on):
    wrong = structure_envelope(
        edge_ops=[edge_op("e-bad", relation="NEXT", from_id="pv-3:s1", to_id="role:clerk")]
    )
    response = apply(client, wrong)
    assert response.status_code == 422
    assert "relation NEXT ends at a ROLE node; it must end at a PROCESS_STEP node" in response.json()["detail"]
    assert store.nodes == {}

    crossed = structure_envelope(
        node_ops=[
            structure_op("sop-1", "SOP", version="sop-1"),
            structure_op("pv-3", "PROCESS_VERSION"),
            structure_op("pv-3:s1", "PROCESS_STEP"),
        ],
        edge_ops=[edge_op("e-cross", relation="HAS_STEP", from_id="sop-1", to_id="pv-3:s1")],
    )
    second = apply(client, crossed)
    assert second.status_code == 422
    assert "relation HAS_STEP cannot join a SOP node to a PROCESS_STEP node" in second.json()["detail"]
    assert store.nodes == {}


def test_a_structure_edge_alone_is_refused_while_the_lane_is_off(store, client):
    """The switch reads edges too, so structure cannot be drawn between nodes
    that happen to exist."""
    assert ProjectionEnvelopeV2.model_validate(structure_envelope()).carries_process_structure()
    assert not ProjectionEnvelopeV2.model_validate(envelope()).carries_process_structure()
    response = apply(client, structure_envelope())
    assert response.status_code == 422
    assert store.mutating_queries() == []


def test_reprojecting_the_same_approved_version_converges(store, client, lane_on):
    assert apply(client, structure_envelope()).status_code == 200
    before = store.snapshot()
    again = apply(client, structure_envelope(operation_id="op-2", ordinal=1))
    assert again.status_code == 200
    nodes_before, edges_before, _ = before
    nodes_after, edges_after, _ = store.snapshot()
    assert nodes_after == nodes_before
    assert edges_after == edges_before


def test_a_responsibility_chart_stays_refused_with_only_the_structure_lane_on(store, client, lane_on):
    response = apply(client, raci_envelope())
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_RACI" in json.dumps(response.json())
    assert store.mutating_queries() == []


def test_the_edge_identity_check_is_answered_from_an_index_not_a_walk_of_every_edge(store, client, lane_on):
    """Matching an untyped edge by uuid walked every edge of the projection for
    each wanted id. Each relation has its own uuid index and the check asks
    them, one typed lookup per relation."""
    assert apply(client, structure_envelope()).status_code == 200
    lookups = [q for q, _ in store.queries if q.startswith("UNWIND $edge_ids")]
    assert {re.search(r"-\[e:(\w+) ", q).group(1) for q in lookups} == {r.value for r in pm.ProjectionRelation}
    assert all("-[e {uuid: wanted}]" not in q for q in lookups)
    for relation in pm.ProjectionRelation:
        assert f"CREATE INDEX FOR ()-[e:{relation.value}]-() ON (e.uuid)" in store.indices


def test_an_edge_id_already_bound_under_another_relation_is_still_found(store, client, lane_on):
    assert apply(client, structure_envelope()).status_code == 200
    moved = structure_envelope(
        operation_id="op-2",
        ordinal=1,
        edge_ops=[edge_op("e-next", relation="HAS_STEP", from_id="pv-3", to_id="pv-3:s2")],
    )
    response = apply(client, moved)
    assert response.status_code == 422
    assert "already exists" in response.json()["detail"]


def test_an_edge_is_verified_where_the_operation_put_it(store, client, lane_on):
    assert apply(client, structure_envelope()).status_code == 200
    (statement,) = {q for q, _ in store.queries if q.startswith("UNWIND $rows") and "]->" in q and "MERGE" not in q}
    assert "uuid: row.from_node_id" in statement and "uuid: row.to_node_id" in statement



def test_an_envelope_of_structure_edges_alone_is_refused_while_the_lane_is_off(store, client, lane_on, monkeypatch):
    """With the switch flipped back off, structure must not be redrawn between
    process nodes that are already stored."""
    assert apply(client, structure_envelope()).status_code == 200
    monkeypatch.setattr(settings, "projection_v2_allow_process_structure", False, raising=False)
    before = store.snapshot()
    edges_only = envelope(
        operation_id="op-edges",
        ordinal=1,
        node_ops=[],
        edge_ops=[edge_op("e-extra", relation="NEXT", from_id="pv-3:s2", to_id="pv-3:s1")],
    )
    assert ProjectionEnvelopeV2.model_validate(edges_only).carries_process_structure()
    response = apply(client, edges_only)
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_PROCESS_STRUCTURE" in json.dumps(response.json())
    assert store.snapshot() == before


def test_the_new_property_rules_do_not_reach_the_kinds_admitted_before_them(store, client):
    """An envelope of Finding kinds validates exactly as it always did, whatever
    it carries under names the process-structure rules now give a meaning."""
    body = envelope(
        node_ops=[node_op(properties={"text": "   ", "workflow_node_id": "wf-1", "workflow_name": 7})]
    )
    assert apply(client, body).status_code == 200


# ==========================================================================
# Responsibility charts: the approved chart and its rows, behind their own switch
# ==========================================================================


def test_an_approved_chart_applies_with_both_lanes_on_and_the_workflow_points_at_it(store, client, lane_on, raci_on):
    response = apply(client, raci_envelope())
    assert response.status_code == 200, response.json()
    assert response.json()["nodes_applied"] == 6 and response.json()["edges_applied"] == 9
    labels = {node["props"]["uuid"]: sorted(set(node["labels"]) - {pm.PROJECTION_NODE_LABEL}) for node in store.nodes.values()}
    assert labels == {
        "chart-1": ["RaciMatrix"], "chart-1:r1": ["RaciAssignment"], "chart-1:r2": ["RaciAssignment"],
        "role:clerk": ["Role"], "role:lead": ["Role"], "wf-1": ["Workflow"],
    }
    props = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert props["wf-1"]["current_raci_version_id"] == "rv-1"
    assert {props[n]["projection_source_version_id"] for n in ("chart-1", "chart-1:r1", "chart-1:r2")} == {"rv-1"}
    assert {props[n]["workflow_node_id"] for n in ("chart-1", "chart-1:r1", "chart-1:r2")} == {"wf-1"}
    assert props["chart-1:r2"]["sop_step_numbers"] == ["6.2"]


def test_sending_the_same_chart_again_replays_and_writes_nothing(store, client, lane_on, raci_on):
    assert apply(client, raci_envelope()).status_code == 200
    before = store.snapshot()
    again = apply(client, raci_envelope())
    assert again.status_code == 200 and again.json()["replay"] is True and again.json()["applied"] is False
    assert store.snapshot() == before


def test_the_chart_lane_needs_the_structure_lane_too(store, client, raci_on):
    response = apply(client, raci_envelope())
    assert response.status_code == 422
    assert store.mutating_queries() == []
    # Rows alone carry no process-structure kind, so only the chart lane's own check can stop them:
    # with the chart switch on and the structure switch off they must still be refused.
    rows_only = envelope(node_ops=[structure_op("chart-1:r1", "RACI", version="rv-1", properties={"text": "Who does it."})])
    response = apply(client, rows_only)
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_PROCESS_STRUCTURE" in json.dumps(response.json()) and "PROJECTION_V2_ALLOW_RACI" in json.dumps(response.json())
    assert store.mutating_queries() == []


def test_every_pointer_serves_its_own_kinds_and_every_record_kind_has_a_pointer():
    assert tuple(pm.POINTER_RECORD_KINDS) == pm.CURRENT_VERSION_PROPERTIES
    served = [kind for kinds in pm.POINTER_RECORD_KINDS.values() for kind in kinds]
    assert len(served) == len(set(served)) and set(served) == pm.VERSIONED_RECORD_KINDS
    assert pm.POINTER_RECORD_KINDS["current_raci_version_id"] == pm.P3_SOURCE_KINDS


def test_an_envelope_admitted_before_the_chart_lane_hashes_as_it_always_did():
    """A fixed envelope and the hash it had before this vocabulary existed. A change to what an older
    envelope hashes to would make every stored receipt a conflict."""
    fixed = {
        "schema_version": "projection.v2", "client_slug": "north-dock", "namespace": "client_north_dock", "engagement_id": "eng-1",
        "origin": "DIRECT_IMPORT", "cursor": {"import_manifest_hash": "ab" * 32, "batch_ordinal": 0, "event_id": None},
        "operation_id": "process-records:map:" + "ab" * 16 + ":0", "cutoff": {"cutoff_id": "approved:" + "ab" * 16, "cutoff_at": None},
        "node_ops": [
            {"op": "MERGE_NODE", "node_id": "wf-1", "source": {"kind": "WORKFLOW", "id": "wf-1", "immutable_version_id": "wf-1@a", "content_hash": "e6" * 32},
             "properties": {"name": "Receiving", "current_map_version_id": "pv-3"}, "lifecycle": None},
            {"op": "MERGE_NODE", "node_id": "pv-3:s1", "source": {"kind": "PROCESS_STEP", "id": "s1", "immutable_version_id": "pv-3", "content_hash": "e6" * 32},
             "properties": {"workflow_node_id": "wf-1", "text": "Step 1 is a task."}, "lifecycle": None},
        ],
        "edge_ops": [],
    }
    assert pm.ProjectionEnvelopeV2.model_validate(fixed).canonical_hash() == GOLDEN_STRUCTURE_HASH


def test_a_workflow_that_only_points_at_a_chart_is_a_chart_write(store, client, lane_on):
    """The pointer is what puts a chart in force, so it cannot be written past the chart lane's switch."""
    pointer_only = envelope(node_ops=[structure_op("wf-1", "WORKFLOW", version="wf-1@c", workflow=None, properties={"name": "Receiving", "current_raci_version_id": "rv-1"})])
    response = apply(client, pointer_only)
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_RACI" in json.dumps(response.json())
    assert store.mutating_queries() == []
    without = envelope(node_ops=[structure_op("wf-1", "WORKFLOW", version="wf-1@d", workflow=None, properties={"name": "Receiving", "current_map_version_id": "pv-3"})])
    assert apply(client, without).status_code == 200


def test_a_chart_row_must_say_which_workflow_it_belongs_to_and_only_a_workflow_points_at_a_chart(store, client, lane_on, raci_on):
    adrift = raci_envelope()
    del adrift["node_ops"][1]["properties"]["workflow_node_id"]
    response = apply(client, adrift)
    assert response.status_code == 422 and "must name its workflow node" in json.dumps(response.json())

    misplaced = raci_envelope()
    misplaced["node_ops"][1]["properties"]["current_raci_version_id"] = "rv-1"
    response = apply(client, misplaced)
    assert response.status_code == 422 and "only carried by a WORKFLOW node" in json.dumps(response.json())

    blank = raci_envelope()
    blank["node_ops"][1]["properties"]["text"] = "   "
    response = apply(client, blank)
    assert response.status_code == 422 and "must be a non-empty string" in json.dumps(response.json())
    assert store.nodes == {}


@pytest.mark.parametrize(
    "edge, message",
    [
        (("RESPONSIBLE_FOR", "chart-1:r1", "chart-1:r2"), "relation RESPONSIBLE_FOR starts at a RACI node; it must start at a ROLE node"),
        (("ACCOUNTABLE_FOR", "role:clerk", "chart-1"), "relation ACCOUNTABLE_FOR ends at a RACI_MATRIX node; it must end at a RACI node"),
        (("HAS_ASSIGNMENT", "chart-1", "role:clerk"), "relation HAS_ASSIGNMENT ends at a ROLE node; it must end at a RACI node"),
        (("CHART_OF", "chart-1:r1", "wf-1"), "relation CHART_OF starts at a RACI node; it must start at a RACI_MATRIX node"),
        (("PERFORMED_BY", "chart-1:r1", "role:clerk"), "relation PERFORMED_BY starts at a RACI node; it must start at a PROCESS_STEP or SOP_STEP node"),
    ],
)
def test_a_chart_edge_must_join_the_kinds_its_relation_declares(edge, message, store, client, lane_on, raci_on):
    relation, from_id, to_id = edge
    body = raci_envelope()
    body["edge_ops"].append(edge_op("e-wrong", relation=relation, from_id=from_id, to_id=to_id))
    response = apply(client, body)
    assert response.status_code == 422
    assert message in json.dumps(response.json())
    assert store.nodes == {}


def test_a_chart_envelope_reports_which_lanes_it_belongs_to():
    chart = pm.ProjectionEnvelopeV2.model_validate(raci_envelope())
    structure = pm.ProjectionEnvelopeV2.model_validate(structure_envelope())
    rows_only = pm.ProjectionEnvelopeV2.model_validate(envelope(node_ops=[structure_op("chart-1:r1", "RACI", version="rv-1", properties={"text": "Who does it."})]))
    assert [chart.carries_raci(), chart.carries_process_structure()] == [True, True]
    assert [structure.carries_raci(), structure.carries_process_structure()] == [False, True]
    assert [rows_only.carries_raci(), rows_only.carries_process_structure()] == [True, False]


def test_the_structure_lane_is_as_it_was_with_the_chart_lane_on(store, client, lane_on, raci_on):
    assert apply(client, structure_envelope()).status_code == 200
    by_id = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert by_id["wf-1"].get("current_raci_version_id") is None and by_id["wf-1"]["current_map_version_id"] == "pv-3"


# ==========================================================================
# Summaries: statements counted by code from an engagement's approved records
# ==========================================================================


@pytest.fixture
def summaries_on(monkeypatch):
    monkeypatch.setattr(settings, "projection_v2_allow_summaries", True, raising=False)


COUNTED_AT = 1_790_000_000


def record_set_envelope(**overrides):
    """Two summaries of an engagement's approved records and the engagement's record set pointing at them."""
    node_ops = overrides.pop(
        "node_ops",
        [
            structure_op("set-1:dept:receiving", "RECORD_SET_SUMMARY", version="sum-1", workflow="set-1", properties={"text": "The Receiving department has 3 approved SOPs: A; B; C.", "count": 3, "names": ["A", "B", "C"]}),
            structure_op("set-1:months", "RECORD_SET_SUMMARY", version="sum-1", workflow="set-1", properties={"text": "SOPs approved by month: March 2026: 2; April 2026: 1."}),
            structure_op("set-1", "RECORD_SET", version="set-1@a", workflow=None, properties={"name": "Approved records", "current_summary_version_id": "sum-1", "counted_at": COUNTED_AT}),
        ],
    )
    edge_ops = overrides.pop(
        "edge_ops",
        [
            edge_op("e-sum-1", relation="SUMMARIZES", from_id="set-1:dept:receiving", to_id="set-1"),
            edge_op("e-sum-2", relation="SUMMARIZES", from_id="set-1:months", to_id="set-1"),
        ],
    )
    return envelope(node_ops=node_ops, edge_ops=edge_ops, **overrides)


SUMMARY_ALONE = {
    "a summary alone": lambda: envelope(node_ops=[structure_op("set-1:months", "RECORD_SET_SUMMARY", version="sum-1", workflow="set-1", properties={"text": "By month: 2."})]),
    "a record set alone": lambda: envelope(node_ops=[structure_op("set-1", "RECORD_SET", version="set-1@a", workflow=None, properties={"name": "Approved records"})]),
    "a record set that points at summaries": lambda: envelope(node_ops=[structure_op("set-1", "RECORD_SET", version="set-1@a", workflow=None, properties={"current_summary_version_id": "sum-1", "counted_at": COUNTED_AT})]),
    "a summary edge alone": lambda: envelope(edge_ops=[edge_op("e-sum-1", relation="SUMMARIZES", from_id="set-1:months", to_id="set-1")]),
}
SUMMARY_BUILDS = [record_set_envelope, *SUMMARY_ALONE.values()]
SUMMARY_IDS = ["with its record set", *SUMMARY_ALONE]


@pytest.mark.parametrize("build", SUMMARY_BUILDS, ids=SUMMARY_IDS)
def test_a_summary_is_refused_until_its_lane_is_switched_on(build, store, client, lane_on):
    response = apply(client, build())
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_SUMMARIES" in json.dumps(response.json())
    assert store.mutating_queries() == []


@pytest.mark.parametrize("build", SUMMARY_BUILDS, ids=SUMMARY_IDS)
def test_the_summary_lane_needs_the_structure_lane_too(build, store, client, summaries_on):
    # None of these carries a process-structure kind, so only the summary lane's own check can stop them.
    response = apply(client, build())
    assert response.status_code == 422
    said = json.dumps(response.json())
    assert "PROJECTION_V2_ALLOW_PROCESS_STRUCTURE" in said and "PROJECTION_V2_ALLOW_SUMMARIES" in said
    assert store.mutating_queries() == []


def test_summaries_apply_and_the_record_set_points_at_them(store, client, lane_on, summaries_on):
    response = apply(client, record_set_envelope())
    assert response.status_code == 200, response.json()
    assert response.json()["nodes_applied"] == 3 and response.json()["edges_applied"] == 2
    labels = {node["props"]["uuid"]: sorted(set(node["labels"]) - {pm.PROJECTION_NODE_LABEL}) for node in store.nodes.values()}
    assert labels == {"set-1": ["RecordSet"], "set-1:dept:receiving": ["RecordSetSummary"], "set-1:months": ["RecordSetSummary"]}
    props = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert props["set-1"]["current_summary_version_id"] == "sum-1" and props["set-1"]["counted_at"] == COUNTED_AT
    assert {props[n]["projection_source_version_id"] for n in ("set-1:dept:receiving", "set-1:months")} == {"sum-1"}
    assert {props[n]["workflow_node_id"] for n in ("set-1:dept:receiving", "set-1:months")} == {"set-1"}
    assert props["set-1:dept:receiving"]["count"] == 3 and props["set-1:dept:receiving"]["names"] == ["A", "B", "C"]


def test_sending_the_same_summaries_again_replays_and_writes_nothing(store, client, lane_on, summaries_on):
    assert apply(client, record_set_envelope()).status_code == 200
    before = store.snapshot()
    again = apply(client, record_set_envelope())
    assert again.status_code == 200 and again.json()["replay"] is True and again.json()["applied"] is False
    assert store.snapshot() == before


def test_a_summary_names_its_record_set_and_each_pointer_sits_only_on_its_own_kind_of_node(store, client, lane_on, summaries_on):
    adrift = record_set_envelope()
    del adrift["node_ops"][0]["properties"]["workflow_node_id"]
    response = apply(client, adrift)
    assert response.status_code == 422 and "must name its workflow node" in json.dumps(response.json())

    # A workflow cannot say which summaries are in force...
    on_a_workflow = envelope(node_ops=[structure_op("wf-1", "WORKFLOW", version="wf-1@b", workflow=None, properties={"name": "Receiving", "current_summary_version_id": "sum-1"})])
    response = apply(client, on_a_workflow)
    assert response.status_code == 422 and "'current_summary_version_id' says which version is in force and is only carried by a RECORD_SET node, not by a WORKFLOW node" in json.dumps(response.json())

    # ...a record set cannot say which map, SOP or chart is in force...
    for pointer in ("current_map_version_id", "current_sop_version_id", "current_raci_version_id"):
        on_a_set = record_set_envelope()
        on_a_set["node_ops"][2]["properties"][pointer] = "pv-3"
        response = apply(client, on_a_set)
        assert response.status_code == 422 and f"'{pointer}' says which version is in force and is only carried by a WORKFLOW node, not by a RECORD_SET node" in json.dumps(response.json())

    # ...and no summary carries a pointer at all.
    pointing = record_set_envelope()
    pointing["node_ops"][0]["properties"]["current_summary_version_id"] = "sum-1"
    response = apply(client, pointing)
    assert response.status_code == 422 and "only carried by a RECORD_SET node, not by a RECORD_SET_SUMMARY node" in json.dumps(response.json())

    # A record set is not a record of a version and does not name a workflow.
    named = record_set_envelope()
    named["node_ops"][2]["properties"]["workflow_node_id"] = "wf-1"
    response = apply(client, named)
    assert response.status_code == 422 and "is only carried by records of an approved version, not by a RECORD_SET node" in json.dumps(response.json())

    blank = record_set_envelope()
    blank["node_ops"][0]["properties"]["text"] = "   "
    response = apply(client, blank)
    assert response.status_code == 422 and "must be a non-empty string" in json.dumps(response.json())
    assert store.nodes == {}


@pytest.mark.parametrize(
    "extra, edge, message",
    [
        (("wf-1", "WORKFLOW", "wf-1@b", None), ("SUMMARIZES", "set-1:months", "wf-1"), "relation SUMMARIZES ends at a WORKFLOW node; it must end at a RECORD_SET node"),
        (("pv-3", "PROCESS_VERSION", "pv-3", "wf-1"), ("SUMMARIZES", "set-1:months", "pv-3"), "relation SUMMARIZES ends at a PROCESS_VERSION node; it must end at a RECORD_SET node"),
        (None, ("SUMMARIZES", "set-1", "set-1:months"), "relation SUMMARIZES starts at a RECORD_SET node; it must start at a RECORD_SET_SUMMARY node"),
        (None, ("SUMMARIZES", "set-1:months", "set-1:dept:receiving"), "relation SUMMARIZES ends at a RECORD_SET_SUMMARY node; it must end at a RECORD_SET node"),
        (("pv-3", "PROCESS_VERSION", "pv-3", "wf-1"), ("HAS_STEP", "pv-3", "set-1:months"), "relation HAS_STEP ends at a RECORD_SET_SUMMARY node"),
        (("wf-1", "WORKFLOW", "wf-1@b", None), ("VERSION_OF", "set-1", "wf-1"), "relation VERSION_OF starts at a RECORD_SET node; it must start at a PROCESS_VERSION node"),
    ],
)
def test_a_summary_edge_joins_a_summary_to_its_record_set_and_nothing_else(extra, edge, message, store, client, lane_on, summaries_on):
    relation, from_id, to_id = edge
    body = record_set_envelope()
    if extra:
        node_id, kind, version, workflow = extra
        body["node_ops"].append(structure_op(node_id, kind, version=version, workflow=workflow, properties={"name": "Receiving"} if kind == "WORKFLOW" else {"text": "Version 3 of the map was approved."}))
    body["edge_ops"].append(edge_op("e-wrong", relation=relation, from_id=from_id, to_id=to_id))
    response = apply(client, body)
    assert response.status_code == 422, response.json()
    assert message in json.dumps(response.json())
    assert store.nodes == {}


@pytest.mark.parametrize("counted_at", ["absent", None, 0, -5, 1.5, "1790000000", True, [COUNTED_AT]])
def test_a_record_set_that_points_at_summaries_must_say_when_they_were_counted(counted_at, store, client, lane_on, summaries_on):
    body = record_set_envelope()
    if counted_at == "absent":
        del body["node_ops"][2]["properties"]["counted_at"]
    else:
        body["node_ops"][2]["properties"]["counted_at"] = counted_at
    response = apply(client, body)
    assert response.status_code == 422, response.json()
    assert "must say when they were counted in 'counted_at'" in json.dumps(response.json())
    assert store.nodes == {}


def test_a_record_set_that_points_at_nothing_needs_no_time_and_takes_its_summaries_out_of_force(store, client, lane_on, summaries_on):
    assert apply(client, record_set_envelope()).status_code == 200
    without = envelope(operation_id="op-2", ordinal=1, node_ops=[structure_op("set-1", "RECORD_SET", version="set-1@b", workflow=None, properties={"name": "Approved records"})])
    response = apply(client, without)
    assert response.status_code == 200, response.json()
    props = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert props["set-1"].get("current_summary_version_id") is None and props["set-1"].get("counted_at") is None


@pytest.mark.parametrize("ahead, refused", [(0, False), (3600, False), (2 * 24 * 3600, True), ("milliseconds", True)])
def test_a_counting_time_far_ahead_of_the_clock_is_refused(ahead, refused, store, client, lane_on, summaries_on):
    # Milliseconds sent as seconds, or a clock set wrong, would be served as current for good.
    now = int(time.time())
    body = record_set_envelope()
    body["node_ops"][2]["properties"]["counted_at"] = now * 1000 if ahead == "milliseconds" else now + ahead
    response = apply(client, body)
    if refused:
        assert response.status_code == 422 and "more than a day ahead of this service's clock" in json.dumps(response.json())
        assert store.nodes == {}
    else:
        assert response.status_code == 200, response.json()


def test_a_summary_is_restated_in_place_and_keeps_its_vector_only_while_it_says_the_same_thing():
    from app.routers import projection as route

    assert route.RESTATED_IN_PLACE_LABELS == {"RecordSetSummary"}
    assert route.COUNTED_AT_AHEAD_SECONDS == 24 * 3600
    # Word for word: the node is found by its exact identity, its statement and vector are read before the
    # whole property map is replaced, and the vector is put back only if the statement has not changed.
    assert route._node_apply_query("RecordSetSummary") == (
        "UNWIND $rows AS row "
        # Held off a uuid a section of a law already holds, in the same statement (see _node_apply_query).
        "OPTIONAL MATCH (other:LawRecordNode {uuid: row.node_id, group_id: $group_id, engagement_id: $engagement_id}) "
        "WITH row WHERE other IS NULL "
        "MERGE (n:ProjectionNode {uuid: row.node_id, group_id: $group_id, engagement_id: $engagement_id}) "
        "WITH n, row, n.text AS said, n.text_embedding AS vector, n.text_emb_model AS model, "
        "n.text_emb_failed AS failed, n.text_emb_failed_at AS failed_at "
        "SET n:RecordSetSummary "
        "SET n = row.props "
        "WITH n, row, said, vector, model, failed, failed_at "
        "FOREACH (_ IN CASE WHEN vector IS NOT NULL AND said = row.props.text THEN [1] ELSE [] END | "
        "SET n.text_embedding = vector, n.text_emb_model = model) "
        # The mark that says the provider refused this statement is kept on the same terms, or a refused
        # statement would be asked again by every night's run.
        "FOREACH (_ IN CASE WHEN failed IS NOT NULL AND said = row.props.text THEN [1] ELSE [] END | "
        "SET n.text_emb_failed = failed, n.text_emb_failed_at = failed_at) "
        "RETURN count(n) AS applied"
    )
    # Every other kind is written exactly as it always was: restated whole, nothing carried over. A section of a
    # law is found by its exact identity under its own base label, never as a ProjectionNode.
    for label in sorted(set(pm.SOURCE_KIND_NODE_LABEL.values()) - route.RESTATED_IN_PLACE_LABELS):
        base, other = ("LawRecordNode", "ProjectionNode") if label == "LawSection" else ("ProjectionNode", "LawRecordNode")
        assert route._node_apply_query(label) == (
            "UNWIND $rows AS row "
            f"OPTIONAL MATCH (other:{other} {{uuid: row.node_id, group_id: $group_id, engagement_id: $engagement_id}}) "
            "WITH row WHERE other IS NULL "
            f"MERGE (n:{base} {{uuid: row.node_id, group_id: $group_id, engagement_id: $engagement_id}}) "
            f"SET n:{label} "
            "SET n = row.props "
            "RETURN count(n) AS applied"
        )


def _node_writes(store):
    """The type label of each node write the store received, in the order it received them."""
    return [re.search(r"SET n:(\w+) ", q).group(1) for q, _ in store.queries if q.startswith("UNWIND $rows AS row ") and "MERGE (n:" in q]


def _second_count(operation_id="op-2", ordinal=1):
    """The next run's counts: one statement restated, one no longer made, and the pointer moved on."""
    return envelope(
        operation_id=operation_id,
        ordinal=ordinal,
        node_ops=[
            structure_op("set-1:dept:receiving", "RECORD_SET_SUMMARY", version="sum-2", workflow="set-1", properties={"text": "The Receiving department has 2 approved SOPs: A; B.", "count": 2}),
            structure_op("set-1", "RECORD_SET", version="set-1@b", workflow=None, properties={"name": "Approved records", "current_summary_version_id": "sum-2", "counted_at": COUNTED_AT + 60}),
        ],
        edge_ops=[edge_op("e-sum-1", relation="SUMMARIZES", from_id="set-1:dept:receiving", to_id="set-1")],
    )


def _in_force(store):
    """The summaries whose version is the one the record set points at: what retrieval would serve."""
    props = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    pointer = props["set-1"].get("current_summary_version_id")
    return sorted(uuid for uuid, p in props.items() if p.get("projection_source_kind") == "RECORD_SET_SUMMARY" and p["projection_source_version_id"] == pointer)


def test_the_record_set_is_written_before_its_summaries_however_the_envelope_orders_them(store, client, lane_on, summaries_on):
    from app.routers import projection as route

    assert route.POINTER_FIRST_LABELS == {"RecordSet"}
    # Sent with the summaries first and the record set last...
    assert [op["source"]["kind"] for op in record_set_envelope()["node_ops"]] == ["RECORD_SET_SUMMARY", "RECORD_SET_SUMMARY", "RECORD_SET"]
    assert apply(client, record_set_envelope()).status_code == 200
    # ...and written the other way round.
    assert _node_writes(store) == ["RecordSet", "RecordSetSummary"]


def test_every_other_lane_is_still_written_in_the_order_it_was_sent(store, client, lane_on, raci_on, summaries_on):
    # A workflow's pointer goes last, after the records of the version it names: those are new nodes, and
    # must be there before anything points at them.
    sent = structure_envelope()
    order = list(dict.fromkeys(pm.SOURCE_KIND_NODE_LABEL[pm.ProjectionSourceKind(op["source"]["kind"])] for op in sent["node_ops"]))
    assert apply(client, sent).status_code == 200
    assert _node_writes(store) == order and "RecordSet" not in order


def test_a_write_that_stops_half_way_leaves_no_count_in_force_rather_than_an_outdated_one(store, client, lane_on, summaries_on):
    assert apply(client, record_set_envelope()).status_code == 200
    assert _in_force(store) == ["set-1:dept:receiving", "set-1:months"]
    # The next run no longer makes the by-month statement. Its write dies after the record set, before the summaries.
    store.fail_node_apply_on = store.node_apply_calls + 2
    assert apply(client, _second_count()).status_code == 500
    # The pointer has moved and nothing has been restated: nothing is served. With the summaries written
    # first, the by-month statement (the one that is no longer made) would be the only one still served.
    assert _in_force(store) == []
    # The run that follows writes everything, and only what it makes is in force.
    store.fail_node_apply_on = None
    assert apply(client, _second_count("op-3", 2)).status_code == 200
    assert _in_force(store) == ["set-1:dept:receiving"]


def test_a_statement_sent_twice_in_one_envelope_is_refused_before_anything_is_written(store, client, lane_on, summaries_on):
    # The in-place write reads what a node said before the batch, once. Two rows for one node in a batch could
    # put a vector back on words it was not made from, so the envelope may never carry them.
    twice = record_set_envelope()
    twice["node_ops"].insert(1, copy.deepcopy(twice["node_ops"][0]))
    twice["node_ops"][1]["properties"]["text"] = "The Receiving department has 4 approved SOPs: A; B; C; D."
    response = apply(client, twice)
    assert response.status_code == 422 and "node_id is repeated within the envelope" in json.dumps(response.json())
    assert store.mutating_queries() == []


def test_an_envelope_reports_whether_it_belongs_to_the_summary_lane():
    load = pm.ProjectionEnvelopeV2.model_validate
    record_set, structure, chart = load(record_set_envelope()), load(structure_envelope()), load(raci_envelope())
    assert [record_set.carries_summaries(), record_set.carries_process_structure(), record_set.carries_raci()] == [True, False, False]
    assert [structure.carries_summaries(), chart.carries_summaries()] == [False, False]
    for name, build in SUMMARY_ALONE.items():
        alone = load(build())
        assert alone.carries_summaries() is True, name
        assert alone.carries_process_structure() is False and alone.carries_raci() is False, name


def test_the_structure_and_chart_lanes_are_as_they_were_with_the_summary_lane_on(store, client, lane_on, raci_on, summaries_on):
    assert apply(client, structure_envelope()).status_code == 200
    assert apply(client, raci_envelope(operation_id="op-2", ordinal=1)).status_code == 200
    by_id = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert by_id["wf-1"].get("current_summary_version_id") is None
    assert not [uuid for uuid, props in by_id.items() if props.get("projection_source_kind", "").startswith("RECORD_SET")]


def test_the_records_of_a_map_an_sop_and_a_chart_are_put_in_force_by_the_pointers_they_always_were():
    assert pm.POINTER_RECORD_KINDS["current_map_version_id"] == {pm.ProjectionSourceKind.PROCESS_VERSION, pm.ProjectionSourceKind.PROCESS_STEP}
    assert pm.POINTER_RECORD_KINDS["current_sop_version_id"] == {
        pm.ProjectionSourceKind.SOP, pm.ProjectionSourceKind.SOP_STEP, pm.ProjectionSourceKind.SOP_SECTION,
        pm.ProjectionSourceKind.SOP_EXCEPTION, pm.ProjectionSourceKind.SOP_DEFINITION,
    }
    assert pm.POINTER_RECORD_KINDS["current_raci_version_id"] == pm.P3_SOURCE_KINDS
    assert pm.POINTER_RECORD_KINDS["current_summary_version_id"] == {pm.ProjectionSourceKind.RECORD_SET_SUMMARY}
    assert pm.SUMMARY_SOURCE_KINDS == {pm.ProjectionSourceKind.RECORD_SET, pm.ProjectionSourceKind.RECORD_SET_SUMMARY}


def test_the_summary_switch_is_off_unless_it_is_set():
    assert type(settings).model_fields["projection_v2_allow_summaries"].default is False


# ==========================================================================
# Law: the rules an engagement's work is done under, section by section
# ==========================================================================
# The law here is invented: a dock authority's rules, two sections of them.


@pytest.fixture
def law_on(monkeypatch):
    monkeypatch.setattr(settings, "projection_v2_allow_law", True, raising=False)


def law_envelope(version="law-v1", **overrides):
    """Two sections of one law, and the law pointing at the version they belong to."""
    node_ops = overrides.pop(
        "node_ops",
        [
            structure_op(f"law-1:{version}:2.01", "LAW_SECTION", version=version, workflow="law-1", properties={"text": "Section 2.01 Seals. Every container leaving the dock shall carry an unbroken seal.", "section": "2.01", "title": "Seals"}),
            structure_op(f"law-1:{version}:2.02", "LAW_SECTION", version=version, workflow="law-1", properties={"text": "Section 2.02 Records. The dock shall keep each seal number for three years.", "section": "2.02", "title": "Records"}),
            structure_op("law-1", "LAW_SOURCE", version=f"law-1@{version}", workflow=None, properties={"name": "Dock Authority Rules", "current_law_version_id": version}),
        ],
    )
    edge_ops = overrides.pop(
        "edge_ops",
        [
            edge_op(f"e-{version}-1", relation="SECTION_OF", from_id=f"law-1:{version}:2.01", to_id="law-1"),
            edge_op(f"e-{version}-2", relation="SECTION_OF", from_id=f"law-1:{version}:2.02", to_id="law-1"),
        ],
    )
    return envelope(node_ops=node_ops, edge_ops=edge_ops, **overrides)


LAW_ALONE = {
    "a section alone": lambda: envelope(node_ops=[structure_op("law-1:law-v1:2.01", "LAW_SECTION", version="law-v1", workflow="law-1", properties={"text": "Section 2.01 Seals."})]),
    "a law alone": lambda: envelope(node_ops=[structure_op("law-1", "LAW_SOURCE", version="law-1@a", workflow=None, properties={"name": "Dock Authority Rules"})]),
    "a law that points at a version": lambda: envelope(node_ops=[structure_op("law-1", "LAW_SOURCE", version="law-1@a", workflow=None, properties={"current_law_version_id": "law-v1"})]),
    "a section edge alone": lambda: envelope(edge_ops=[edge_op("e-law-1", relation="SECTION_OF", from_id="law-1:law-v1:2.01", to_id="law-1")]),
}
LAW_BUILDS = [law_envelope, *LAW_ALONE.values()]
LAW_IDS = ["with its law", *LAW_ALONE]


@pytest.mark.parametrize("build", LAW_BUILDS, ids=LAW_IDS)
def test_a_law_is_refused_until_its_lane_is_switched_on(build, store, client, lane_on, raci_on, summaries_on):
    # Every other lane is on: only the law lane's own switch can stop these.
    response = apply(client, build())
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_LAW" in json.dumps(response.json())
    assert store.mutating_queries() == []


def test_a_law_needs_only_its_own_switch(store, client, law_on):
    # The process-structure switch is off: a law is not hung on a process.
    response = apply(client, law_envelope())
    assert response.status_code == 200, response.json()


def test_a_law_applies_and_points_at_the_version_of_its_text_in_force(store, client, law_on):
    response = apply(client, law_envelope())
    assert response.status_code == 200, response.json()
    assert response.json()["nodes_applied"] == 3 and response.json()["edges_applied"] == 2
    labels = {node["props"]["uuid"]: sorted(node["labels"]) for node in store.nodes.values()}
    # A section carries a base label of its own and not ProjectionNode, so it is not counted in the process records'
    # keyword index at all; the law keeps ProjectionNode, which the in-force join finds it by.
    assert labels == {
        "law-1": ["LawSource", "ProjectionNode"],
        "law-1:law-v1:2.01": ["LawRecordNode", "LawSection"],
        "law-1:law-v1:2.02": ["LawRecordNode", "LawSection"],
    }
    props = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert props["law-1"]["current_law_version_id"] == "law-v1"
    assert {props[n]["projection_source_version_id"] for n in ("law-1:law-v1:2.01", "law-1:law-v1:2.02")} == {"law-v1"}
    assert {props[n]["workflow_node_id"] for n in ("law-1:law-v1:2.01", "law-1:law-v1:2.02")} == {"law-1"}
    assert "text" not in props["law-1:law-v1:2.01"]  # a section's words sit under their own name, out of the records' keyword index
    assert props["law-1:law-v1:2.01"]["law_text"] == "Section 2.01 Seals. Every container leaving the dock shall carry an unbroken seal."


def test_sending_the_same_law_again_replays_and_writes_nothing(store, client, law_on):
    assert apply(client, law_envelope()).status_code == 200
    before = store.snapshot()
    again = apply(client, law_envelope())
    assert again.status_code == 200 and again.json()["replay"] is True and again.json()["applied"] is False
    assert store.snapshot() == before


def test_a_new_version_of_a_law_is_new_sections_and_one_restated_pointer_and_the_old_text_stays_on_record(store, client, law_on):
    assert apply(client, law_envelope()).status_code == 200
    amended = law_envelope(version="law-v2", operation_id="op-2", ordinal=1)
    assert apply(client, amended).status_code == 200, "the amended law"
    props = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    # The old sections are still there, word for word; only the pointer moved.
    assert {"law-1:law-v1:2.01", "law-1:law-v1:2.02", "law-1:law-v2:2.01", "law-1:law-v2:2.02"} <= set(props)
    assert props["law-1"]["current_law_version_id"] == "law-v2"
    assert props["law-1:law-v1:2.01"]["projection_source_version_id"] == "law-v1"


def test_a_section_names_its_law_and_the_law_pointer_sits_only_on_a_law(store, client, law_on, lane_on):
    adrift = law_envelope()
    del adrift["node_ops"][0]["properties"]["workflow_node_id"]
    response = apply(client, adrift)
    assert response.status_code == 422 and "must name its workflow node" in json.dumps(response.json())

    # A workflow cannot say which text of a law is in force...
    on_a_workflow = envelope(node_ops=[structure_op("wf-1", "WORKFLOW", version="wf-1@b", workflow=None, properties={"name": "Receiving", "current_law_version_id": "law-v1"})])
    response = apply(client, on_a_workflow)
    assert response.status_code == 422 and "'current_law_version_id' says which version is in force and is only carried by a LAW_SOURCE node, not by a WORKFLOW node" in json.dumps(response.json())

    # ...a law cannot say which map, SOP, chart or summaries are in force...
    for pointer, anchor in (("current_map_version_id", "WORKFLOW"), ("current_sop_version_id", "WORKFLOW"), ("current_raci_version_id", "WORKFLOW"), ("current_summary_version_id", "RECORD_SET")):
        on_a_law = law_envelope()
        on_a_law["node_ops"][2]["properties"][pointer] = "pv-3"
        response = apply(client, on_a_law)
        assert response.status_code == 422 and f"'{pointer}' says which version is in force and is only carried by a {anchor} node, not by a LAW_SOURCE node" in json.dumps(response.json())

    # ...and no section carries a pointer at all.
    pointing = law_envelope()
    pointing["node_ops"][0]["properties"]["current_law_version_id"] = "law-v1"
    response = apply(client, pointing)
    assert response.status_code == 422 and "only carried by a LAW_SOURCE node, not by a LAW_SECTION node" in json.dumps(response.json())

    # A law is not a record of a version and does not name a workflow.
    named = law_envelope()
    named["node_ops"][2]["properties"]["workflow_node_id"] = "wf-1"
    response = apply(client, named)
    assert response.status_code == 422 and "is only carried by records of an approved version, not by a LAW_SOURCE node" in json.dumps(response.json())

    blank = law_envelope()
    blank["node_ops"][0]["properties"]["text"] = "   "
    response = apply(client, blank)
    assert response.status_code == 422 and "must be a non-empty string" in json.dumps(response.json())
    assert store.nodes == {}


@pytest.mark.parametrize(
    "extra, edge, message",
    [
        (("wf-1", "WORKFLOW", "wf-1@b", None), ("SECTION_OF", "law-1:law-v1:2.01", "wf-1"), "relation SECTION_OF ends at a WORKFLOW node; it must end at a LAW_SOURCE node"),
        (None, ("SECTION_OF", "law-1", "law-1:law-v1:2.01"), "relation SECTION_OF starts at a LAW_SOURCE node; it must start at a LAW_SECTION node"),
        (None, ("SECTION_OF", "law-1:law-v1:2.01", "law-1:law-v1:2.02"), "relation SECTION_OF ends at a LAW_SECTION node; it must end at a LAW_SOURCE node"),
        (("pv-3", "PROCESS_VERSION", "pv-3", "wf-1"), ("HAS_STEP", "pv-3", "law-1:law-v1:2.01"), "relation HAS_STEP ends at a LAW_SECTION node"),
    ],
)
def test_a_section_edge_joins_a_section_to_its_law_and_nothing_else(extra, edge, message, store, client, law_on, lane_on):
    relation, from_id, to_id = edge
    body = law_envelope()
    if extra:
        node_id, kind, version, workflow = extra
        body["node_ops"].append(structure_op(node_id, kind, version=version, workflow=workflow, properties={"name": "Receiving"} if kind == "WORKFLOW" else {"text": "Version 3 of the map was approved."}))
    body["edge_ops"].append(edge_op("e-wrong", relation=relation, from_id=from_id, to_id=to_id))
    response = apply(client, body)
    assert response.status_code == 422, response.json()
    assert message in json.dumps(response.json())
    assert store.mutating_queries() == []


def test_an_envelope_reports_whether_it_belongs_to_the_law_lane():
    load = pm.ProjectionEnvelopeV2.model_validate
    law, structure, chart, record_set = load(law_envelope()), load(structure_envelope()), load(raci_envelope()), load(record_set_envelope())
    assert [law.carries_law(), law.carries_process_structure(), law.carries_raci(), law.carries_summaries()] == [True, False, False, False]
    assert [structure.carries_law(), chart.carries_law(), record_set.carries_law()] == [False, False, False]
    for name, build in LAW_ALONE.items():
        alone = load(build())
        assert alone.carries_law() is True, name
        assert not (alone.carries_process_structure() or alone.carries_raci() or alone.carries_summaries()), name


def test_the_other_lanes_are_as_they_were_with_the_law_lane_on(store, client, lane_on, raci_on, summaries_on, law_on):
    assert apply(client, structure_envelope()).status_code == 200
    assert apply(client, raci_envelope(operation_id="op-2", ordinal=1)).status_code == 200
    by_id = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert by_id["wf-1"].get("current_law_version_id") is None
    assert not [uuid for uuid, props in by_id.items() if props.get("projection_source_kind", "").startswith("LAW_")]


def test_a_section_of_a_law_is_put_in_force_by_its_law_alone():
    assert pm.POINTER_RECORD_KINDS["current_law_version_id"] == {pm.ProjectionSourceKind.LAW_SECTION}
    assert pm.POINTER_ANCHOR_KIND["current_law_version_id"] is pm.ProjectionSourceKind.LAW_SOURCE
    assert pm.LAW_SOURCE_KINDS == {pm.ProjectionSourceKind.LAW_SOURCE, pm.ProjectionSourceKind.LAW_SECTION}
    assert pm.SOURCE_KIND_NODE_LABEL[pm.ProjectionSourceKind.LAW_SECTION] == "LawSection"
    assert pm.SOURCE_KIND_NODE_LABEL[pm.ProjectionSourceKind.LAW_SOURCE] == "LawSource"


def test_the_law_switch_is_off_unless_it_is_set():
    assert type(settings).model_fields["projection_v2_allow_law"].default is False


def _law_in_force(store):
    """The sections whose version is the one the law points at: what retrieval would serve."""
    props = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    pointer = props.get("law-1", {}).get("current_law_version_id")
    return sorted(uuid for uuid, p in props.items() if p.get("projection_source_kind") == "LAW_SECTION" and p["projection_source_version_id"] == pointer)


def test_a_law_is_written_after_its_sections_however_the_envelope_orders_them(store, client, law_on):
    from app.routers import projection as route

    assert route.POINTER_LAST_LABELS == {"LawSource"} and route.POINTER_FIRST_LABELS == {"RecordSet"}
    body = law_envelope()
    body["node_ops"] = [body["node_ops"][2], *body["node_ops"][:2]]  # the law first, its sections after
    assert [op["source"]["kind"] for op in body["node_ops"]] == ["LAW_SOURCE", "LAW_SECTION", "LAW_SECTION"]
    assert apply(client, body).status_code == 200
    assert _node_writes(store) == ["LawSection", "LawSource"]


def test_a_new_version_whose_write_stops_half_way_leaves_the_old_text_in_force(store, client, law_on):
    assert apply(client, law_envelope()).status_code == 200
    assert _law_in_force(store) == ["law-1:law-v1:2.01", "law-1:law-v1:2.02"]
    amended = law_envelope(version="law-v2", operation_id="op-2", ordinal=1)
    amended["node_ops"] = [amended["node_ops"][2], *amended["node_ops"][:2]]  # sent with the law first
    # The write dies on the second node write. With the law written last that is the law itself, after its sections
    # are in: the old text stays in force. (Were the law written first, as it was sent, the sections would be what
    # failed, after the pointer had moved, and nothing would be in force.)
    store.fail_node_apply_on = store.node_apply_calls + 2
    assert apply(client, amended).status_code == 500
    assert _law_in_force(store) == ["law-1:law-v1:2.01", "law-1:law-v1:2.02"]
    store.fail_node_apply_on = None
    assert apply(client, law_envelope(version="law-v2", operation_id="op-3", ordinal=2)).status_code == 200
    assert _law_in_force(store) == ["law-1:law-v2:2.01", "law-1:law-v2:2.02"]


@pytest.mark.parametrize(
    "law_part",
    [
        lambda: {"node_ops": [structure_op("law-1:law-v1:2.01", "LAW_SECTION", version="law-v1", workflow="law-1", properties={"text": "Section 2.01 Seals."})]},
        lambda: {"edge_ops": [edge_op("e-law-1", relation="SECTION_OF", from_id="law-1:law-v1:2.01", to_id="law-1")]},
    ],
    ids=["a section among process records", "a section edge among process edges"],
)
@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_a_law_hidden_in_a_process_envelope_is_refused_until_its_lane_is_on(law_part, where, store, client, lane_on):
    body = structure_envelope()
    part = law_part()
    for key in ("node_ops", "edge_ops"):
        at = {"first": 0, "middle": len(body[key]) // 2, "last": len(body[key])}[where]
        body[key][at:at] = part.get(key, [])
    response = apply(client, body)
    assert response.status_code == 422
    assert "PROJECTION_V2_ALLOW_LAW" in json.dumps(response.json())
    assert store.mutating_queries() == []


# A section of a law carries a base label of its own (LawRecordNode), never ProjectionNode: the process records'
# keyword index counts every node that carries ProjectionNode, whatever its properties, so a law's sections moved
# process scores while they carried it. Every path that finds a node by its base label finds a section by its own.


def _rows_sent(store, prefix):
    return [(q, [row.get("node_id") or row.get("edge_id") for row in p.get("rows", [])]) for q, p in store.queries if q.startswith(prefix)]


def test_a_section_is_written_verified_and_joined_to_its_law_under_its_own_base_label(store, client, law_on):
    response = apply(client, law_envelope())
    assert response.status_code == 200, response.json()
    sections = ["law-1:law-v1:2.01", "law-1:law-v1:2.02"]
    writes = {re.search(r"SET n:(\w+) ", q).group(1): q for q, _ in store.queries if q.startswith("UNWIND $rows AS row ") and "MERGE (n:" in q}
    assert "MERGE (n:LawRecordNode {uuid: row.node_id, group_id: $group_id, engagement_id: $engagement_id}) " in writes["LawSection"]
    # It names ProjectionNode only to stay off a uuid a process record holds.
    assert writes["LawSection"].startswith(
        "UNWIND $rows AS row OPTIONAL MATCH (other:ProjectionNode {uuid: row.node_id, group_id: $group_id, "
        "engagement_id: $engagement_id}) WITH row WHERE other IS NULL MERGE (n:LawRecordNode "
    )
    assert "MERGE (n:ProjectionNode {uuid: row.node_id, " in writes["LawSource"]
    # Read back under the base label each was written under, each node once.
    verifies = [(q, ids) for q, ids in _rows_sent(store, "UNWIND $rows AS row MATCH (n:")]
    assert [(q.split(" {", 1)[0], ids) for q, ids in verifies] == [
        ("UNWIND $rows AS row MATCH (n:LawRecordNode", sections),
        ("UNWIND $rows AS row MATCH (n:ProjectionNode", ["law-1"]),
    ]
    # The edge from a section to its law runs from a LawRecordNode to a ProjectionNode, written and read back so.
    edge_write = next(q for q, _ in store.queries if "MERGE (a)-[e:SECTION_OF" in q)
    assert "MATCH (a:LawRecordNode {uuid: row.from_node_id, " in edge_write and "MATCH (b:ProjectionNode {uuid: row.to_node_id, " in edge_write
    edge_verify = next(q for q, _ in store.queries if q.startswith("UNWIND $rows AS row MATCH (:"))
    assert edge_verify.startswith("UNWIND $rows AS row MATCH (:LawRecordNode {uuid: row.from_node_id, ")
    assert "(:ProjectionNode {uuid: row.to_node_id, " in edge_verify
    # The identity check holds SECTION_OF's ends to the same two labels.
    identity = next(q for q, _ in store.queries if q.startswith("UNWIND $edge_ids") and "[e:SECTION_OF " in q)
    assert "WHERE a:LawRecordNode AND b:ProjectionNode " in identity
    # The base label's own indexes are made beside the projection's, so a section's MERGE is an index probe.
    for prop in ("uuid", "group_id", "engagement_id"):
        assert f"CREATE INDEX FOR (n:LawRecordNode) ON (n.{prop})" in store.indices
    # A human can find a section with the probe offered beside it.
    probe = response.json()["spot_probe"]
    assert probe["law_node_cypher"].startswith("MATCH (n:LawRecordNode {group_id: $group_id, uuid: $uuid})")
    assert probe["law_edge_cypher"].startswith("MATCH (:LawRecordNode {group_id: $group_id})-[e {uuid: $uuid}]->()")


def test_a_section_that_did_not_land_under_its_own_base_label_is_not_counted_as_applied(store, client, law_on, monkeypatch):
    # Written under ProjectionNode by mistake, a section is not found by the read-back and the receipt stays PENDING.
    from app.routers import projection as route

    real = route._node_apply_query
    monkeypatch.setattr(route, "_node_apply_query", lambda label: real(label).replace("MERGE (n:LawRecordNode ", "MERGE (n:ProjectionNode "))
    response = apply(client, law_envelope())
    assert response.status_code == 500
    assert "verified 1/3 nodes" in json.dumps(response.json())
    assert [r["status"] for r in store.receipts.values()] == ["PENDING"]


def test_process_records_are_written_and_read_back_exactly_as_before_beside_the_law(store, client, lane_on, law_on):
    from app.routers import projection as route

    assert apply(client, structure_envelope()).status_code == 200
    # A write of a process envelope names the law's base label only to stay off a uuid a section holds (the check in
    # the same statement as the write, see _node_apply_query); nothing is written under it...
    guard = "OPTIONAL MATCH (other:LawRecordNode {uuid: row.node_id, group_id: $group_id, engagement_id: $engagement_id}) WITH row WHERE other IS NULL "
    for q in store.mutating_queries():
        assert pm.LAW_SECTION_BASE_LABEL not in q.replace(guard, ""), q
    verifies = [q for q, _ in store.queries if q.startswith("UNWIND $rows AS row MATCH") and "MERGE" not in q]
    assert len(set(verifies)) == 2 and set(verifies) <= {route.VERIFY_NODES_QUERY, route.VERIFY_EDGES_QUERY}
    # ...and the read-backs are word for word the statements they always were.
    assert route.VERIFY_NODES_QUERY == (
        "UNWIND $rows AS row MATCH (n:ProjectionNode {uuid: row.node_id, group_id: $group_id, "
        "engagement_id: $engagement_id}) WHERE n.projection_op_hash = row.op_hash RETURN count(DISTINCT n.uuid) AS verified"
    )
    assert route.VERIFY_EDGES_QUERY == (
        "UNWIND $rows AS row MATCH (:ProjectionNode {uuid: row.from_node_id, group_id: $group_id, "
        "engagement_id: $engagement_id})-[e {uuid: row.edge_id}]->(:ProjectionNode {uuid: row.to_node_id, "
        "group_id: $group_id, engagement_id: $engagement_id}) WHERE e.projection_op_hash = row.op_hash "
        "RETURN count(DISTINCT e.uuid) AS verified"
    )
    # The only statements that name it are reads (the lookup of ids under it, and the identity check of the one
    # relation that starts at a section) and the indexes on it.
    named = [q.replace(guard, "") for q, _ in store.queries if pm.LAW_SECTION_BASE_LABEL in q]
    reads = {route.RESOLVE_NODES_QUERIES[1], route._resolve_edges_query("SECTION_OF")}
    assert named and all(q in reads or q.startswith("CREATE INDEX") or pm.LAW_SECTION_BASE_LABEL not in q for q in named)
    assert "law_node_cypher" not in apply(client, structure_envelope()).json()["spot_probe"]


def test_a_section_and_a_process_record_can_never_share_an_id_either_way_round(store, client, lane_on, law_on):
    assert apply(client, structure_envelope()).status_code == 200
    # A section declared under a process step's id is refused, nothing written...
    body = law_envelope(operation_id="op-2", ordinal=1)
    body["node_ops"][0]["node_id"] = "pv-3:s1"
    body["edge_ops"][0]["from_node_id"] = "pv-3:s1"
    before = store.snapshot()
    response = apply(client, body)
    assert response.status_code == 422 and "pv-3:s1" in json.dumps(response.json())
    assert store.snapshot() == before
    # ...and a process step declared under a section's id is refused too, though the section is not a ProjectionNode.
    assert apply(client, law_envelope(operation_id="op-3", ordinal=2)).status_code == 200
    step = structure_envelope(operation_id="op-4", ordinal=3)
    step["node_ops"][2]["node_id"] = "law-1:law-v1:2.01"
    step["edge_ops"] = []
    before = store.snapshot()
    response = apply(client, step)
    assert response.status_code == 422 and "law-1:law-v1:2.01" in json.dumps(response.json())
    assert store.snapshot() == before


def test_a_section_left_under_projection_node_by_an_earlier_write_is_refused_not_duplicated(store, client, law_on):
    assert apply(client, law_envelope()).status_code == 200
    # As a section written before sections had a base label of their own would be stored.
    for key in [k for k in store.nodes if k[0] == pm.LAW_SECTION_BASE_LABEL]:
        node = store.nodes.pop(key)
        node["labels"] = {pm.PROJECTION_NODE_LABEL, "LawSection"}
        store.nodes[(pm.PROJECTION_NODE_LABEL,) + key[1:]] = node
    before = store.snapshot()
    response = apply(client, law_envelope(operation_id="op-2", ordinal=1))
    assert response.status_code == 422, response.json()
    detail = response.json()["detail"]
    assert "under a base label their kind does not carry" in detail
    # Only under the old label: the reason given is that one, and the delete names the old label.
    assert "'law-1:law-v1:2.01' is stored under ProjectionNode only, a label its kind does not carry" in detail
    assert f'MATCH (n:ProjectionNode {{group_id: "{GRAPH}", engagement_id: "{ENGAGEMENT}", uuid: "law-1:law-v1:2.01"}}) DETACH DELETE n' in detail
    assert "stored twice" not in detail
    assert store.snapshot() == before


@pytest.mark.parametrize("name", sorted(pm.LAW_OWNED_PROPERTIES))
def test_the_laws_own_property_names_are_written_by_the_service_and_refused_from_a_caller(name, store, client, law_on):
    # A section sent with a refusal mark far in the future would be passed over by the embedding pass for good; one
    # sent with a value in place of its vector would fail the meaning read for the whole graph. Neither is sent.
    assert pm.LAW_OWNED_PROPERTIES == {
        f"{pm.LAW_TEXT_PROPERTY}{suffix}" for suffix in ("", "_embedding", "_emb_model", "_emb_failed", "_emb_failed_at")
    }
    assert pm.LAW_OWNED_PROPERTIES <= pm.SERVICE_OWNED_PROPERTIES
    body = law_envelope()
    body["node_ops"][0]["properties"][name] = 9_999_999_999 if name.endswith("_at") else "x"
    response = apply(client, body)
    assert response.status_code == 422
    assert f"property '{name}' is written by the projection service" in json.dumps(response.json())
    assert store.mutating_queries() == []


def test_a_process_records_text_stays_text_and_only_a_sections_words_move(store, client, lane_on, law_on):
    assert apply(client, structure_envelope()).status_code == 200
    assert apply(client, law_envelope(operation_id="op-2", ordinal=1)).status_code == 200
    props = {node["props"]["uuid"]: node["props"] for node in store.nodes.values()}
    assert props["pv-3:s1"]["text"] == "Step 1 is a task." and pm.LAW_TEXT_PROPERTY not in props["pv-3:s1"]
    assert props["pv-3"]["text"] == "Version 3 was approved." and pm.LAW_TEXT_PROPERTY not in props["pv-3"]
    assert "text" not in props["law-1:law-v1:2.01"] and pm.LAW_TEXT_PROPERTY in props["law-1:law-v1:2.01"]


def test_a_section_found_under_both_base_labels_is_refused_with_how_to_recover(store, client, law_on):
    assert apply(client, law_envelope()).status_code == 200
    # As a graph written before sections had a label of their own and then written again by this service would hold.
    key = next(k for k in store.nodes if k[0] == pm.LAW_SECTION_BASE_LABEL and ("uuid", "law-1:law-v1:2.01") in k)
    store.nodes[(pm.PROJECTION_NODE_LABEL,) + key[1:]] = copy.deepcopy(store.nodes[key])
    before = store.snapshot()
    response = apply(client, law_envelope(operation_id="op-2", ordinal=1))
    assert response.status_code == 422, response.json()
    detail = response.json()["detail"]
    assert "law-1:law-v1:2.01" in detail and "Nothing was written" in detail
    # It says what the operator does: delete those nodes, then send the envelope again.
    # The reason fits the case (stored twice, not "written before"), and the delete removes exactly the node in the
    # way, written out with its own label, tenant, engagement and id, so it runs as it stands.
    assert "'law-1:law-v1:2.01' is stored twice, under LawRecordNode (the label its kind carries) and under ProjectionNode" in detail
    assert (
        f'to recover, run MATCH (n:ProjectionNode {{group_id: "{GRAPH}", engagement_id: "{ENGAGEMENT}", '
        'uuid: "law-1:law-v1:2.01"}) DETACH DELETE n; then send this envelope again.'
    ) in detail
    assert "$uuid" not in detail and "MATCH (n:LawRecordNode" not in detail
    assert store.snapshot() == before


@pytest.mark.parametrize("first", ["section", "process step"])
def test_two_applies_at_once_cannot_leave_two_nodes_under_one_id(first, store, client, lane_on, law_on):
    # Both pass their preflight before either writes (the fake hides the other's node from the lookup, as a
    # lookup made before the other's write would). The write itself stays off a uuid the other base label holds,
    # in the same statement, so the second writes nothing for that row, its read-back finds it missing, and it
    # fails with its receipt left PENDING.
    section = law_envelope(node_ops=[structure_op("shared", "LAW_SECTION", version="law-v1", workflow="law-1", properties={"text": "A section."})], edge_ops=[], operation_id="op-law", ordinal=1)
    step = structure_envelope(node_ops=[structure_op("shared", "PROCESS_STEP", properties={"text": "A step."})], edge_ops=[], operation_id="op-step", ordinal=2)
    winner, loser = (section, step) if first == "section" else (step, section)
    assert apply(client, winner).status_code == 200
    store.drop_after_resolve = "shared"
    response = apply(client, loser)
    detail = response.json()["detail"]
    assert response.status_code == 500 and "verified 0/1 nodes" in detail
    # It says what happened and that a resend cannot cure it, not that the identical envelope may be resent.
    assert "not written because a node of another kind already holds each of them" in detail and "['shared']" in detail
    assert "Resending this envelope as it is will be refused until those ids are free" in detail
    assert "the identical envelope may be resent" not in detail
    resend = apply(client, loser)
    assert resend.status_code == 422 and "shared" in json.dumps(resend.json())
    holders = [k for k in store.nodes if ("uuid", "shared") in k]
    assert len(holders) == 1
    assert {r["operation_id"]: r["status"] for r in store.receipts.values()} == {winner["operation_id"]: "APPLIED", loser["operation_id"]: "PENDING"}


def test_a_row_short_for_any_other_cause_still_says_the_identical_envelope_may_be_resent(store, client, law_on):
    store.skip_node_row = "law-1:law-v1:2.01"
    response = apply(client, law_envelope())
    detail = response.json()["detail"]
    assert response.status_code == 500 and "the identical envelope may be resent" in detail
    assert "another kind already holds" not in detail


# The guard that keeps a write off a uuid the other base label holds (_node_apply_query) is checked here against
# this fake, which obeys its text. This repository's CI runs no graph engine, so the statement itself is checked on
# a real FalkorDB by the law acceptance run that is kept outside this repository (invented data, a local engine,
# two applies at once declaring one id as a section and as a process record, 100 times), before a change to it
# is merged.
