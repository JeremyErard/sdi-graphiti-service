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
        endpoints = _LABEL_KEY.findall(q)
        assert endpoints, q
        scope = {
            k: _resolve(v, params, None)
            for k, v in _parse_map(endpoints[0][2]).items()
            if k != "uuid"
        }
        rows = []
        for wanted in params["edge_ids"]:
            for (_, edge_id, from_id, to_id), edge in self.store.edges.items():
                if edge_id != wanted:
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

        match = _LABEL_KEY.search(q)
        assert match, q
        _, merge_label, body = match.groups()
        keymap = _parse_map(body)
        label_set = _SET_LABELS.search(q)
        extra_labels = [lbl for lbl in (label_set.group(2).split(":") if label_set else []) if lbl]
        setter = _SET_MAP.search(q)
        assert setter, q

        applied = 0
        for row in params["rows"]:
            if row.get("node_id") == self.store.skip_node_row:
                applied += 1
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
            for (_, edge_id, _f, _t), edge in self.store.edges.items():
                if edge_id != row["edge_id"]:
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


@pytest.mark.parametrize("kind", ["WORKFLOW", "PROCESS_VERSION", "SOP", "RACI"])
def test_p3_source_kinds_are_refused_today(kind, store, client):
    body = envelope(node_ops=[node_op(source={**finding_source(), "kind": kind}, lifecycle=None)])
    response = apply(client, body)
    assert response.status_code == 422
    assert "P3" in json.dumps(response.json())
    assert store.mutating_queries() == []


def test_a_p3_shaped_payload_cannot_ride_in_under_an_admitted_source_kind(store, client):
    """The phase gate cannot be a self-declared metadata field: what actually
    lands in the graph is what has to be constrained."""
    labelled = envelope(node_ops=[node_op()])
    labelled["node_ops"][0]["labels"] = ["SOP", "Workflow"]
    assert apply(client, labelled).status_code == 422
    assert store.nodes == {}

    raci = envelope(
        node_ops=[node_op(), driver_op()],
        edge_ops=[edge_op(relation="ACCOUNTABLE_FOR")],
    )
    response = apply(client, raci)
    assert response.status_code == 422
    assert "P3" in json.dumps(response.json())
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
    assert "P2-prime allowlist" in str(excinfo.value)


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
    assert pm.P2_PRIME_SOURCE_KINDS | pm.P3_SOURCE_KINDS == kinds
    assert not (pm.P2_PRIME_SOURCE_KINDS & pm.P3_SOURCE_KINDS)
    assert set(pm.SOURCE_KIND_NODE_LABEL) == kinds
    assert {pm.SOURCE_KIND_NODE_LABEL[k] for k in pm.P3_SOURCE_KINDS} <= pm.P3_NODE_LABELS


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
            envelope(node_ops=[node_op(source={**finding_source(), "kind": "SOP"}, lifecycle=None)])
        )
    assert "P2-prime allowlist" in str(excinfo.value)


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
            assert f":{pm.PROJECTION_NODE_LABEL} {{" in statement
    for statement in [q for q, _ in store.queries if q.startswith("UNWIND $node_ids")]:
        assert f"MATCH (n:{pm.PROJECTION_NODE_LABEL} {{uuid: wanted" in statement


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


def test_adding_the_route_did_not_change_any_existing_route(client):
    paths = {route.path for route in app.routes}
    assert PATH in paths
    assert RECEIPTS_PATH in paths
    assert "/ingest/structured/v2" not in paths
    for existing in ["/ingest/episode", "/search/context", "/graph/nodes-and-edges", "/health"]:
        assert existing in paths


def test_the_route_is_mounted_under_the_ingest_scope_with_one_shared_dependency():
    from app import main

    mounted = [route for route in app.routes if getattr(route, "path", "") == PATH]
    assert mounted
    calls = [d.call for d in mounted[0].dependant.dependencies]
    # The mount-level guard and the handler's principal are the same object, so
    # verify_request runs once and the request nonce is consumed once.
    assert calls
    assert set(calls) == {projection.INGEST_PRINCIPAL}
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
