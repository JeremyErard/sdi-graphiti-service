"""Governed bulk exact-ID projection: POST /ingest/projection/v2.

This is the P2-prime import lane. It is deliberately not /ingest/structured/v2:
that route is source-anchored compatibility ingestion which merges by fuzzy name,
and it must never be mistaken for authoritative exact-ID projection. It is also
not /ingest/episode: nothing here runs an LLM or extracts anything. The caller
supplies deterministic node and edge identities and this service applies them
verbatim into the tenant graph.

What this endpoint guarantees, all of it enforced below and proven by
tests/test_projection_v2.py:

* the lane refuses to operate at all unless the signed perimeter is closed, and
  the tenant graph is derived from the authenticated principal as well as the
  body; a namespace that disagrees with the derived graph is refused, not coerced
* the applied receipt is durable, lives in the same tenant graph, and is keyed on
  (group_id, engagement_id, operation_id)
* one cursor position in one tenant is claimed by at most one operation ID
* an exact replay of an applied operation returns the prior receipt and applies
  nothing
* the same operation ID with a different envelope hash is a conflict and applies
  nothing
* an operation interrupted mid-flight is resumable, and reprojecting the same
  immutable source version converges instead of duplicating
* memory headroom is reported before and after the apply, and per-operation
  detail is returned so a spot probe can verify graph-matches-source

The projection owns its own label space. Every projected node carries the
ProjectionNode label and the exact-ID MERGE keys on it, so a node's identity
never depends on its type label, and nothing here can bind, rewrite, or graft
onto a node that graphiti's extraction path owns.

The tenant-local relational database stays authoritative. This service reads
nothing from it and writes nothing back to it. Every hash and identity is
carried by the caller; none is derived from content this service invented.
"""

import hashlib
import logging
import time
import uuid as uuidlib
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from app.auth import GraphPrincipal, require_scope
from app.config import settings
from app.graph_names import graph_name_for_client
from app.models.projection import (
    PROJECTION_NODE_LABEL,
    PROJECTION_RECEIPT_LABEL,
    FindingLifecycle,
    ProjectionEdgeOp,
    ProjectionEnvelopeV2,
    ProjectionNodeOp,
    ProjectionOrigin,
    ProjectionRelation,
    RELATION_ENDPOINT_KINDS,
    canonical_json,
    operation_hash,
)

logger = logging.getLogger("graphiti_service")

router = APIRouter()

# One dependency object shared by the route and by the mount in app/main.py.
# require_scope() builds a new closure per call, and FastAPI caches a resolved
# dependency by callable identity, so two separate closures would run
# verify_request twice and burn the request nonce twice. Sharing one object keeps
# the mount-level guard and gives the handler the principal from a single check.
INGEST_PRINCIPAL = require_scope("ingest")

RECEIPT_STATUS_PENDING = "PENDING"
RECEIPT_STATUS_APPLIED = "APPLIED"

# How many operation identities are echoed in the response. The full batch is
# covered by operations_digest, so the response stays bounded at the operation
# budget instead of growing to hundreds of kilobytes.
MAX_ECHOED_OPERATIONS = 50
SPOT_PROBE_SAMPLE = 5

# Rows per UNWIND. One statement carrying the whole operation budget would be a
# single oversized parameter payload with no partial progress on rejection.
APPLY_CHUNK_ROWS = 500

RECEIPT_KEY = (
    f"(r:{PROJECTION_RECEIPT_LABEL} {{group_id: $group_id, "
    "engagement_id: $engagement_id, operation_id: $operation_id})"
)

READ_RECEIPT_QUERY = f"MATCH {RECEIPT_KEY} RETURN properties(r) AS props LIMIT 1"

# Create-if-absent. The MERGE key is the receipt identity, and the property write
# is guarded on the receipt not already carrying an envelope hash, so a
# concurrent writer that got there first is never overwritten, and a receipt left
# half written by a crash is adopted rather than poisoning its operation ID
# forever. The authoritative answer is the read that follows.
CREATE_RECEIPT_QUERY = (
    f"MERGE {RECEIPT_KEY} WITH r WHERE r.envelope_hash IS NULL SET r = $props"
)

FINALIZE_RECEIPT_QUERY = (
    f"MATCH {RECEIPT_KEY} "
    "SET r.status = $status, r.nodes_applied = $nodes_applied, "
    "r.edges_applied = $edges_applied, r.finalized_at = $finalized_at "
    "RETURN properties(r) AS props"
)

PROGRESS_RECEIPT_QUERY = (
    f"MATCH {RECEIPT_KEY} "
    "SET r.nodes_applied = $nodes_applied, r.edges_applied = $edges_applied"
)

# One cursor position in one tenant belongs to at most one operation. Without
# this an importer that re-cuts a batch under a fresh operation ID would apply
# that ordinal twice with no signal.
CURSOR_CLAIM_QUERY = (
    f"MATCH (r:{PROJECTION_RECEIPT_LABEL} {{group_id: $group_id, "
    "engagement_id: $engagement_id, cursor_key: $cursor_key}) "
    "RETURN r.operation_id AS operation_id LIMIT 5"
)

# The manifest is frozen at one cutoff. Two envelopes of one manifest declaring
# different cutoffs would mean the import was not taken against a frozen cutoff.
MANIFEST_CUTOFF_QUERY = (
    f"MATCH (r:{PROJECTION_RECEIPT_LABEL} {{group_id: $group_id, "
    "import_manifest_hash: $import_manifest_hash}) "
    "RETURN DISTINCT r.cutoff_id AS cutoff_id LIMIT 5"
)

# Scoped to the projection label space. An unscoped MATCH on uuid would bind
# graphiti's Entity, Episodic, Community, and Saga nodes, which would let an edge
# graft projected content onto extracted content and would scan the whole graph.
RESOLVE_NODES_QUERY = (
    "UNWIND $node_ids AS wanted "
    f"MATCH (n:{PROJECTION_NODE_LABEL} {{uuid: wanted, group_id: $group_id, "
    "engagement_id: $engagement_id}) "
    "RETURN wanted AS id, n.projection_source_kind AS kind"
)

RESOLVE_EDGES_QUERY = (
    "UNWIND $edge_ids AS wanted "
    f"MATCH (a:{PROJECTION_NODE_LABEL} {{group_id: $group_id, "
    "engagement_id: $engagement_id})-[e {uuid: wanted}]->"
    f"(b:{PROJECTION_NODE_LABEL} {{group_id: $group_id, engagement_id: $engagement_id}}) "
    "RETURN wanted AS id, type(e) AS relation, a.uuid AS from_id, b.uuid AS to_id"
)

VERIFY_NODES_QUERY = (
    "UNWIND $rows AS row "
    f"MATCH (n:{PROJECTION_NODE_LABEL} {{uuid: row.node_id, group_id: $group_id, "
    "engagement_id: $engagement_id}) "
    "WHERE n.projection_op_hash = row.op_hash "
    "RETURN count(DISTINCT n.uuid) AS verified"
)

VERIFY_EDGES_QUERY = (
    "UNWIND $rows AS row "
    f"MATCH (:{PROJECTION_NODE_LABEL} {{group_id: $group_id, "
    "engagement_id: $engagement_id})-[e {uuid: row.edge_id}]->"
    f"(:{PROJECTION_NODE_LABEL} {{group_id: $group_id, engagement_id: $engagement_id}}) "
    "WHERE e.projection_op_hash = row.op_hash "
    "RETURN count(DISTINCT e.uuid) AS verified"
)

# Best-effort. Without these every exact-ID MERGE is a label scan, which turns a
# manifest-scale import quadratic while the memory probe still reports healthy.
INDEX_STATEMENTS = (
    f"CREATE INDEX FOR (n:{PROJECTION_NODE_LABEL}) ON (n.uuid)",
    f"CREATE INDEX FOR (n:{PROJECTION_NODE_LABEL}) ON (n.group_id)",
    f"CREATE INDEX FOR (n:{PROJECTION_NODE_LABEL}) ON (n.engagement_id)",
    f"CREATE INDEX FOR (r:{PROJECTION_RECEIPT_LABEL}) ON (r.group_id)",
    f"CREATE INDEX FOR (r:{PROJECTION_RECEIPT_LABEL}) ON (r.operation_id)",
    f"CREATE INDEX FOR (r:{PROJECTION_RECEIPT_LABEL}) ON (r.cursor_key)",
    f"CREATE INDEX FOR (r:{PROJECTION_RECEIPT_LABEL}) ON (r.import_manifest_hash)",
)

# Graphs whose projection indices this process has already attempted.
_INDEXED_GRAPHS: set[str] = set()

# Spot-probe queries a human can run against the tenant graph to compare a
# projected row with the authoritative source record.
NODE_PROBE_CYPHER = (
    f"MATCH (n:{PROJECTION_NODE_LABEL} {{group_id: $group_id, uuid: $uuid}}) "
    "RETURN properties(n)"
)
EDGE_PROBE_CYPHER = (
    f"MATCH (:{PROJECTION_NODE_LABEL} {{group_id: $group_id}})-[e {{uuid: $uuid}}]->() "
    "RETURN properties(e)"
)

CONFLICT_CODE_OPERATION = "projection_operation_conflict"
CONFLICT_CODE_CURSOR = "projection_cursor_conflict"


class MemoryHeadroom(BaseModel):
    """FalkorDB memory surface, tolerated as absent."""

    available: bool
    probed_at: str | None = None
    used_memory_bytes: int | None = None
    used_memory_rss_bytes: int | None = None
    maxmemory_bytes: int | None = None
    headroom_bytes: int | None = None


class ProjectionOperationProbe(BaseModel):
    kind: str
    id: str
    op_hash: str
    source_id: str
    source_version_id: str
    source_content_hash: str


class ProjectionReceiptView(BaseModel):
    receipt_id: str
    operation_id: str
    envelope_hash: str
    status: str
    group_id: str
    engagement_id: str
    schema_version: str
    origin: str
    cursor_space: str
    cursor_key: str
    import_manifest_hash: str | None = None
    batch_ordinal: int | None = None
    event_id: str | None = None
    cutoff_id: str
    source_identity_digest: str
    node_ops_requested: int
    edge_ops_requested: int
    nodes_applied: int
    edges_applied: int
    created_at: str
    finalized_at: str | None = None


class ProjectionApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_name: str
    receipt: ProjectionReceiptView
    applied: bool
    replay: bool
    resumed: bool
    nodes_applied: int
    edges_applied: int
    memory_before: MemoryHeadroom
    memory_after: MemoryHeadroom
    operations_total: int
    operations_digest: str
    operations: list[ProjectionOperationProbe]
    spot_probe: dict[str, Any]
    elapsed_ms: int


class ProjectionReceiptQuery(BaseModel):
    """Read-only ledger query.

    It is a POST because the signed perimeter binds the tenant claim to a
    top-level client_slug in the JSON body; a GET has no body and would have to
    be platform-scoped, which is the wrong scope for reading one tenant's ledger.
    Nothing here writes.
    """

    model_config = ConfigDict(extra="forbid")

    client_slug: str = Field(min_length=1, max_length=128)
    engagement_id: str = Field(min_length=1, max_length=256)
    operation_id: str | None = Field(default=None, min_length=1, max_length=256)
    import_manifest_hash: str | None = Field(default=None, min_length=1, max_length=64)
    limit: int = Field(default=50, ge=1, le=500)


class ProjectionReceiptListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_name: str
    receipts: list[ProjectionReceiptView]


def _fail(status_code: int, detail: Any) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _guard_lane(principal: GraphPrincipal, client_slug: str) -> None:
    """Refuse before anything else unless the perimeter is closed and bound.

    A governed lane cannot inherit a staged-rollout leniency. In off or optional
    mode verify_request never reads the body's tenant claim, so the tenant would
    be whatever the body said; this lane declines to run at all in that posture,
    and independently requires the authenticated principal to name the same
    tenant the envelope does.
    """
    if settings.graphiti_auth_mode != "required":
        raise _fail(
            503,
            "governed projection requires GRAPHITI_AUTH_MODE=required; in off or "
            "optional mode the tenant claim in the body is not bound to an "
            "authenticated principal and this lane refuses to write",
        )
    if principal.client_slug != client_slug:
        raise _fail(
            403,
            "the authenticated principal does not match the envelope's client_slug",
        )


async def _read_memory_info() -> dict[str, Any]:
    """Read the FalkorDB memory INFO surface. Raises if it is unreachable."""
    import redis.asyncio as redis

    client = redis.Redis(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    try:
        return await client.info("memory")
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


async def _memory_headroom() -> MemoryHeadroom:
    """P2 gate element. Never fails the request: absence is reported, not raised."""
    probed_at = datetime.now(timezone.utc).isoformat()
    try:
        info = await _read_memory_info()
        used = _as_int(info.get("used_memory"))
        rss = _as_int(info.get("used_memory_rss"))
        maxmemory = _as_int(info.get("maxmemory"))
        headroom = None
        if maxmemory is not None and maxmemory > 0 and used is not None:
            headroom = maxmemory - used
        return MemoryHeadroom(
            available=True,
            probed_at=probed_at,
            used_memory_bytes=used,
            used_memory_rss_bytes=rss,
            maxmemory_bytes=maxmemory,
            headroom_bytes=headroom,
        )
    except Exception as exc:
        logger.info("[graphiti] projection memory probe unavailable: %s", exc)
        return MemoryHeadroom(available=False, probed_at=probed_at)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _receipt_id(group_id: str, engagement_id: str, operation_id: str) -> str:
    """Deterministic receipt identity, so a replay names the same receipt."""
    material = f"{group_id}\x1f{engagement_id}\x1f{operation_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


async def _query(graph: Any, statement: str, params: dict[str, Any] | None = None) -> Any:
    """Run one Cypher statement off the event loop.

    The FalkorDB driver is synchronous. A manifest-scale envelope executed inline
    in an async handler would hold the loop for its whole duration and stall
    /health and every other tenant's search alongside it.
    """
    return await run_in_threadpool(graph.query, statement, params)


async def _ensure_indices(graph: Any, graph_name: str) -> None:
    """Best-effort, once per graph per process. Never fails the request."""
    if graph_name in _INDEXED_GRAPHS:
        return
    for statement in INDEX_STATEMENTS:
        try:
            await _query(graph, statement)
        except Exception as exc:
            logger.info(
                "[graphiti] projection index not created on %s (%s): %s",
                graph_name,
                statement,
                exc,
            )
    _INDEXED_GRAPHS.add(graph_name)


def _lifecycle_properties(lifecycle: FindingLifecycle | None) -> dict[str, Any]:
    """Lifecycle facts as stated by the caller, never as inferred here.

    An unreviewed import carries no validator property at all and carries
    lifecycle_human_validated=false, so it cannot be rendered anywhere as
    named-human validation. A legacy validation flag that could not be proven
    survives as audit metadata and never sets lifecycle_human_validated.
    lifecycle_retained is the only flag a retrieval surface should gate on.
    """
    if lifecycle is None:
        return {}
    props: dict[str, Any] = {
        "lifecycle_state": lifecycle.state.value,
        "lifecycle_retained": lifecycle.retained,
        "lifecycle_human_validated": lifecycle.human_validated,
        "lifecycle_legacy_validated_flag": lifecycle.legacy_validated_flag,
    }
    if lifecycle.merged_into_id is not None:
        props["lifecycle_merged_into_id"] = lifecycle.merged_into_id
    if lifecycle.superseded_by_id is not None:
        props["lifecycle_superseded_by_id"] = lifecycle.superseded_by_id
    if lifecycle.human_validated:
        props["lifecycle_validated_by"] = lifecycle.validated_by
        props["lifecycle_validator_subject_id"] = lifecycle.validator_subject_id
        props["lifecycle_validated_at"] = lifecycle.validated_at.isoformat()
        props["lifecycle_validated_version_id"] = lifecycle.validated_version_id
        props["lifecycle_validated_content_hash"] = lifecycle.validated_content_hash
    return props


def _envelope_properties(envelope: ProjectionEnvelopeV2, group_id: str) -> dict[str, Any]:
    """Envelope-scoped identity carried onto every projected row.

    Deliberately free of wall-clock values and of everything that identifies one
    operation rather than one source version: no operation ID, no envelope hash,
    no cursor. Those live on the receipt, which is the ledger. Keeping them off
    the row is what makes reprojecting the same immutable source version at the
    same cutoff produce a byte-identical row rather than churning metadata, so
    convergence is observable without excusing any field from comparison.
    """
    return {
        "group_id": group_id,
        "engagement_id": envelope.engagement_id,
        "projection_schema_version": envelope.schema_version,
        "projection_origin": envelope.origin.value,
        "projection_cutoff_id": envelope.cutoff.cutoff_id,
    }


def _source_properties(source: Any) -> dict[str, Any]:
    """This row's own immutable source identity, not the batch's."""
    return {
        "projection_source_kind": source.kind.value,
        "projection_source_id": source.id,
        "projection_source_version_id": source.immutable_version_id,
        "projection_source_content_hash": source.content_hash,
    }


def _node_row(op: ProjectionNodeOp, envelope_props: dict[str, Any]) -> dict[str, Any]:
    props: dict[str, Any] = dict(op.properties)
    props.update(envelope_props)
    props.update(_source_properties(op.source))
    props.update(_lifecycle_properties(op.lifecycle))
    props["uuid"] = op.node_id
    props["projection_op_hash"] = operation_hash(op)
    return {"node_id": op.node_id, "op_hash": props["projection_op_hash"], "props": props}


def _edge_row(
    op: ProjectionEdgeOp,
    envelope_props: dict[str, Any],
    source_by_node: dict[str, Any],
) -> dict[str, Any]:
    props: dict[str, Any] = dict(op.properties)
    props.update(envelope_props)
    source = source_by_node.get(op.from_node_id)
    if source is not None:
        props.update(_source_properties(source))
    props["uuid"] = op.edge_id
    props["projection_op_hash"] = operation_hash(op)
    return {
        "edge_id": op.edge_id,
        "op_hash": props["projection_op_hash"],
        "from_node_id": op.from_node_id,
        "to_node_id": op.to_node_id,
        "props": props,
    }


def _node_apply_query(type_label: str) -> str:
    """MERGE on the exact identity, then restate the full property map.

    The MERGE key is the constant projection label plus the tenant, engagement,
    and caller-supplied uuid. Identity therefore does not depend on the node's
    type label, so a node whose type changes between projections updates in place
    instead of splitting into two rows that share a uuid. The type label is
    inlined because Cypher cannot parameterize a label; it is never caller text,
    it is the fixed label this service maps the declared source kind to.
    """
    return (
        "UNWIND $rows AS row "
        f"MERGE (n:{PROJECTION_NODE_LABEL} {{uuid: row.node_id, "
        "group_id: $group_id, engagement_id: $engagement_id}) "
        f"SET n:{type_label} "
        "SET n = row.props "
        "RETURN count(n) AS applied"
    )


def _edge_apply_query(relation: str) -> str:
    return (
        "UNWIND $rows AS row "
        f"MATCH (a:{PROJECTION_NODE_LABEL} {{uuid: row.from_node_id, "
        "group_id: $group_id, engagement_id: $engagement_id}) "
        f"MATCH (b:{PROJECTION_NODE_LABEL} {{uuid: row.to_node_id, "
        "group_id: $group_id, engagement_id: $engagement_id}) "
        f"MERGE (a)-[e:{relation} {{uuid: row.edge_id}}]->(b) "
        "SET e = row.props "
        "RETURN count(e) AS applied"
    )


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _first_scalar(result: Any, default: int = 0) -> int:
    rows = getattr(result, "result_set", None) or []
    if not rows or not rows[0]:
        return default
    value = _as_int(rows[0][0])
    return default if value is None else value


def _first_props(result: Any) -> dict[str, Any] | None:
    rows = getattr(result, "result_set", None) or []
    if not rows or not rows[0]:
        return None
    props = rows[0][0]
    return dict(props) if props else None


def _rows_of(result: Any) -> list[list[Any]]:
    return list(getattr(result, "result_set", None) or [])


def _receipt_view(props: dict[str, Any]) -> ProjectionReceiptView:
    return ProjectionReceiptView(
        receipt_id=str(props.get("receipt_id", "")),
        operation_id=str(props.get("operation_id", "")),
        envelope_hash=str(props.get("envelope_hash", "")),
        status=str(props.get("status", "")),
        group_id=str(props.get("group_id", "")),
        engagement_id=str(props.get("engagement_id", "")),
        schema_version=str(props.get("schema_version", "")),
        origin=str(props.get("origin", "")),
        cursor_space=str(props.get("cursor_space", "")),
        cursor_key=str(props.get("cursor_key", "")),
        import_manifest_hash=(
            str(props["import_manifest_hash"])
            if props.get("import_manifest_hash") is not None
            else None
        ),
        batch_ordinal=_as_int(props.get("batch_ordinal")),
        event_id=(str(props["event_id"]) if props.get("event_id") is not None else None),
        cutoff_id=str(props.get("cutoff_id", "")),
        source_identity_digest=str(props.get("source_identity_digest", "")),
        node_ops_requested=_as_int(props.get("node_ops_requested")) or 0,
        edge_ops_requested=_as_int(props.get("edge_ops_requested")) or 0,
        nodes_applied=_as_int(props.get("nodes_applied")) or 0,
        edges_applied=_as_int(props.get("edges_applied")) or 0,
        created_at=str(props.get("created_at", "")),
        finalized_at=(
            str(props["finalized_at"]) if props.get("finalized_at") is not None else None
        ),
    )


def _all_operation_hashes(envelope: ProjectionEnvelopeV2) -> list[str]:
    return [operation_hash(op) for op in envelope.node_ops] + [
        operation_hash(op) for op in envelope.edge_ops
    ]


def _operations_digest(hashes: Sequence[str]) -> str:
    """One digest over every operation in order, so the bounded echo still lets a
    caller verify that the batch this service applied is the batch it sent."""
    return hashlib.sha256(canonical_json(list(hashes)).encode("utf-8")).hexdigest()


def _source_identity_digest(envelope: ProjectionEnvelopeV2) -> str:
    """Digest over every source identity in the batch, recorded on the receipt.

    The receipt cannot carry one source ID without misnaming the rest of the
    batch, so it carries the digest of all of them plus the frozen cutoff.
    """
    identities = [
        [op.source.kind.value, op.source.id, op.source.immutable_version_id, op.source.content_hash]
        for op in envelope.node_ops
    ]
    return hashlib.sha256(canonical_json(identities).encode("utf-8")).hexdigest()


def _spread(items: Sequence[Any], count: int) -> list[Any]:
    """An evenly spaced sample including the first and last item.

    Taking the first N would make a systematic error in the tail of a batch
    invisible to the spot probe, which is the P2 gate element this feeds.
    """
    if len(items) <= count:
        return list(items)
    if count <= 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    picked: list[Any] = []
    for index in range(count):
        candidate = items[round(index * step)]
        if candidate not in picked:
            picked.append(candidate)
    return picked


def _operation_probes(envelope: ProjectionEnvelopeV2) -> list[ProjectionOperationProbe]:
    node_probes = [
        ProjectionOperationProbe(
            kind="node",
            id=op.node_id,
            op_hash=operation_hash(op),
            source_id=op.source.id,
            source_version_id=op.source.immutable_version_id,
            source_content_hash=op.source.content_hash,
        )
        for op in _spread(envelope.node_ops, MAX_ECHOED_OPERATIONS)
    ]
    remaining = max(0, MAX_ECHOED_OPERATIONS - len(node_probes))
    source_by_node = {op.node_id: op.source for op in envelope.node_ops}
    edge_probes = []
    for op in _spread(envelope.edge_ops, remaining):
        source = source_by_node.get(op.from_node_id)
        edge_probes.append(
            ProjectionOperationProbe(
                kind="edge",
                id=op.edge_id,
                op_hash=operation_hash(op),
                source_id=source.id if source else "",
                source_version_id=source.immutable_version_id if source else "",
                source_content_hash=source.content_hash if source else "",
            )
        )
    return node_probes + edge_probes


def _spot_probe(envelope: ProjectionEnvelopeV2, group_id: str) -> dict[str, Any]:
    sample_nodes = _spread(envelope.node_ops, SPOT_PROBE_SAMPLE)
    return {
        "group_id": group_id,
        "node_cypher": NODE_PROBE_CYPHER,
        "edge_cypher": EDGE_PROBE_CYPHER,
        "cutoff_id": envelope.cutoff.cutoff_id,
        "samples": [
            {
                "node_id": op.node_id,
                "source_kind": op.source.kind.value,
                "source_id": op.source.id,
                "source_immutable_version_id": op.source.immutable_version_id,
                "source_content_hash": op.source.content_hash,
            }
            for op in sample_nodes
        ],
        "sample_edge_ids": [op.edge_id for op in _spread(envelope.edge_ops, SPOT_PROBE_SAMPLE)],
    }


# Module-local FalkorDB handle, DELIBERATELY not shared with the rest of the
# service.
#
# Constructing FalkorDB() opens a connection pool and _open_graph ran per
# request, leaking one every time; on 2026-08-27 the accumulated leaks across
# this service exhausted FalkorDB's client limit and it refused every query
# with "Too many connections".
#
# The obvious fix — importing the shared accessor the other modules now use —
# is not available here and should not be made available. This module is pinned
# by test_the_projection_module_opens_no_path_back_to_a_system_of_record to
# import nothing that could become a reverse-write path, and that module is
# named on its forbidden list. So the pooling fix is duplicated rather than
# shared: a few lines of duplication is the cheaper price than a hole in that
# boundary.
_falkor_db: Any = None


def _get_falkor_db() -> Any:
    global _falkor_db
    if _falkor_db is None:
        from falkordb import FalkorDB

        _falkor_db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
        )
    return _falkor_db


def _reset_falkor_db() -> None:
    """Drop the handle so the next call reconnects (tests inject their own)."""
    global _falkor_db
    _falkor_db = None


def _open_graph(graph_name: str) -> Any:
    return _get_falkor_db().select_graph(graph_name)


def _derive_graph_name(client_slug: str) -> str:
    try:
        return graph_name_for_client(client_slug)
    except ValueError as exc:
        raise _fail(422, f"Invalid client_slug: {exc}")


async def _preflight(
    graph: Any,
    envelope: ProjectionEnvelopeV2,
    graph_name: str,
) -> None:
    """Everything that can refuse the envelope, before anything is written."""
    scope = {"group_id": graph_name, "engagement_id": envelope.engagement_id}

    declared_kinds = envelope.node_kinds()
    endpoint_ids = {op.from_node_id for op in envelope.edge_ops} | {
        op.to_node_id for op in envelope.edge_ops
    }
    wanted_ids = sorted(set(declared_kinds) | endpoint_ids)

    stored_kinds: dict[str, str] = {}
    if wanted_ids:
        for chunk in _chunks([{"id": node_id} for node_id in wanted_ids], APPLY_CHUNK_ROWS):
            rows = _rows_of(
                await _query(
                    graph,
                    RESOLVE_NODES_QUERY,
                    params={**scope, "node_ids": [row["id"] for row in chunk]},
                )
            )
            for row in rows:
                if row:
                    stored_kinds[str(row[0])] = str(row[1]) if len(row) > 1 else ""

    missing = [
        node_id
        for node_id in sorted(endpoint_ids - set(declared_kinds))
        if node_id not in stored_kinds
    ]
    if missing:
        raise _fail(
            422,
            "edge endpoints are not present in this envelope and do not exist in "
            f"{graph_name}: {missing[:10]}",
        )

    # A node whose declared kind differs from the kind already stored would keep
    # the old type label (SET n:Label only adds) while its whole property map is
    # replaced, leaving a row labelled Finding with no lifecycle at all.
    retyped = [
        node_id
        for node_id, kind in declared_kinds.items()
        if node_id in stored_kinds
        and stored_kinds[node_id]
        and stored_kinds[node_id] != kind.value
    ]
    if retyped:
        raise _fail(
            422,
            "these node IDs are already projected under a different source kind, "
            "and a projected node's type label cannot be removed: "
            f"{sorted(retyped)[:10]}",
        )

    # Endpoint kinds must match the relation's declared shape, so a Finding
    # cannot be recorded as derived from another Finding.
    all_kinds: dict[str, str] = {node_id: kind.value for node_id, kind in declared_kinds.items()}
    for node_id, kind in stored_kinds.items():
        all_kinds.setdefault(node_id, kind)
    for op in envelope.edge_ops:
        expected = RELATION_ENDPOINT_KINDS[ProjectionRelation(op.relation)]
        from_kind = all_kinds.get(op.from_node_id)
        to_kind = all_kinds.get(op.to_node_id)
        if from_kind and from_kind != expected[0].value:
            raise _fail(
                422,
                f"edge '{op.edge_id}' relation {op.relation} starts at a "
                f"{from_kind} node; it must start at a {expected[0].value} node",
            )
        if to_kind and to_kind != expected[1].value:
            raise _fail(
                422,
                f"edge '{op.edge_id}' relation {op.relation} ends at a {to_kind} "
                f"node; it must end at a {expected[1].value} node",
            )

    # An edge ID already bound to different endpoints or a different relation
    # would leave the old edge in place next to the new one, so the graph would
    # hold two contradictory rows for one edge identity.
    if envelope.edge_ops:
        wanted_by_id = {op.edge_id: op for op in envelope.edge_ops}
        for chunk in _chunks([{"id": edge_id} for edge_id in sorted(wanted_by_id)], APPLY_CHUNK_ROWS):
            rows = _rows_of(
                await _query(
                    graph,
                    RESOLVE_EDGES_QUERY,
                    params={**scope, "edge_ids": [row["id"] for row in chunk]},
                )
            )
            for row in rows:
                if not row or len(row) < 4:
                    continue
                op = wanted_by_id.get(str(row[0]))
                if op is None:
                    continue
                if (
                    str(row[1]) != op.relation
                    or str(row[2]) != op.from_node_id
                    or str(row[3]) != op.to_node_id
                ):
                    raise _fail(
                        422,
                        f"edge '{op.edge_id}' already exists in {graph_name} joining "
                        f"{row[2]}-[{row[1]}]->{row[3]}; re-pointing an edge identity "
                        "would leave the previous edge in place alongside the new one",
                    )

    # One cursor position, at most one operation.
    claimants = {
        str(row[0])
        for row in _rows_of(
            await _query(
                graph,
                CURSOR_CLAIM_QUERY,
                params={**scope, "cursor_key": envelope.cursor_key()},
            )
        )
        if row
    }
    foreign = sorted(claimants - {envelope.operation_id})
    if foreign:
        raise _fail(
            409,
            {
                "code": CONFLICT_CODE_CURSOR,
                "message": (
                    "this cursor position is already claimed by another operation; "
                    "nothing was applied. Resend the original operation rather than "
                    "re-cutting the batch under a new operation ID"
                ),
                "cursor_key": envelope.cursor_key(),
                "claimed_by": foreign,
            },
        )

    if envelope.cursor.import_manifest_hash is not None:
        declared_cutoffs = {
            str(row[0])
            for row in _rows_of(
                await _query(
                    graph,
                    MANIFEST_CUTOFF_QUERY,
                    params={
                        "group_id": graph_name,
                        "import_manifest_hash": envelope.cursor.import_manifest_hash,
                    },
                )
            )
            if row and row[0] is not None
        }
        other = sorted(declared_cutoffs - {envelope.cutoff.cutoff_id})
        if other:
            raise _fail(
                422,
                "this import manifest was already projected against a different "
                f"frozen cutoff {other}; a manifest freezes at one cutoff",
            )


@router.post("/projection/v2", response_model=ProjectionApplyResponse)
async def apply_projection_v2(
    envelope: ProjectionEnvelopeV2,
    principal: GraphPrincipal = Depends(INGEST_PRINCIPAL),
):
    """Apply one governed projection envelope to the tenant graph.

    Ordering matters and is part of the contract: everything that can refuse the
    envelope runs before anything is written, so a refused envelope leaves no
    receipt and no partial graph state.
    """
    start = time.monotonic()
    correlation_id = uuidlib.uuid4().hex

    _guard_lane(principal, envelope.client_slug)
    graph_name = _derive_graph_name(envelope.client_slug)

    if envelope.namespace != graph_name:
        raise _fail(
            422,
            f"namespace '{envelope.namespace}' does not match the graph derived "
            f"from client_slug '{envelope.client_slug}' ('{graph_name}'); the "
            "projection refuses a namespace mismatch rather than coercing it",
        )

    if (
        envelope.origin is ProjectionOrigin.OUTCOME_EVENT
        and not settings.projection_v2_allow_outcome_event
    ):
        raise _fail(
            422,
            "OUTCOME_EVENT projection is not authorized in this phase. The "
            "envelope carries the origin so the Outcome projector emits the same "
            "shape later; set PROJECTION_V2_ALLOW_OUTCOME_EVENT once that lane is "
            "ratified",
        )

    envelope_hash = envelope.canonical_hash()
    scope = {"group_id": graph_name, "engagement_id": envelope.engagement_id}
    receipt_key = {**scope, "operation_id": envelope.operation_id}
    memory_before = await _memory_headroom()

    try:
        graph = _open_graph(graph_name)
        await _ensure_indices(graph, graph_name)

        prior = _first_props(await _query(graph, READ_RECEIPT_QUERY, params=receipt_key))
        prior_hash = str(prior.get("envelope_hash") or "") if prior else ""

        if prior_hash and prior_hash != envelope_hash:
            logger.warning(
                "[graphiti] projection conflict on %s operation_id=%s: stored hash "
                "differs from the submitted envelope; applying nothing",
                graph_name,
                envelope.operation_id,
            )
            raise _fail(
                409,
                {
                    "code": CONFLICT_CODE_OPERATION,
                    "message": (
                        f"operation_id '{envelope.operation_id}' already exists with "
                        "a different envelope hash; nothing was applied. A changed "
                        "envelope needs a new operation_id. This is not a replayed "
                        "signature: an identical envelope may be re-signed and resent"
                    ),
                    "operation_id": envelope.operation_id,
                    "stored_envelope_hash": prior_hash,
                    "submitted_envelope_hash": envelope_hash,
                },
            )

        if prior_hash and str(prior.get("status", "")) == RECEIPT_STATUS_APPLIED:
            memory_after = await _memory_headroom()
            logger.info(
                "[graphiti] projection replay on %s operation_id=%s: returning the "
                "prior receipt, applying nothing",
                graph_name,
                envelope.operation_id,
            )
            hashes = _all_operation_hashes(envelope)
            return ProjectionApplyResponse(
                graph_name=graph_name,
                receipt=_receipt_view(prior),
                applied=False,
                replay=True,
                resumed=False,
                nodes_applied=_as_int(prior.get("nodes_applied")) or 0,
                edges_applied=_as_int(prior.get("edges_applied")) or 0,
                memory_before=memory_before,
                memory_after=memory_after,
                operations_total=len(hashes),
                operations_digest=_operations_digest(hashes),
                operations=_operation_probes(envelope),
                spot_probe=_spot_probe(envelope, graph_name),
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )

        # A receipt that exists with a claimed hash equal to ours was interrupted
        # mid-flight. A receipt that exists without a hash was half written and is
        # adopted below rather than poisoning this operation ID forever.
        resumed = bool(prior_hash)

        await _preflight(graph, envelope, graph_name)

        if not resumed:
            receipt_props = {
                "receipt_id": _receipt_id(
                    graph_name, envelope.engagement_id, envelope.operation_id
                ),
                "group_id": graph_name,
                "engagement_id": envelope.engagement_id,
                "operation_id": envelope.operation_id,
                "envelope_hash": envelope_hash,
                "status": RECEIPT_STATUS_PENDING,
                "schema_version": envelope.schema_version,
                "origin": envelope.origin.value,
                "cursor_space": envelope.cursor_space(),
                "cursor_key": envelope.cursor_key(),
                "import_manifest_hash": envelope.cursor.import_manifest_hash,
                "batch_ordinal": envelope.cursor.batch_ordinal,
                "event_id": envelope.cursor.event_id,
                "cutoff_id": envelope.cutoff.cutoff_id,
                "source_identity_digest": _source_identity_digest(envelope),
                "node_ops_requested": len(envelope.node_ops),
                "edge_ops_requested": len(envelope.edge_ops),
                "nodes_applied": 0,
                "edges_applied": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            receipt_props = {k: v for k, v in receipt_props.items() if v is not None}
            await _query(
                graph,
                CREATE_RECEIPT_QUERY,
                params={**receipt_key, "props": receipt_props},
            )
            # Re-read: if a concurrent writer created this receipt first, the
            # guarded write above did nothing and the stored hash is theirs.
            stored = _first_props(await _query(graph, READ_RECEIPT_QUERY, params=receipt_key))
            if stored is None or str(stored.get("envelope_hash") or "") != envelope_hash:
                raise _fail(
                    409,
                    {
                        "code": CONFLICT_CODE_OPERATION,
                        "message": (
                            f"operation_id '{envelope.operation_id}' was claimed "
                            "concurrently by a different envelope; nothing was applied"
                        ),
                        "operation_id": envelope.operation_id,
                        "stored_envelope_hash": str(stored.get("envelope_hash") or "")
                        if stored
                        else "",
                        "submitted_envelope_hash": envelope_hash,
                    },
                )

        envelope_props = _envelope_properties(envelope, graph_name)
        source_by_node = {op.node_id: op.source for op in envelope.node_ops}

        grouped_nodes: dict[str, list[dict[str, Any]]] = {}
        for op in envelope.node_ops:
            grouped_nodes.setdefault(op.node_label, []).append(_node_row(op, envelope_props))
        node_rows_all = [row for rows in grouped_nodes.values() for row in rows]

        nodes_written = 0
        for type_label, rows in grouped_nodes.items():
            statement = _node_apply_query(type_label)
            for chunk in _chunks(rows, APPLY_CHUNK_ROWS):
                result = await _query(graph, statement, params={**scope, "rows": chunk})
                nodes_written += _first_scalar(result, default=0)
                await _query(
                    graph,
                    PROGRESS_RECEIPT_QUERY,
                    params={**receipt_key, "nodes_applied": nodes_written, "edges_applied": 0},
                )

        grouped_edges: dict[str, list[dict[str, Any]]] = {}
        for op in envelope.edge_ops:
            grouped_edges.setdefault(op.relation, []).append(
                _edge_row(op, envelope_props, source_by_node)
            )
        edge_rows_all = [row for rows in grouped_edges.values() for row in rows]

        edges_written = 0
        for relation, rows in grouped_edges.items():
            statement = _edge_apply_query(relation)
            for chunk in _chunks(rows, APPLY_CHUNK_ROWS):
                result = await _query(graph, statement, params={**scope, "rows": chunk})
                edges_written += _first_scalar(result, default=0)
                await _query(
                    graph,
                    PROGRESS_RECEIPT_QUERY,
                    params={
                        **receipt_key,
                        "nodes_applied": nodes_written,
                        "edges_applied": edges_written,
                    },
                )

        # Verification, not bookkeeping. Counting the rows an UNWIND bound would
        # always return the number of rows sent; this reads back from the graph
        # and only counts rows whose stored op hash is the one this envelope
        # describes, so a row that did not land is caught.
        nodes_applied = 0
        for chunk in _chunks(node_rows_all, APPLY_CHUNK_ROWS):
            verify_rows = [{"node_id": r["node_id"], "op_hash": r["op_hash"]} for r in chunk]
            nodes_applied += _first_scalar(
                await _query(graph, VERIFY_NODES_QUERY, params={**scope, "rows": verify_rows}),
                default=0,
            )

        edges_applied = 0
        for chunk in _chunks(edge_rows_all, APPLY_CHUNK_ROWS):
            verify_rows = [{"edge_id": r["edge_id"], "op_hash": r["op_hash"]} for r in chunk]
            edges_applied += _first_scalar(
                await _query(graph, VERIFY_EDGES_QUERY, params={**scope, "rows": verify_rows}),
                default=0,
            )

        if nodes_applied != len(envelope.node_ops) or edges_applied != len(envelope.edge_ops):
            # The receipt stays PENDING, so the identical envelope can be resent
            # and will converge rather than duplicate.
            raise _fail(
                500,
                "projection verified "
                f"{nodes_applied}/{len(envelope.node_ops)} nodes and "
                f"{edges_applied}/{len(envelope.edge_ops)} edges against the graph; "
                "the receipt stays PENDING and the identical envelope may be resent",
            )

        finalized = _first_props(
            await _query(
                graph,
                FINALIZE_RECEIPT_QUERY,
                params={
                    **receipt_key,
                    "status": RECEIPT_STATUS_APPLIED,
                    "nodes_applied": nodes_applied,
                    "edges_applied": edges_applied,
                    "finalized_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
        if finalized is None:
            raise _fail(500, "projection receipt could not be finalized after applying")

        memory_after = await _memory_headroom()
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "[graphiti] projection applied to %s operation_id=%s origin=%s "
            "nodes=%d edges=%d resumed=%s in %dms",
            graph_name,
            envelope.operation_id,
            envelope.origin.value,
            nodes_applied,
            edges_applied,
            resumed,
            elapsed_ms,
        )

        hashes = _all_operation_hashes(envelope)
        return ProjectionApplyResponse(
            graph_name=graph_name,
            receipt=_receipt_view(finalized),
            applied=True,
            replay=False,
            resumed=resumed,
            nodes_applied=nodes_applied,
            edges_applied=edges_applied,
            memory_before=memory_before,
            memory_after=memory_after,
            operations_total=len(hashes),
            operations_digest=_operations_digest(hashes),
            operations=_operation_probes(envelope),
            spot_probe=_spot_probe(envelope, graph_name),
            elapsed_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        # The exception text can carry Cypher and tenant property values, and this
        # lane runs over regulator content. It goes to the log, not to the caller.
        logger.exception(
            "[graphiti] projection failed correlation_id=%s graph=%s operation_id=%s: %s",
            correlation_id,
            graph_name,
            envelope.operation_id,
            exc,
        )
        raise _fail(
            500,
            {
                "code": "projection_failed",
                "message": "projection failed; see the service log for this correlation id",
                "correlation_id": correlation_id,
            },
        )


@router.post("/projection/v2/receipts", response_model=ProjectionReceiptListResponse)
async def read_projection_receipts(
    query: ProjectionReceiptQuery,
    principal: GraphPrincipal = Depends(INGEST_PRINCIPAL),
):
    """Read the projection ledger. Writes nothing.

    Without this an operator cannot answer which batches of a manifest landed, or
    inspect an operation left PENDING by a crash, except by hand-writing Cypher
    against a live tenant graph.
    """
    _guard_lane(principal, query.client_slug)
    graph_name = _derive_graph_name(query.client_slug)
    correlation_id = uuidlib.uuid4().hex

    filters = [
        f"MATCH (r:{PROJECTION_RECEIPT_LABEL} {{group_id: $group_id, "
        "engagement_id: $engagement_id})"
    ]
    params: dict[str, Any] = {
        "group_id": graph_name,
        "engagement_id": query.engagement_id,
    }
    predicates = []
    if query.operation_id is not None:
        predicates.append("r.operation_id = $operation_id")
        params["operation_id"] = query.operation_id
    if query.import_manifest_hash is not None:
        predicates.append("r.import_manifest_hash = $import_manifest_hash")
        params["import_manifest_hash"] = query.import_manifest_hash
    if predicates:
        filters.append("WHERE " + " AND ".join(predicates))
    # limit is a validated integer, never caller text.
    filters.append(f"RETURN properties(r) AS props ORDER BY r.cursor_key LIMIT {query.limit}")

    try:
        graph = _open_graph(graph_name)
        rows = _rows_of(await _query(graph, " ".join(filters), params=params))
    except Exception as exc:
        logger.exception(
            "[graphiti] projection receipt read failed correlation_id=%s graph=%s: %s",
            correlation_id,
            graph_name,
            exc,
        )
        raise _fail(
            500,
            {
                "code": "projection_receipt_read_failed",
                "message": "receipt read failed; see the service log for this correlation id",
                "correlation_id": correlation_id,
            },
        )

    return ProjectionReceiptListResponse(
        graph_name=graph_name,
        receipts=[_receipt_view(dict(row[0])) for row in rows if row and row[0]],
    )
