"""ProjectionEnvelopeV2: the governed envelope for exact-ID graph projection.

One envelope shape is emitted by both the direct import path and, later, the
Outcome projector. Everything the ratified severability contract requires of the
envelope is enforced in this module rather than in the route, so a caller cannot
construct an envelope that violates it and then have the route decide what to do
about it:

* tenant / engagement / namespace / schema version
* origin (DIRECT_IMPORT or OUTCOME_EVENT) with origin-scoped cursors whose
  sequence spaces cannot collide
* deterministic operation ID plus batch ordinal, or event ID, plus the
  import-manifest hash
* source kind / ID / immutable version ID / content hash, carried per operation,
  and the frozen-cutoff identity carried once per envelope
* deterministic node and edge operations plus a canonical envelope hash

The service never derives content. Every identity and every hash is carried by
the caller from the tenant-local relational system of record, which stays
authoritative. The graph projection is disposable and replayable and never
writes back to that system of record.

Canonical serialization, stated precisely enough that a non-Python emitter
reproduces the digest byte for byte:

1. Build the envelope object with EVERY field present, including fields the
   caller did not set. Model defaults are materialized: an omitted optional is
   the explicit JSON null, an omitted boolean is its default, and an omitted
   discriminator (``op``) is its literal default. Nothing is dropped for being
   absent or empty.
2. Remove the fields named in NON_SEMANTIC_FIELDS. envelope_hash is the caller's
   assertion about the digest, not an input to it; emitted_at is transport
   timing, so the same projection emitted twice must replay rather than conflict.
3. Normalize the values, which this module enforces by refusal rather than by
   rewriting caller content: every string is Unicode NFC, every float is finite,
   every integer is inside the exact-representable range shared with a
   JavaScript emitter, and every datetime is RFC3339 with an explicit offset.
4. Serialize with keys sorted at every level, no whitespace between tokens
   (separators "," and ":"), non-ASCII emitted as UTF-8 rather than escaped, and
   NaN/Infinity refused.
5. The digest is the SHA-256 of those UTF-8 bytes.

Operation order is preserved because node_ops and edge_ops are ordered arrays.

Vocabulary provenance: the P3 node labels and relationship types denied below are
quoted from the ratified ontology lock (":Workflow", ":ProcessVersion", ":SOP",
":RaciMatrix", ":RaciAssignment"; GOVERNS, VERSION_OF, ACCOUNTABLE_FOR,
RESPONSIBLE_FOR, CONSULTED, INFORMED, OWNED_BY), as is ":Finding" and the
DERIVED_FROM provenance edge. The two dependent-record tokens for a Finding's
direct driver and its receipts are named in the phase map but not spelled in the
lock; they are isolated in ProjectionSourceKind so a correction is one enum edit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

# The only envelope schema this service applies today. A future schema is a new
# literal plus an explicit apply path, never a silent reinterpretation.
PROJECTION_SCHEMA_VERSION = "projection.v2"

# Fields excluded from the canonical envelope hash. See step 2 above.
NON_SEMANTIC_FIELDS = frozenset({"envelope_hash", "emitted_at"})

# A single governed request stays bounded. The manifest is walked in batches and
# each batch carries its own ordinal in the DIRECT_IMPORT cursor, so these limits
# constrain one request and never the size of a corpus. None of them is a client
# fact or an acceptance constant.
MAX_OPERATIONS_PER_ENVELOPE = 5000
MAX_PROPERTIES_PER_OPERATION = 128
MAX_PROPERTY_STRING_LENGTH = 32768
MAX_PROPERTY_LIST_LENGTH = 512
MAX_CANONICAL_ENVELOPE_BYTES = 16 * 1024 * 1024

# Integers outside this range cannot survive a JavaScript emitter, and A11
# requires one envelope that the Python direct import and the later Outcome
# projector both produce. Refusing here keeps the digest reproducible across
# emitters instead of diverging silently.
MAX_EXACT_INTEGER = 2**53 - 1

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_SHA256_HEX = r"^[0-9a-f]{64}$"
_CLIENT_SLUG = r"^[a-z0-9-]+$"

# Every projected node carries this label, and the exact-ID MERGE keys on it.
# Node identity is therefore independent of a node's type label: a node whose
# type changes can never split into two rows sharing one uuid, which is what
# "reprojecting the same immutable source version converges" requires.
PROJECTION_NODE_LABEL = "ProjectionNode"

# The ledger label. It is not in the projection label space and no caller can
# write it, because callers do not supply labels at all.
PROJECTION_RECEIPT_LABEL = "ProjectionReceipt"

# P3 vocabulary, quoted from the ratified ontology lock. These are refused today
# with an error that names P3, so a caller who sends them learns the phase rule
# rather than a generic validation message.
P3_NODE_LABELS = frozenset(
    {"Workflow", "ProcessVersion", "SOP", "RaciMatrix", "RaciAssignment"}
)
P3_EDGE_RELATIONS = frozenset(
    {
        "GOVERNS",
        "VERSION_OF",
        "ACCOUNTABLE_FOR",
        "RESPONSIBLE_FOR",
        "CONSULTED",
        "INFORMED",
        "OWNED_BY",
    }
)

# Ratified, but it links a Finding to P3 node types (workflow, system,
# department), so it cannot be projected before those types exist.
P3_DEPENDENT_EDGE_RELATIONS = frozenset({"EVIDENCES"})


class ProjectionOrigin(str, Enum):
    """Which emitter produced this envelope. Cursors are scoped by this value."""

    DIRECT_IMPORT = "DIRECT_IMPORT"
    OUTCOME_EVENT = "OUTCOME_EVENT"


class ProjectionSourceKind(str, Enum):
    """Source kinds the envelope is shaped to carry.

    P2-prime admits exact-ID/hash Finding import only, plus a Finding's direct
    driver and its receipts. The workflow, process-version, SOP, and RACI kinds
    are P3 backfill: the envelope carries them so the shape does not change
    later, and this phase refuses them.
    """

    FINDING = "FINDING"
    FINDING_DRIVER = "FINDING_DRIVER"
    FINDING_RECEIPT = "FINDING_RECEIPT"
    WORKFLOW = "WORKFLOW"
    PROCESS_VERSION = "PROCESS_VERSION"
    SOP = "SOP"
    RACI = "RACI"


# The node type label written for each admitted kind. A caller never supplies a
# label, so a P3 label cannot be smuggled in past a phase gate that only reads
# the declared kind.
SOURCE_KIND_NODE_LABEL: dict[ProjectionSourceKind, str] = {
    ProjectionSourceKind.FINDING: "Finding",
    ProjectionSourceKind.FINDING_DRIVER: "FindingDriver",
    ProjectionSourceKind.FINDING_RECEIPT: "FindingReceipt",
    ProjectionSourceKind.WORKFLOW: "Workflow",
    ProjectionSourceKind.PROCESS_VERSION: "ProcessVersion",
    ProjectionSourceKind.SOP: "SOP",
    ProjectionSourceKind.RACI: "RaciAssignment",
}

P2_PRIME_SOURCE_KINDS = frozenset(
    {
        ProjectionSourceKind.FINDING,
        ProjectionSourceKind.FINDING_DRIVER,
        ProjectionSourceKind.FINDING_RECEIPT,
    }
)

P3_SOURCE_KINDS = frozenset(
    {
        ProjectionSourceKind.WORKFLOW,
        ProjectionSourceKind.PROCESS_VERSION,
        ProjectionSourceKind.SOP,
        ProjectionSourceKind.RACI,
    }
)


class ProjectionRelation(str, Enum):
    """The relationship types P2-prime may project.

    DERIVED_FROM is the ratified provenance edge. ATTESTED_BY carries a Finding
    to one of its receipts. Everything else, including the ratified EVIDENCES
    edge, points at P3 node types and is refused by ProjectionEdgeOp.
    """

    DERIVED_FROM = "DERIVED_FROM"
    ATTESTED_BY = "ATTESTED_BY"


# Which endpoint kinds each relation may join. An edge whose endpoints do not
# match is refused, so a Finding cannot be recorded as attested by another
# Finding, and a driver cannot be given a provenance edge it does not have.
RELATION_ENDPOINT_KINDS: dict[ProjectionRelation, tuple[ProjectionSourceKind, ProjectionSourceKind]] = {
    ProjectionRelation.DERIVED_FROM: (
        ProjectionSourceKind.FINDING,
        ProjectionSourceKind.FINDING_DRIVER,
    ),
    ProjectionRelation.ATTESTED_BY: (
        ProjectionSourceKind.FINDING,
        ProjectionSourceKind.FINDING_RECEIPT,
    ),
}


class FindingLifecycleState(str, Enum):
    """The declared lifecycle states, in the ratified precedence order.

    Precedence: a merge target wins, then rejection, then a provable named-human
    validation, then unreviewed import. SUPERSEDED is classified separately and
    only when the caller can prove it with the superseding identity.
    """

    MERGED = "MERGED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    RETAINED_HUMAN = "RETAINED_HUMAN"
    RETAINED_IMPORT_UNREVIEWED = "RETAINED_IMPORT_UNREVIEWED"


# Only these states are retained, and only retained states may feed retrieval.
RETAINED_LIFECYCLE_STATES = frozenset(
    {
        FindingLifecycleState.RETAINED_HUMAN,
        FindingLifecycleState.RETAINED_IMPORT_UNREVIEWED,
    }
)


# Property keys the projection service writes itself. A caller supplying one of
# these could otherwise restate tenancy, provenance, or human validation.
SERVICE_OWNED_PROPERTIES = frozenset(
    {
        "uuid",
        "group_id",
        "engagement_id",
        "projection_schema_version",
        "projection_origin",
        "projection_op_hash",
        "projection_source_kind",
        "projection_source_id",
        "projection_source_version_id",
        "projection_source_content_hash",
        "projection_cutoff_id",
        "projection_cutoff_at",
        "lifecycle_state",
        "lifecycle_retained",
        "lifecycle_human_validated",
        "lifecycle_validated_by",
        "lifecycle_validator_subject_id",
        "lifecycle_validated_at",
        "lifecycle_validated_version_id",
        "lifecycle_validated_content_hash",
        "lifecycle_legacy_validated_flag",
        "lifecycle_merged_into_id",
        "lifecycle_superseded_by_id",
        # Derived vectors and their provenance marker are written by the
        # embedding path from text, never carried in a projection.
        "name_embedding",
        "fact_embedding",
        "emb_model",
    }
)


def _check_scalar(key: str, value: Any) -> None:
    """Refuse any scalar whose canonical form would not survive a round trip.

    bool is checked before int because bool is an int subclass in Python.
    """
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > MAX_EXACT_INTEGER:
            raise ValueError(
                f"property '{key}' carries an integer outside the exactly "
                f"representable range (+/-{MAX_EXACT_INTEGER}); it could not be "
                "reproduced by a non-Python emitter of the same envelope"
            )
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"property '{key}' is NaN or Infinity, which has no canonical "
                "JSON form; the canonical hash could not distinguish it from "
                "null while the graph would store the raw value"
            )
        return
    if isinstance(value, str):
        if len(value) > MAX_PROPERTY_STRING_LENGTH:
            raise ValueError(
                f"property '{key}' exceeds {MAX_PROPERTY_STRING_LENGTH} characters"
            )
        if unicodedata.normalize("NFC", value) != value:
            raise ValueError(
                f"property '{key}' is not Unicode NFC; the canonical hash is "
                "taken over the bytes as sent, so a non-NFC value would make an "
                "otherwise identical envelope hash differently. Normalize to NFC "
                "at the emitter rather than having this service rewrite content"
            )
        return
    raise ValueError(
        f"property '{key}' may only be a scalar or a flat list of scalars"
    )


def _check_property_value(key: str, value: Any) -> None:
    if isinstance(value, list):
        if len(value) > MAX_PROPERTY_LIST_LENGTH:
            raise ValueError(
                f"property '{key}' exceeds {MAX_PROPERTY_LIST_LENGTH} list items"
            )
        for item in value:
            if isinstance(item, list) or isinstance(item, dict):
                raise ValueError(
                    f"property '{key}' may only contain scalars or a flat list "
                    "of scalars"
                )
            _check_scalar(key, item)
        return
    if isinstance(value, dict):
        raise ValueError(
            f"property '{key}' may only be a scalar or a flat list of scalars"
        )
    _check_scalar(key, value)


def _check_property_map(properties: dict[str, Any]) -> None:
    """Property keys and values must be safe, flat, and caller-owned.

    The key grammar is load-bearing beyond tidiness. The FalkorDB client
    serializes query parameters into the Cypher text itself and emits map keys
    unquoted, so a key outside this grammar would be Cypher, not data.
    """
    if len(properties) > MAX_PROPERTIES_PER_OPERATION:
        raise ValueError(
            f"an operation carries at most {MAX_PROPERTIES_PER_OPERATION} properties"
        )
    for key, value in properties.items():
        if not _IDENTIFIER.fullmatch(key):
            raise ValueError(
                f"property name '{key}' must match [A-Za-z_][A-Za-z0-9_]{{0,63}}"
            )
        if key in SERVICE_OWNED_PROPERTIES:
            raise ValueError(
                f"property '{key}' is written by the projection service and "
                "cannot be supplied by the caller"
            )
        _check_property_value(key, value)


def _require_nfc(field: str, value: str) -> str:
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field} must be Unicode NFC")
    return value


class ProjectionSource(BaseModel):
    """The immutable source identity one projected row is anchored to.

    This sits on the operation, not on the envelope. A governed bulk envelope
    carries many source records, so an envelope-level source identity would
    stamp one record's ID, version, and content hash onto every other row in the
    batch. The P2 gate's spot-probe element exists to recompute a projected row
    against its own source record, and that is only possible when each row
    carries its own identity.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ProjectionSourceKind
    id: str = Field(min_length=1, max_length=256)
    immutable_version_id: str = Field(min_length=1, max_length=256)
    content_hash: str = Field(pattern=_SHA256_HEX)

    @model_validator(mode="after")
    def _validate_source(self) -> "ProjectionSource":
        _require_nfc("source.id", self.id)
        _require_nfc("source.immutable_version_id", self.immutable_version_id)
        if self.kind in P3_SOURCE_KINDS:
            raise ValueError(
                f"source.kind {self.kind.value} is P3 backfill (exact workflow, "
                "ProcessVersion, SOP, and published-to-date RACI) and is refused "
                "in this phase; P2-prime admits "
                f"{sorted(k.value for k in P2_PRIME_SOURCE_KINDS)}"
            )
        if self.kind not in P2_PRIME_SOURCE_KINDS:
            raise ValueError(
                f"source.kind {self.kind.value} is outside the P2-prime allowlist "
                f"{sorted(k.value for k in P2_PRIME_SOURCE_KINDS)}"
            )
        return self

    @property
    def node_label(self) -> str:
        return SOURCE_KIND_NODE_LABEL[self.kind]


class ProjectionCutoff(BaseModel):
    """The frozen-cutoff identity the whole envelope is taken against.

    Unlike source identity this genuinely is batch-level: every row in an import
    is read at one cutoff, and the manifest freezes at that cutoff.
    """

    model_config = ConfigDict(extra="forbid")

    cutoff_id: str = Field(min_length=1, max_length=256)
    cutoff_at: datetime | None = None


class FindingLifecycle(BaseModel):
    """The retention classification carried with a projected Finding.

    Nothing here is derived by this service. The caller states the facts held by
    the authoritative relational store and this model refuses any combination
    that contradicts the ratified precedence, so an unreviewed import can never
    be recorded, or later rendered, as named-human validation.

    RETAINED_HUMAN is the state the ratified lifecycle calls provable, and it is
    only accepted with the two things the instrument requires to be provable:
    actor identity, and a binding to the exact version and content the validator
    saw. ProjectionNodeOp checks that binding against the operation's own source
    identity, so a review cannot appear to adjudicate a statement it did not see.

    A legacy validation flag that cannot be proven does not become
    RETAINED_HUMAN. It persists as audit metadata on legacy_validated_flag,
    which never sets lifecycle_human_validated.
    """

    model_config = ConfigDict(extra="forbid")

    state: FindingLifecycleState
    merged_into_id: str | None = Field(default=None, min_length=1, max_length=256)
    is_rejected: bool = False
    superseded_by_id: str | None = Field(default=None, min_length=1, max_length=256)
    validated_by: str | None = Field(default=None, min_length=1, max_length=512)
    validator_subject_id: str | None = Field(default=None, min_length=1, max_length=256)
    validated_at: datetime | None = None
    validated_version_id: str | None = Field(default=None, min_length=1, max_length=256)
    validated_content_hash: str | None = Field(default=None, pattern=_SHA256_HEX)
    legacy_validated_flag: bool = False

    @property
    def retained(self) -> bool:
        return self.state in RETAINED_LIFECYCLE_STATES

    @property
    def human_validated(self) -> bool:
        return self.state is FindingLifecycleState.RETAINED_HUMAN

    @model_validator(mode="after")
    def _enforce_precedence(self) -> "FindingLifecycle":
        state = self.state
        if self.merged_into_id is not None and state is not FindingLifecycleState.MERGED:
            raise ValueError(
                "lifecycle precedence: a finding with merged_into_id set is MERGED, "
                f"not {state.value}"
            )
        if (
            self.merged_into_id is None
            and self.is_rejected
            and state is not FindingLifecycleState.REJECTED
        ):
            raise ValueError(
                "lifecycle precedence: a rejected finding that is not merged is "
                f"REJECTED, not {state.value}"
            )
        if state is FindingLifecycleState.MERGED and self.merged_into_id is None:
            raise ValueError("MERGED requires merged_into_id")
        if state is FindingLifecycleState.REJECTED and not self.is_rejected:
            raise ValueError("REJECTED requires is_rejected=true")
        if state is FindingLifecycleState.SUPERSEDED and self.superseded_by_id is None:
            raise ValueError(
                "SUPERSEDED is classified separately and only when provable: "
                "superseded_by_id is required"
            )
        if state is not FindingLifecycleState.SUPERSEDED and self.superseded_by_id is not None:
            raise ValueError(
                f"superseded_by_id is only meaningful for SUPERSEDED, not {state.value}"
            )

        validation_fields = {
            "validated_by": self.validated_by,
            "validator_subject_id": self.validator_subject_id,
            "validated_at": self.validated_at,
            "validated_version_id": self.validated_version_id,
            "validated_content_hash": self.validated_content_hash,
        }
        if state is FindingLifecycleState.RETAINED_HUMAN:
            missing = sorted(name for name, value in validation_fields.items() if value is None)
            if missing:
                raise ValueError(
                    "RETAINED_HUMAN is the provable state and requires actor "
                    "identity plus a version and content binding; missing "
                    f"{missing}. A validation that cannot be proven stays "
                    "RETAINED_IMPORT_UNREVIEWED and carries "
                    "legacy_validated_flag=true as audit metadata instead"
                )
            if self.validated_at.tzinfo is None:
                raise ValueError("validated_at requires an explicit UTC offset")
            _require_nfc("lifecycle.validated_by", self.validated_by)
            _require_nfc("lifecycle.validator_subject_id", self.validator_subject_id)
        else:
            present = sorted(name for name, value in validation_fields.items() if value is not None)
            if present:
                raise ValueError(
                    f"{state.value} must not carry named-human validation; only "
                    f"RETAINED_HUMAN may carry {present}"
                )
        return self


class ProjectionNodeOp(BaseModel):
    """One exact-ID node upsert. node_id is the caller's deterministic identity.

    There is deliberately no label field. The node's type label is a pure
    function of its declared source kind, so a phase gate that admits a kind
    cannot then have a different kind's label written under it, and the exact-ID
    MERGE key never depends on caller-ordered label text.
    """

    model_config = ConfigDict(extra="forbid")

    op: Literal["MERGE_NODE"] = "MERGE_NODE"
    node_id: str = Field(min_length=1, max_length=256)
    source: ProjectionSource
    properties: dict[str, Any] = Field(default_factory=dict)
    lifecycle: FindingLifecycle | None = None

    @property
    def node_label(self) -> str:
        return self.source.node_label

    @model_validator(mode="after")
    def _validate_node(self) -> "ProjectionNodeOp":
        _require_nfc("node_id", self.node_id)
        _check_property_map(self.properties)
        is_finding = self.source.kind is ProjectionSourceKind.FINDING
        if is_finding and self.lifecycle is None:
            raise ValueError(
                "a FINDING node must carry an explicit lifecycle; only declared "
                "retained states may feed retrieval"
            )
        if not is_finding and self.lifecycle is not None:
            raise ValueError("lifecycle is only carried by FINDING nodes")
        if self.lifecycle is not None and self.lifecycle.human_validated:
            if self.lifecycle.validated_version_id != self.source.immutable_version_id:
                raise ValueError(
                    "RETAINED_HUMAN validated_version_id must equal this "
                    "operation's source.immutable_version_id; a review cannot "
                    "adjudicate a version it did not see"
                )
            if self.lifecycle.validated_content_hash != self.source.content_hash:
                raise ValueError(
                    "RETAINED_HUMAN validated_content_hash must equal this "
                    "operation's source.content_hash; a review cannot adjudicate "
                    "a statement it did not see"
                )
        return self


class ProjectionEdgeOp(BaseModel):
    """One exact-ID edge upsert between two exact node identities."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["MERGE_EDGE"] = "MERGE_EDGE"
    edge_id: str = Field(min_length=1, max_length=256)
    relation: str = Field(min_length=1, max_length=64)
    from_node_id: str = Field(min_length=1, max_length=256)
    to_node_id: str = Field(min_length=1, max_length=256)
    properties: dict[str, Any] = Field(default_factory=dict)

    @property
    def relation_enum(self) -> ProjectionRelation:
        return ProjectionRelation(self.relation)

    @model_validator(mode="after")
    def _validate_edge(self) -> "ProjectionEdgeOp":
        _require_nfc("edge_id", self.edge_id)
        _require_nfc("from_node_id", self.from_node_id)
        _require_nfc("to_node_id", self.to_node_id)
        if not _IDENTIFIER.fullmatch(self.relation):
            raise ValueError(
                f"relation '{self.relation}' must match [A-Za-z_][A-Za-z0-9_]{{0,63}}"
            )
        if self.relation in P3_EDGE_RELATIONS:
            raise ValueError(
                f"relation '{self.relation}' belongs to the P3 backfill vocabulary "
                "(exact workflow, ProcessVersion, SOP, and published-to-date RACI) "
                "and is refused in this phase"
            )
        if self.relation in P3_DEPENDENT_EDGE_RELATIONS:
            raise ValueError(
                f"relation '{self.relation}' joins a Finding to P3 node types and "
                "cannot be projected before those types exist; it is refused in "
                "this phase"
            )
        if self.relation not in {r.value for r in ProjectionRelation}:
            raise ValueError(
                f"relation '{self.relation}' is outside the P2-prime allowlist "
                f"{sorted(r.value for r in ProjectionRelation)}"
            )
        if self.from_node_id == self.to_node_id:
            raise ValueError("an edge cannot join a node to itself")
        _check_property_map(self.properties)
        return self


class ProjectionCursor(BaseModel):
    """Origin-scoped cursor fields.

    Which fields are required, and which are refused, is decided by the
    envelope's declared origin. The two sequence spaces are kept apart at the
    key level by cursor_key(), which prefixes every key with its origin.
    """

    model_config = ConfigDict(extra="forbid")

    import_manifest_hash: str | None = Field(default=None, pattern=_SHA256_HEX)
    batch_ordinal: int | None = Field(default=None, ge=0, le=MAX_EXACT_INTEGER)
    event_id: str | None = Field(default=None, min_length=1, max_length=256)


def canonical_json(payload: Any) -> str:
    """The one canonical serialization used for every hash in this module."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def operation_hash(op: ProjectionNodeOp | ProjectionEdgeOp) -> str:
    """Per-operation digest.

    It is a pure function of the operation's own semantic content, including its
    source identity, and carries no envelope-scoped or wall-clock value. That is
    what lets it be written onto the row: reprojecting the same immutable source
    version under a different operation ID produces a byte-identical row, and the
    apply can still verify from the graph that the row it just wrote is the row
    this operation describes.
    """
    return _sha256_of(op.model_dump(mode="json"))


class ProjectionEnvelopeV2(BaseModel):
    """The governed projection envelope.

    client_slug is deliberately a top-level field: the request signature binds
    the tenant claim to the top-level client_slug of the JSON body, so nesting
    the tenant anywhere else would make every signed request fail closed.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["projection.v2"] = PROJECTION_SCHEMA_VERSION
    client_slug: str = Field(pattern=_CLIENT_SLUG, min_length=1, max_length=128)
    namespace: str = Field(min_length=1, max_length=256)
    engagement_id: str = Field(min_length=1, max_length=256)
    origin: ProjectionOrigin
    cursor: ProjectionCursor
    operation_id: str = Field(min_length=1, max_length=256)
    cutoff: ProjectionCutoff
    node_ops: list[ProjectionNodeOp] = Field(default_factory=list)
    edge_ops: list[ProjectionEdgeOp] = Field(default_factory=list)
    envelope_hash: str | None = Field(default=None, pattern=_SHA256_HEX)
    emitted_at: datetime | None = None

    _canonical_hash: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def _validate_envelope(self) -> "ProjectionEnvelopeV2":
        _require_nfc("engagement_id", self.engagement_id)
        _require_nfc("operation_id", self.operation_id)
        self._enforce_cursor_scope()
        self._enforce_operation_budget()

        serialized = canonical_json(self.canonical_payload())
        encoded = serialized.encode("utf-8")
        if len(encoded) > MAX_CANONICAL_ENVELOPE_BYTES:
            raise ValueError(
                f"the canonical envelope is {len(encoded)} bytes, over the "
                f"{MAX_CANONICAL_ENVELOPE_BYTES} byte limit; split it across "
                "batch ordinals of the same manifest"
            )
        self._canonical_hash = hashlib.sha256(encoded).hexdigest()

        stated = self.envelope_hash
        if stated is not None and stated != self._canonical_hash:
            raise ValueError(
                "envelope_hash does not match the canonical hash of this "
                "envelope. The canonical form materializes every model default, "
                "drops only envelope_hash and emitted_at, and sorts keys at every "
                "level; see app/models/projection.py for the exact recipe"
            )
        return self

    def _enforce_cursor_scope(self) -> None:
        cursor = self.cursor
        if self.origin is ProjectionOrigin.DIRECT_IMPORT:
            if cursor.import_manifest_hash is None or cursor.batch_ordinal is None:
                raise ValueError(
                    "DIRECT_IMPORT requires cursor.import_manifest_hash and "
                    "cursor.batch_ordinal"
                )
            if cursor.event_id is not None:
                raise ValueError(
                    "cursor.event_id belongs to the OUTCOME_EVENT sequence space "
                    "and cannot be carried by a DIRECT_IMPORT envelope"
                )
            return
        if cursor.event_id is None:
            raise ValueError("OUTCOME_EVENT requires cursor.event_id")
        if cursor.import_manifest_hash is not None or cursor.batch_ordinal is not None:
            raise ValueError(
                "cursor.import_manifest_hash and cursor.batch_ordinal belong to the "
                "DIRECT_IMPORT sequence space and cannot be carried by an "
                "OUTCOME_EVENT envelope"
            )

    def _enforce_operation_budget(self) -> None:
        total = len(self.node_ops) + len(self.edge_ops)
        if total == 0:
            raise ValueError(
                "an envelope must carry at least one node or edge operation; an "
                "empty envelope would put an entry in the ledger that projects "
                "nothing"
            )
        if total > MAX_OPERATIONS_PER_ENVELOPE:
            raise ValueError(
                f"an envelope carries at most {MAX_OPERATIONS_PER_ENVELOPE} "
                f"operations; this one carries {total}. Split it across batch "
                "ordinals of the same manifest"
            )
        node_ids = [op.node_id for op in self.node_ops]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_id is repeated within the envelope")
        edge_ids = [op.edge_id for op in self.edge_ops]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("edge_id is repeated within the envelope")

    def canonical_payload(self) -> dict[str, Any]:
        """The envelope's semantic content, with explicit nulls preserved."""
        payload = self.model_dump(mode="json")
        for field in NON_SEMANTIC_FIELDS:
            payload.pop(field, None)
        return payload

    def canonical_hash(self) -> str:
        """SHA-256 over the canonical serialization of the semantic content."""
        return self._canonical_hash

    def cursor_space(self) -> str:
        """The name of the sequence space this envelope's cursor belongs to."""
        return self.origin.value

    def cursor_key(self) -> str:
        """The stored and compared form of the cursor.

        Every key is prefixed with its origin, so a DIRECT_IMPORT cursor and an
        OUTCOME_EVENT cursor can never alias even when their raw component values
        are identical. The component payload is canonical JSON, so a separator
        inside a caller-supplied event ID cannot forge another key either.

        The router enforces that one cursor position in one tenant is claimed by
        at most one operation ID, so this is a position in a sequence rather than
        a decorative property.
        """
        if self.origin is ProjectionOrigin.DIRECT_IMPORT:
            payload: dict[str, Any] = {
                "import_manifest_hash": self.cursor.import_manifest_hash,
                "batch_ordinal": self.cursor.batch_ordinal,
            }
        else:
            payload = {"event_id": self.cursor.event_id}
        return f"{self.cursor_space()}|{canonical_json(payload)}"

    def node_kinds(self) -> dict[str, ProjectionSourceKind]:
        """Declared kind for every node this envelope creates or restates."""
        return {op.node_id: op.source.kind for op in self.node_ops}
