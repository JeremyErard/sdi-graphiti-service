"""Dormant, metadata-only provenance audit and backfill planning.

Nothing in this module runs at import or service startup. The planner is pure;
``run_provenance_audit`` performs metadata-only reads by default and rejects
``apply=True`` before database access until the cardinality guard is proven on a
disposable FalkorDB instance. The private executor exists for unit verification.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable
import uuid as uuidlib

from app.config import settings
from app.graph_names import graph_name_for_client
from app.models.episode import EpisodeType
from app.provenance_contract import LEGACY_EPISODE_CONTRACT_VERSION


_KIND = r"[a-z][a-z0-9_-]{0,63}"
_IDENTIFIER = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}"
_NAME_PATTERN = re.compile(
    rf"(?P<episode_type>{_KIND}): (?P<source_type>{_KIND})/"
    rf"(?P<source_id>{_IDENTIFIER})"
)
_DESCRIPTION_PATTERN = re.compile(
    rf"Engagement (?P<engagement_id>{_IDENTIFIER}) — "
    rf"(?P<episode_type>{_KIND}) from (?P<source_type>{_KIND})"
)
_EPISODE_TYPES = frozenset(item.value for item in EpisodeType)
_ANCHOR_FIELDS = (
    "source_id",
    "source_type",
    "engagement_id",
    "episode_type",
    "anchor_mode",
    "producer_contract_version",
)
APPLY_BLOCKED_CODE = "APPLY_BLOCKED_CARDINALITY_GUARD_UNVERIFIED"


class ApplyBlockedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedLegacyAnchor:
    source_id: str
    source_type: str
    engagement_id: str
    episode_type: str
    anchor_mode: str = LEGACY_EPISODE_CONTRACT_VERSION
    producer_contract_version: str = LEGACY_EPISODE_CONTRACT_VERSION


@dataclass(frozen=True)
class EpisodeRecord:
    uuid: Any
    name: Any
    source_description: Any
    source_id: Any = None
    source_type: Any = None
    engagement_id: Any = None
    episode_type: Any = None
    anchor_mode: Any = None
    producer_contract_version: Any = None


@dataclass(frozen=True)
class EdgeRecord:
    uuid: Any
    source_uuid: Any
    target_uuid: Any
    actual_source_uuid: Any
    actual_target_uuid: Any
    episodes: Any


@dataclass(frozen=True)
class EpisodeUpdate:
    uuid: str
    anchor: ParsedLegacyAnchor


@dataclass(frozen=True)
class EdgeUpdate:
    uuid: str
    source_uuid: str
    target_uuid: str
    set_endpoints: bool
    episodes: tuple[str, ...] | None
    set_episodes: bool


@dataclass(frozen=True)
class ProvenancePlan:
    episodes_scanned: int
    edges_scanned: int
    episode_updates: tuple[EpisodeUpdate, ...]
    edge_updates: tuple[EdgeUpdate, ...]
    codes: dict[str, int]

    def summary(
        self,
        *,
        apply: bool,
        apply_succeeded: int = 0,
        apply_conflicts: int = 0,
    ) -> dict[str, Any]:
        """Return only bounded counts and stable codes—never graph values."""

        planned_edge_endpoint_updates = sum(
            update.set_endpoints for update in self.edge_updates
        )
        planned_episode_list_updates = sum(
            update.set_episodes for update in self.edge_updates
        )
        return {
            "mode": "apply" if apply else "audit",
            "counts": {
                "episodes_scanned": self.episodes_scanned,
                "edges_scanned": self.edges_scanned,
                "episode_anchor_updates_planned": len(self.episode_updates),
                "edge_endpoint_updates_planned": planned_edge_endpoint_updates,
                "edge_episode_list_updates_planned": planned_episode_list_updates,
                "apply_attempted": (
                    len(self.episode_updates) + len(self.edge_updates)
                    if apply
                    else 0
                ),
                "apply_succeeded": apply_succeeded,
                "apply_conflicts": apply_conflicts,
            },
            "codes": dict(sorted(self.codes.items())),
        }


def canonical_uuid(value: Any) -> str | None:
    if not isinstance(value, (str, uuidlib.UUID)):
        return None
    try:
        return str(uuidlib.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def parse_exact_legacy_anchor(
    name: Any,
    source_description: Any,
) -> tuple[ParsedLegacyAnchor | None, str]:
    """Parse only the two exact legacy formats emitted by ``ingest_episode``."""

    if not isinstance(name, str) or not isinstance(source_description, str):
        return None, "EPISODE_UNRESOLVED_FORMAT"
    if len(name) > 512 or len(source_description) > 512:
        return None, "EPISODE_UNRESOLVED_FORMAT"
    name_match = _NAME_PATTERN.fullmatch(name)
    description_match = _DESCRIPTION_PATTERN.fullmatch(source_description)
    if name_match is None or description_match is None:
        return None, "EPISODE_UNRESOLVED_FORMAT"
    name_values = name_match.groupdict()
    description_values = description_match.groupdict()
    if name_values["episode_type"] not in _EPISODE_TYPES:
        return None, "EPISODE_UNRESOLVED_EPISODE_TYPE"
    if (
        name_values["episode_type"] != description_values["episode_type"]
        or name_values["source_type"] != description_values["source_type"]
    ):
        return None, "EPISODE_UNRESOLVED_MISMATCH"
    return (
        ParsedLegacyAnchor(
            source_id=name_values["source_id"],
            source_type=name_values["source_type"],
            engagement_id=description_values["engagement_id"],
            episode_type=name_values["episode_type"],
        ),
        "EPISODE_CANONICAL",
    )


def normalize_provable_episode_list(
    value: Any,
    *,
    known_episode_ids: frozenset[str],
) -> tuple[tuple[str, ...] | None, bool]:
    """Return a canonical list only when representation and references prove it."""

    candidate = value
    representation_is_list = isinstance(candidate, list)
    if isinstance(candidate, str):
        if len(candidate) > 100_000:
            return None, False
        try:
            candidate = ast.literal_eval(candidate)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return None, False
    if not isinstance(candidate, list):
        return None, False

    normalized: list[str] = []
    for item in candidate:
        episode_id = canonical_uuid(item)
        if episode_id is None or episode_id not in known_episode_ids:
            return None, False
        normalized.append(episode_id)
    canonical = tuple(normalized)
    already_normalized = representation_is_list and candidate == list(canonical)
    return canonical, already_normalized


def _field_value(record: EpisodeRecord, field: str) -> Any:
    return getattr(record, field)


def _expected_value(anchor: ParsedLegacyAnchor, field: str) -> str:
    return getattr(anchor, field)


def build_provenance_plan(
    episodes: Iterable[EpisodeRecord],
    edges: Iterable[EdgeRecord],
) -> ProvenancePlan:
    """Build a deterministic, idempotent plan from metadata-only observations."""

    episode_rows = tuple(episodes)
    edge_rows = tuple(edges)
    codes: Counter[str] = Counter()

    episode_groups: dict[str, list[EpisodeRecord]] = {}
    for episode in episode_rows:
        episode_id = canonical_uuid(episode.uuid)
        if episode_id is None:
            codes["EPISODE_UNRESOLVED_ID"] += 1
            continue
        episode_groups.setdefault(episode_id, []).append(episode)

    unique_episode_ids = frozenset(
        episode_id
        for episode_id, rows in episode_groups.items()
        if len(rows) == 1
    )
    episode_updates: list[EpisodeUpdate] = []
    for episode_id in sorted(episode_groups):
        rows = episode_groups[episode_id]
        if len(rows) != 1:
            codes["EPISODE_UNRESOLVED_DUPLICATE_ID"] += 1
            continue
        record = rows[0]
        anchor, parse_code = parse_exact_legacy_anchor(
            record.name,
            record.source_description,
        )
        if anchor is None:
            codes[parse_code] += 1
            continue
        conflicts = any(
            _field_value(record, field) not in (None, "", _expected_value(anchor, field))
            for field in _ANCHOR_FIELDS
        )
        if conflicts:
            codes["EPISODE_UNRESOLVED_ANCHOR_CONFLICT"] += 1
            continue
        if all(
            _field_value(record, field) == _expected_value(anchor, field)
            for field in _ANCHOR_FIELDS
        ):
            codes["EPISODE_ALREADY_ANCHORED"] += 1
            continue
        episode_updates.append(EpisodeUpdate(uuid=episode_id, anchor=anchor))
        codes["EPISODE_ANCHOR_UPDATE_READY"] += 1

    edge_groups: dict[str, list[EdgeRecord]] = {}
    for edge in edge_rows:
        edge_id = canonical_uuid(edge.uuid)
        if edge_id is None:
            codes["EDGE_UNRESOLVED_ID"] += 1
            continue
        edge_groups.setdefault(edge_id, []).append(edge)

    edge_updates: list[EdgeUpdate] = []
    for edge_id in sorted(edge_groups):
        rows = edge_groups[edge_id]
        if len(rows) != 1:
            codes["EDGE_UNRESOLVED_DUPLICATE_ID"] += 1
            continue
        record = rows[0]
        actual_source = canonical_uuid(record.actual_source_uuid)
        actual_target = canonical_uuid(record.actual_target_uuid)
        if actual_source is None or actual_target is None:
            codes["EDGE_UNRESOLVED_ENDPOINT_ID"] += 1
            continue
        current_source = canonical_uuid(record.source_uuid)
        current_target = canonical_uuid(record.target_uuid)
        set_endpoints = (
            current_source != actual_source or current_target != actual_target
        )
        codes[
            "EDGE_ENDPOINT_UPDATE_READY"
            if set_endpoints
            else "EDGE_ENDPOINT_ALREADY_REPAIRED"
        ] += 1

        normalized_episodes, already_normalized = normalize_provable_episode_list(
            record.episodes,
            known_episode_ids=unique_episode_ids,
        )
        set_episodes = normalized_episodes is not None and not already_normalized
        if normalized_episodes is None:
            codes["EDGE_UNRESOLVED_EPISODE_LIST"] += 1
        elif set_episodes:
            codes["EDGE_EPISODE_LIST_UPDATE_READY"] += 1
        else:
            codes["EDGE_EPISODE_LIST_ALREADY_NORMALIZED"] += 1

        if set_endpoints or set_episodes:
            edge_updates.append(
                EdgeUpdate(
                    uuid=edge_id,
                    source_uuid=actual_source,
                    target_uuid=actual_target,
                    set_endpoints=set_endpoints,
                    episodes=normalized_episodes,
                    set_episodes=set_episodes,
                )
            )

    return ProvenancePlan(
        episodes_scanned=len(episode_rows),
        edges_scanned=len(edge_rows),
        episode_updates=tuple(episode_updates),
        edge_updates=tuple(edge_updates),
        codes=dict(codes),
    )


def _episode_records(rows: Iterable[Any]) -> tuple[EpisodeRecord, ...]:
    records: list[EpisodeRecord] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 9:
            records.append(EpisodeRecord(None, None, None))
            continue
        records.append(EpisodeRecord(*row[:9]))
    return tuple(records)


def _edge_records(rows: Iterable[Any]) -> tuple[EdgeRecord, ...]:
    records: list[EdgeRecord] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            records.append(EdgeRecord(None, None, None, None, None, None))
            continue
        records.append(EdgeRecord(*row[:6]))
    return tuple(records)


def _result_count(result: Any) -> int:
    rows = getattr(result, "result_set", None)
    if not isinstance(rows, list) or len(rows) != 1 or not rows[0]:
        return 0
    try:
        return int(rows[0][0])
    except (TypeError, ValueError):
        return 0


def _apply_plan(graph: Any, graph_name: str, plan: ProvenancePlan) -> tuple[int, int]:
    succeeded = 0
    conflicts = 0
    for update in plan.episode_updates:
        anchor = update.anchor
        result = graph.query(
            """
            MATCH (candidate:Episodic {uuid: $uuid, group_id: $group_id})
            WITH count(candidate) AS cardinality
            WHERE cardinality = 1
            MATCH (episode:Episodic {uuid: $uuid, group_id: $group_id})
            WHERE (episode.source_id IS NULL
                   OR episode.source_id = ''
                   OR episode.source_id = $source_id)
              AND (episode.source_type IS NULL
                   OR episode.source_type = ''
                   OR episode.source_type = $source_type)
              AND (episode.engagement_id IS NULL
                   OR episode.engagement_id = ''
                   OR episode.engagement_id = $engagement_id)
              AND (episode.episode_type IS NULL
                   OR episode.episode_type = ''
                   OR episode.episode_type = $episode_type)
              AND (episode.anchor_mode IS NULL
                   OR episode.anchor_mode = ''
                   OR episode.anchor_mode = $anchor_mode)
              AND (episode.producer_contract_version IS NULL
                   OR episode.producer_contract_version = ''
                   OR episode.producer_contract_version = $producer_contract_version)
            SET episode.source_id = $source_id,
                episode.source_type = $source_type,
                episode.engagement_id = $engagement_id,
                episode.episode_type = $episode_type,
                episode.anchor_mode = $anchor_mode,
                episode.producer_contract_version = $producer_contract_version
            RETURN count(episode)
            """,
            params={
                "uuid": update.uuid,
                "group_id": graph_name,
                "source_id": anchor.source_id,
                "source_type": anchor.source_type,
                "engagement_id": anchor.engagement_id,
                "episode_type": anchor.episode_type,
                "anchor_mode": anchor.anchor_mode,
                "producer_contract_version": anchor.producer_contract_version,
            },
        )
        if _result_count(result) == 1:
            succeeded += 1
        else:
            conflicts += 1

    for update in plan.edge_updates:
        assignments: list[str] = []
        params: dict[str, Any] = {
            "uuid": update.uuid,
            "group_id": graph_name,
            "source_uuid": update.source_uuid,
            "target_uuid": update.target_uuid,
        }
        if update.set_endpoints:
            assignments.extend(
                (
                    "edge.source_uuid = $source_uuid",
                    "edge.target_uuid = $target_uuid",
                )
            )
        if update.set_episodes:
            assignments.append("edge.episodes = $episodes")
            params["episodes"] = list(update.episodes or ())
        result = graph.query(
            "MATCH ()-[candidate:RELATES_TO {uuid: $uuid}]->() "
            "WHERE candidate.group_id = $group_id "
            "WITH count(candidate) AS cardinality "
            "WHERE cardinality = 1 "
            "MATCH (source:Entity)-[edge:RELATES_TO {uuid: $uuid}]->(target:Entity) "
            "WHERE edge.group_id = $group_id "
            "AND source.uuid = $source_uuid AND target.uuid = $target_uuid "
            f"SET {', '.join(assignments)} RETURN count(edge)",
            params=params,
        )
        if _result_count(result) == 1:
            succeeded += 1
        else:
            conflicts += 1
    return succeeded, conflicts


def run_provenance_audit(
    client_slug: str,
    *,
    apply: bool = False,
    db_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Audit one exact tenant graph; reject apply until its guard is proven."""

    graph_name = graph_name_for_client(client_slug)
    if apply:
        # Activation remains blocked until the singleton-conditional mutation
        # query is proven against a disposable FalkorDB instance. Unit query-shape
        # coverage is intentionally not treated as that compatibility proof.
        raise ApplyBlockedError(APPLY_BLOCKED_CODE)
    if db_factory is None:
        from falkordb import FalkorDB

        db_factory = FalkorDB
    db = db_factory(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        password=settings.falkordb_password or None,
    )
    graph = db.select_graph(graph_name)
    episode_rows = graph.query(
        """
        MATCH (episode:Episodic)
        WHERE episode.group_id = $group_id
        RETURN episode.uuid, episode.name, episode.source_description,
               episode.source_id, episode.source_type, episode.engagement_id,
               episode.episode_type, episode.anchor_mode,
               episode.producer_contract_version
        """,
        params={"group_id": graph_name},
    ).result_set
    edge_rows = graph.query(
        """
        MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
        WHERE edge.group_id = $group_id
        RETURN edge.uuid, edge.source_uuid, edge.target_uuid,
               source.uuid, target.uuid, edge.episodes
        """,
        params={"group_id": graph_name},
    ).result_set
    plan = build_provenance_plan(
        _episode_records(episode_rows),
        _edge_records(edge_rows),
    )
    return plan.summary(apply=False)
