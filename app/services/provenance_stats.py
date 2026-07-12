"""Content-free structural provenance aggregates for admin graph statistics."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Iterable, Literal

from app.models.episode import EpisodeType
from app.provenance_contract import (
    LEGACY_EPISODE_CONTRACT_VERSION,
    LEGACY_STRUCTURED_CONTRACT_VERSION,
    STRUCTURED_PROVENANCE_CONTRACT_VERSION,
    V2_ANCHOR_MODES,
    V2_PRODUCER_CONTRACT_VERSIONS,
)
from app.services.provenance_ops import canonical_uuid


StructuralStatus = Literal["chained", "pre_chain", "malformed"]
UNRESOLVED_DIMENSION = "unresolved"
PROVENANCE_STATS_EPISODE_ROW_LIMIT_CODE = (
    "PROVENANCE_STATS_EPISODE_ROW_LIMIT_EXCEEDED"
)
PROVENANCE_STATS_EDGE_ROW_LIMIT_CODE = "PROVENANCE_STATS_EDGE_ROW_LIMIT_EXCEEDED"
PROVENANCE_STATS_ENGAGEMENT_BUCKET_LIMIT_CODE = (
    "PROVENANCE_STATS_ENGAGEMENT_BUCKET_LIMIT_EXCEEDED"
)
PROVENANCE_STATS_READ_SHAPE_CODE = "PROVENANCE_STATS_READ_SHAPE_INVALID"
_MAX_STATS_ROWS = 100_000
_MAX_ENGAGEMENT_BUCKETS = 256
_MAX_EPISODE_STORAGE_BYTES = 100_000
_MAX_EPISODES_PER_FACT = 64
_MAX_TEMPORAL_STORAGE_CHARS = 128
_DISALLOWED_CONTROL_PATTERN = (
    r"[\s\S]*[\x00-\x08\x0B\x0C\x0E-\x1F\x7F][\s\S]*"
)
_NONBLANK_TEXT_PATTERN = r"[\s\S]*\S[\s\S]*"
_OVERSIZED_TEMPORAL_SENTINEL = "invalid:oversized-temporal"
_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}")
_EPISODE_TYPES = frozenset(item.value for item in EpisodeType)


class ProvenanceStatsReadError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class StatsEpisode:
    uuid: Any
    has_name: Any
    has_source_description: Any
    source_type: Any
    source_id: Any
    engagement_id: Any
    episode_type: Any
    anchor_mode: Any
    producer_contract_version: Any
    valid_at: Any = None
    provenance_write_state: Any = None


@dataclass(frozen=True)
class StatsEdge:
    uuid: Any
    episodes: Any
    subject_uuid: Any = None
    subject_is_entity: Any = False
    has_subject_name: Any = False
    has_predicate: Any = False
    object_uuid: Any = None
    object_is_entity: Any = False
    has_object_name: Any = False
    has_fact: Any = False
    valid_at: Any = None
    invalid_at: Any = None
    expired_at: Any = None


def _text(value: Any) -> str | None:
    """Return text safe to echo as a bounded aggregate dimension."""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > 240:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        return None
    return normalized


def _structural_text(value: Any, maximum: int) -> str | None:
    """Mirror search completeness without asserting dimension-render safety."""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > maximum
        or any(
            (ord(character) < 32 and character not in "\t\n\r")
            or ord(character) == 127
            for character in normalized
        )
    ):
        return None
    return normalized


def _present_flag(value: Any) -> bool:
    return value is True or (type(value) is int and value == 1)


def _episode_uuid_list(value: Any) -> tuple[tuple[str, ...], bool]:
    """Parse edge episode storage without requiring observed episode rows."""

    candidate = value
    if candidate is None:
        return (), True
    if isinstance(candidate, str):
        if len(candidate) > _MAX_EPISODE_STORAGE_BYTES:
            return (), False
        try:
            if len(candidate.encode("utf-8")) > _MAX_EPISODE_STORAGE_BYTES:
                return (), False
        except UnicodeError:
            return (), False
        try:
            candidate = ast.literal_eval(candidate)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return (), False
    if not isinstance(candidate, (list, tuple)):
        return (), False
    if len(candidate) > _MAX_EPISODES_PER_FACT:
        return (), False

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidate:
        episode_id = canonical_uuid(item)
        if episode_id is None or episode_id in seen:
            return (), False
        normalized.append(episode_id)
        seen.add(episode_id)
    return tuple(normalized), True


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _temporal_is_valid(value: Any) -> bool:
    if isinstance(value, str) and len(value) > _MAX_TEMPORAL_STORAGE_CHARS:
        return False
    return value in (None, "") or _parse_dt(value) is not None


def _edge_wire_complete(edge: StatsEdge) -> bool:
    return bool(
        canonical_uuid(edge.subject_uuid)
        and canonical_uuid(edge.object_uuid)
        and _present_flag(edge.subject_is_entity)
        and _present_flag(edge.object_is_entity)
        and _present_flag(edge.has_subject_name)
        and _present_flag(edge.has_predicate)
        and _present_flag(edge.has_object_name)
        and _present_flag(edge.has_fact)
        and _temporal_is_valid(edge.valid_at)
        and _temporal_is_valid(edge.invalid_at)
        and _temporal_is_valid(edge.expired_at)
    )


def _kind(value: Any) -> str | None:
    text = _text(value)
    return text if text is not None and _KIND_PATTERN.fullmatch(text) else None


def _identifier(value: Any) -> str | None:
    text = _text(value)
    return (
        text
        if text is not None and _IDENTIFIER_PATTERN.fullmatch(text)
        else None
    )


def _episode_type(value: Any) -> str | None:
    text = _kind(value)
    return text if text in _EPISODE_TYPES else None


def _source_complete(source: StatsEpisode) -> bool:
    source_type = _structural_text(source.source_type, 64)
    source_id = _structural_text(source.source_id, 240)
    engagement_id = _structural_text(source.engagement_id, 240)
    episode_type = _structural_text(source.episode_type, 64)
    anchor_mode = _structural_text(source.anchor_mode, 64)
    producer = _structural_text(source.producer_contract_version, 64)
    write_state = _structural_text(source.provenance_write_state, 32)
    if not all(
        (
            _present_flag(source.has_name),
            _present_flag(source.has_source_description),
            source_type,
            source_id,
            engagement_id,
            episode_type,
            anchor_mode,
            producer,
            _temporal_is_valid(source.valid_at),
        )
    ):
        return False
    if producer == LEGACY_STRUCTURED_CONTRACT_VERSION:
        return False
    if anchor_mode == "engagement" and source_id != engagement_id:
        return False
    if source_type == "engagement" and (
        anchor_mode != "engagement" or source_id != engagement_id
    ):
        return False
    if producer == LEGACY_EPISODE_CONTRACT_VERSION:
        return anchor_mode == LEGACY_EPISODE_CONTRACT_VERSION
    if (
        producer == STRUCTURED_PROVENANCE_CONTRACT_VERSION
        and write_state != "complete"
    ):
        return False
    return producer in V2_PRODUCER_CONTRACT_VERSIONS and anchor_mode in V2_ANCHOR_MODES


def _source_claims_malformed(source: StatsEpisode, *, duplicate: bool) -> bool:
    if duplicate or not _temporal_is_valid(source.valid_at):
        return True
    producer = _structural_text(source.producer_contract_version, 64)
    anchor_mode = _structural_text(source.anchor_mode, 64)
    if producer == LEGACY_STRUCTURED_CONTRACT_VERSION:
        return False
    if (
        producer == STRUCTURED_PROVENANCE_CONTRACT_VERSION
        and _structural_text(source.provenance_write_state, 32) == "staging"
    ):
        return False
    return bool(
        (producer is not None and producer != LEGACY_EPISODE_CONTRACT_VERSION)
        or anchor_mode in V2_ANCHOR_MODES
    )


def _duplicate_episode_sentinel(episode_id: str) -> StatsEpisode:
    """Represent any duplicate UUID group once without echoable dimensions."""

    return StatsEpisode(
        uuid=episode_id,
        has_name=False,
        has_source_description=False,
        source_type=None,
        source_id=None,
        engagement_id=None,
        episode_type=None,
        anchor_mode=None,
        producer_contract_version=None,
        valid_at=None,
        provenance_write_state=None,
    )


def _dimensions(sources: Iterable[StatsEpisode]) -> tuple[set[str], set[str]]:
    episode_types = {
        value
        for source in sources
        if (value := _episode_type(source.episode_type)) is not None
    }
    engagements = {
        value
        for source in sources
        if (value := _identifier(source.engagement_id)) is not None
    }
    return episode_types or {UNRESOLVED_DIMENSION}, engagements or {
        UNRESOLVED_DIMENSION
    }


def build_provenance_aggregates(
    episodes: Iterable[StatsEpisode],
    edges: Iterable[StatsEdge],
) -> dict[str, Any]:
    """Aggregate stable facts without returning names, descriptions, or content."""

    episode_records = tuple(episodes)
    edge_records = tuple(edges)
    episode_groups: dict[str, list[StatsEpisode]] = {}
    malformed_response_events = 0
    for episode in episode_records:
        episode_id = canonical_uuid(episode.uuid)
        if episode_id is None:
            malformed_response_events += 1
            continue
        episode_groups.setdefault(episode_id, []).append(episode)

    edge_groups: dict[str, list[StatsEdge]] = {}
    for edge in edge_records:
        edge_id = canonical_uuid(edge.uuid)
        if edge_id is None:
            malformed_response_events += 1
            continue
        edge_groups.setdefault(edge_id, []).append(edge)

    status_counts: Counter[str] = Counter()
    type_counts: Counter[tuple[str, str]] = Counter()
    engagement_counts: Counter[tuple[str, str]] = Counter()
    for edge_id in sorted(edge_groups):
        rows = edge_groups[edge_id]
        sources: list[StatsEpisode] = []
        status: StructuralStatus
        if len(rows) != 1:
            status = "malformed"
        else:
            edge = rows[0]
            normalized, episodes_valid = _episode_uuid_list(edge.episodes)
            if not episodes_valid or not _edge_wire_complete(edge):
                status = "malformed"
            else:
                source_entries: list[tuple[StatsEpisode, bool]] = []
                for episode_id in normalized:
                    episode_rows = episode_groups.get(episode_id, [])
                    if len(episode_rows) == 1:
                        source_entries.append((episode_rows[0], False))
                    elif len(episode_rows) > 1:
                        source_entries.append(
                            (_duplicate_episode_sentinel(episode_id), True)
                        )
                complete_sources = [
                    source
                    for source, duplicate in source_entries
                    if not duplicate and _source_complete(source)
                ]
                if complete_sources:
                    status = "chained"
                    sources = complete_sources
                elif any(
                    _source_claims_malformed(
                        source,
                        duplicate=duplicate,
                    )
                    for source, duplicate in source_entries
                ):
                    status = "malformed"
                    sources = [source for source, _duplicate in source_entries]
                else:
                    status = "pre_chain"
                    sources = [source for source, _duplicate in source_entries]

        status_counts[status] += 1
        episode_types, engagements = _dimensions(sources)
        for episode_type in episode_types:
            type_counts[(status, episode_type)] += 1
        for engagement_id in engagements:
            engagement_counts[(status, engagement_id)] += 1

    if len(engagement_counts) > _MAX_ENGAGEMENT_BUCKETS:
        raise ProvenanceStatsReadError(
            PROVENANCE_STATS_ENGAGEMENT_BUCKET_LIMIT_CODE
        )

    return {
        "facts_total": sum(status_counts.values()),
        "malformed_response_events": min(
            malformed_response_events,
            len(edge_records) + len(episode_records),
        ),
        "by_structural_status": [
            {"structural_status": status, "count": status_counts.get(status, 0)}
            for status in ("chained", "pre_chain", "malformed")
        ],
        "by_episode_type": [
            {
                "structural_status": status,
                "episode_type": episode_type,
                "count": count,
            }
            for (status, episode_type), count in sorted(type_counts.items())
        ],
        "by_engagement": [
            {
                "structural_status": status,
                "engagement_id": engagement_id,
                "count": count,
            }
            for (status, engagement_id), count in sorted(engagement_counts.items())
        ],
    }


def _episode_records(rows: Iterable[Any]) -> tuple[StatsEpisode, ...]:
    records: list[StatsEpisode] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 11:
            records.append(
                StatsEpisode(
                    None,
                    False,
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
        else:
            records.append(StatsEpisode(*row[:11]))
    return tuple(records)


def _edge_records(rows: Iterable[Any]) -> tuple[StatsEdge, ...]:
    records: list[StatsEdge] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 13:
            records.append(StatsEdge(None, None))
        else:
            records.append(StatsEdge(*row[:13]))
    return tuple(records)


def provenance_stats_for_graph(graph: Any, graph_name: str) -> dict[str, Any]:
    episode_result = graph.ro_query(
        """
        MATCH (episode:Episodic)
        WHERE episode.group_id = $group_id
        RETURN episode.uuid,
               (episode.name IS NOT NULL
                 AND episode.name =~ $nonblank_text_pattern
                 AND size(trim(episode.name)) <= 2000
                 AND NOT (trim(episode.name)
                   =~ $disallowed_control_pattern))
                 AS has_name,
               (episode.source_description IS NOT NULL
                 AND episode.source_description =~ $nonblank_text_pattern
                 AND size(trim(episode.source_description)) <= 2000
                 AND NOT (trim(episode.source_description)
                   =~ $disallowed_control_pattern))
                 AS has_source_description,
               episode.source_type, episode.source_id, episode.engagement_id,
               episode.episode_type, episode.anchor_mode,
               episode.producer_contract_version,
               CASE
                 WHEN episode.valid_at IS NULL
                   OR toString(episode.valid_at) = ''
                   THEN NULL
                 WHEN size(toString(episode.valid_at))
                   <= $temporal_storage_limit
                   THEN toString(episode.valid_at)
                 ELSE $oversized_temporal_sentinel
               END AS valid_at,
               episode.provenance_write_state
        LIMIT 100001
        """,
        params={
            "group_id": graph_name,
            "disallowed_control_pattern": _DISALLOWED_CONTROL_PATTERN,
            "nonblank_text_pattern": _NONBLANK_TEXT_PATTERN,
            "temporal_storage_limit": _MAX_TEMPORAL_STORAGE_CHARS,
            "oversized_temporal_sentinel": _OVERSIZED_TEMPORAL_SENTINEL,
        },
    )
    episode_rows = getattr(episode_result, "result_set", None)
    if not isinstance(episode_rows, list):
        raise ProvenanceStatsReadError(PROVENANCE_STATS_READ_SHAPE_CODE)
    if len(episode_rows) > _MAX_STATS_ROWS:
        raise ProvenanceStatsReadError(PROVENANCE_STATS_EPISODE_ROW_LIMIT_CODE)

    edge_result = graph.ro_query(
        """
        MATCH (subject)-[edge:RELATES_TO]->(object)
        WHERE edge.group_id = $group_id
        RETURN edge.uuid, edge.episodes, subject.uuid,
               'Entity' IN labels(subject) AS subject_is_entity,
               (subject.name IS NOT NULL
                 AND subject.name =~ $nonblank_text_pattern
                 AND size(trim(subject.name)) <= 2000
                 AND NOT (trim(subject.name)
                   =~ $disallowed_control_pattern))
                 AS has_subject_name,
               (edge.name IS NOT NULL
                 AND edge.name =~ $nonblank_text_pattern
                 AND size(trim(edge.name)) <= 160
                 AND NOT (trim(edge.name)
                   =~ $disallowed_control_pattern))
                 AS has_predicate,
               object.uuid,
               'Entity' IN labels(object) AS object_is_entity,
               (object.name IS NOT NULL
                 AND object.name =~ $nonblank_text_pattern
                 AND size(trim(object.name)) <= 2000
                 AND NOT (trim(object.name)
                   =~ $disallowed_control_pattern))
                 AS has_object_name,
               (edge.fact IS NOT NULL
                 AND edge.fact =~ $nonblank_text_pattern
                 AND size(trim(edge.fact)) <= 16000
                 AND NOT (trim(edge.fact)
                   =~ $disallowed_control_pattern))
                 AS has_fact,
               CASE
                 WHEN edge.valid_at IS NULL OR toString(edge.valid_at) = ''
                   THEN NULL
                 WHEN size(toString(edge.valid_at)) <= $temporal_storage_limit
                   THEN toString(edge.valid_at)
                 ELSE $oversized_temporal_sentinel
               END AS valid_at,
               CASE
                 WHEN edge.invalid_at IS NULL OR toString(edge.invalid_at) = ''
                   THEN NULL
                 WHEN size(toString(edge.invalid_at)) <= $temporal_storage_limit
                   THEN toString(edge.invalid_at)
                 ELSE $oversized_temporal_sentinel
               END AS invalid_at,
               CASE
                 WHEN edge.expired_at IS NULL OR toString(edge.expired_at) = ''
                   THEN NULL
                 WHEN size(toString(edge.expired_at)) <= $temporal_storage_limit
                   THEN toString(edge.expired_at)
                 ELSE $oversized_temporal_sentinel
               END AS expired_at
        LIMIT 100001
        """,
        params={
            "group_id": graph_name,
            "disallowed_control_pattern": _DISALLOWED_CONTROL_PATTERN,
            "nonblank_text_pattern": _NONBLANK_TEXT_PATTERN,
            "temporal_storage_limit": _MAX_TEMPORAL_STORAGE_CHARS,
            "oversized_temporal_sentinel": _OVERSIZED_TEMPORAL_SENTINEL,
        },
    )
    edge_rows = getattr(edge_result, "result_set", None)
    if not isinstance(edge_rows, list):
        raise ProvenanceStatsReadError(PROVENANCE_STATS_READ_SHAPE_CODE)
    if len(edge_rows) > _MAX_STATS_ROWS:
        raise ProvenanceStatsReadError(PROVENANCE_STATS_EDGE_ROW_LIMIT_CODE)
    episodes = _episode_records(episode_rows)
    edges = _edge_records(edge_rows)
    return build_provenance_aggregates(episodes, edges)
