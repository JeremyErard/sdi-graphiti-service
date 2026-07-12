"""Content-free structural provenance aggregates for admin graph statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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
from app.services.provenance_ops import canonical_uuid, normalize_provable_episode_list


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
    provenance_write_state: Any = None


@dataclass(frozen=True)
class StatsEdge:
    uuid: Any
    episodes: Any


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
    if duplicate:
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
    observed_episode_ids = frozenset(episode_groups)

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
            normalized, _already = normalize_provable_episode_list(
                rows[0].episodes,
                known_episode_ids=observed_episode_ids,
            )
            if normalized is None:
                status = "pre_chain" if rows[0].episodes is None else "malformed"
            else:
                source_entries: list[tuple[StatsEpisode, bool]] = []
                for episode_id in normalized:
                    episode_rows = episode_groups.get(episode_id, [])
                    if len(episode_rows) == 1:
                        source_entries.append((episode_rows[0], False))
                    elif len(episode_rows) > 1:
                        source_entries.extend(
                            (source, True) for source in episode_rows
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
        if not isinstance(row, (list, tuple)) or len(row) < 10:
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
                )
            )
        else:
            records.append(StatsEpisode(*row[:10]))
    return tuple(records)


def _edge_records(rows: Iterable[Any]) -> tuple[StatsEdge, ...]:
    records: list[StatsEdge] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            records.append(StatsEdge(None, None))
        else:
            records.append(StatsEdge(*row[:2]))
    return tuple(records)


def provenance_stats_for_graph(graph: Any, graph_name: str) -> dict[str, Any]:
    episode_result = graph.ro_query(
        """
        MATCH (episode:Episodic)
        WHERE episode.group_id = $group_id
        RETURN episode.uuid,
               episode.name IS NOT NULL AND trim(episode.name) <> '' AS has_name,
               episode.source_description IS NOT NULL
                 AND trim(episode.source_description) <> '' AS has_source_description,
               episode.source_type, episode.source_id, episode.engagement_id,
               episode.episode_type, episode.anchor_mode,
               episode.producer_contract_version,
               episode.provenance_write_state
        LIMIT 100001
        """,
        params={"group_id": graph_name},
    )
    episode_rows = getattr(episode_result, "result_set", None)
    if not isinstance(episode_rows, list):
        raise ProvenanceStatsReadError(PROVENANCE_STATS_READ_SHAPE_CODE)
    if len(episode_rows) > _MAX_STATS_ROWS:
        raise ProvenanceStatsReadError(PROVENANCE_STATS_EPISODE_ROW_LIMIT_CODE)

    edge_result = graph.ro_query(
        """
        MATCH ()-[edge:RELATES_TO]->()
        WHERE edge.group_id = $group_id
        RETURN edge.uuid, edge.episodes
        LIMIT 100001
        """,
        params={"group_id": graph_name},
    )
    edge_rows = getattr(edge_result, "result_set", None)
    if not isinstance(edge_rows, list):
        raise ProvenanceStatsReadError(PROVENANCE_STATS_READ_SHAPE_CODE)
    if len(edge_rows) > _MAX_STATS_ROWS:
        raise ProvenanceStatsReadError(PROVENANCE_STATS_EDGE_ROW_LIMIT_CODE)
    episodes = _episode_records(episode_rows)
    edges = _edge_records(edge_rows)
    return build_provenance_aggregates(episodes, edges)
