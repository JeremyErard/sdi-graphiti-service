"""Structured ingestion with a legacy pre-chain path and anchored v2 path.

The legacy endpoint remains wire-compatible and intentionally creates pre-chain
episodes. New callers use ``/structured/v2`` and must provide every structural
source anchor Graphiti can truthfully preserve. Graphiti records those anchors;
it never claims that the referenced row exists in a tenant database.
"""

import logging
import re
import uuid as uuidlib
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings
from app.models.episode import EpisodeType
from app.provenance_contract import (
    LEGACY_STRUCTURED_CONTRACT_VERSION,
    STRUCTURED_PROVENANCE_CONTRACT_VERSION,
)
from app.services import graphiti_client

logger = logging.getLogger("graphiti_service")

router = APIRouter()

_SAFE_KIND = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class AnchorMode(str, Enum):
    """Caller-declared source granularity; Graphiti never infers this value."""

    TYPED_SOURCE = "typed_source"
    ENGAGEMENT = "engagement"


def _non_controlled(value: str, label: str, *, maximum: int = 240) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in normalized
    ):
        raise ValueError(f"{label} must be a bounded non-control string")
    return normalized


class StructuredEntity(BaseModel):
    name: str
    type: str
    description: str = ""


class StructuredRelationship(BaseModel):
    source: str
    target: str
    relation: str
    fact: str = ""


class StructuredIngestRequest(BaseModel):
    """Legacy compatibility request. Its resulting facts remain pre-chain."""

    client_slug: str
    episode_name: str
    source_description: str = ""
    reference_time: datetime | None = None
    entities: list[StructuredEntity] = Field(default_factory=list)
    relationships: list[StructuredRelationship] = Field(default_factory=list)


class StructuredEntityV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _non_controlled(value, "entity name")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = _non_controlled(value, "entity type", maximum=64).lower()
        if not _SAFE_KIND.fullmatch(normalized):
            raise ValueError("entity type must be a lower-case kind token")
        return normalized

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if len(value) > 8_000 or any(ord(character) == 0 for character in value):
            raise ValueError("entity description is invalid")
        return value.strip()


class StructuredRelationshipV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: uuidlib.UUID
    source: str
    target: str
    relation: str
    fact: str

    @field_validator("source", "target")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _non_controlled(value, "relationship endpoint")

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, value: str) -> str:
        return _non_controlled(value, "relationship relation", maximum=160)

    @field_validator("fact")
    @classmethod
    def validate_fact(cls, value: str) -> str:
        return _non_controlled(value, "relationship fact", maximum=16_000)


class StructuredIngestV2Request(BaseModel):
    """Strict P1 source-anchored ingestion contract.

    Source existence and content grounding are deliberately absent: only the
    tenant-authoritative backend may make those determinations.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[STRUCTURED_PROVENANCE_CONTRACT_VERSION]
    client_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,127}$")
    engagement_id: str
    episode_uuid: uuidlib.UUID
    episode_name: str
    episode_type: EpisodeType
    source_id: str
    source_type: str
    source_description: str
    anchor_mode: AnchorMode
    producer_contract_version: Literal[STRUCTURED_PROVENANCE_CONTRACT_VERSION]
    reference_time: datetime | None = None
    entities: list[StructuredEntityV2] = Field(min_length=1, max_length=5_000)
    relationships: list[StructuredRelationshipV2] = Field(
        min_length=1, max_length=10_000
    )

    @field_validator("engagement_id", "episode_name", "source_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return _non_controlled(value, "provenance identifier")

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        normalized = _non_controlled(value, "source type", maximum=64).lower()
        if not _SAFE_KIND.fullmatch(normalized):
            raise ValueError("source_type must be a lower-case kind token")
        return normalized

    @field_validator("source_description")
    @classmethod
    def validate_source_description(cls, value: str) -> str:
        return _non_controlled(value, "source description", maximum=2_000)

    @model_validator(mode="after")
    def validate_stable_fact_set(self):
        if (
            self.anchor_mode == AnchorMode.ENGAGEMENT
            and self.source_id != self.engagement_id
        ):
            raise ValueError(
                "engagement anchors require source_id to equal engagement_id"
            )
        if self.source_type == "engagement" and (
            self.anchor_mode != AnchorMode.ENGAGEMENT
            or self.source_id != self.engagement_id
        ):
            raise ValueError(
                "engagement sources require an exact engagement anchor"
            )
        entity_names = [_normalize(entity.name) for entity in self.entities]
        if len(entity_names) != len(set(entity_names)):
            raise ValueError("v2 entity names must be unique after normalization")
        declared = set(entity_names)
        fact_ids = [str(relationship.fact_id) for relationship in self.relationships]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("v2 fact_id values must be unique")
        for relationship in self.relationships:
            if _normalize(relationship.source) not in declared or _normalize(
                relationship.target
            ) not in declared:
                raise ValueError(
                    "v2 relationship endpoints must be declared in entities"
                )
        return self


class StructuredIngestResponse(BaseModel):
    graph_name: str
    episode_uuid: str
    entities_created: int
    entities_merged: int
    relationships_created: int
    relationships_skipped: int
    elapsed_ms: int


class StructuredIngestV2Response(StructuredIngestResponse):
    contract_version: Literal[STRUCTURED_PROVENANCE_CONTRACT_VERSION] = (
        STRUCTURED_PROVENANCE_CONTRACT_VERSION
    )
    chain_status: Literal["chained"] = "chained"
    fact_ids: list[str]


def _normalize(name: str) -> str:
    return name.strip().lower()


async def _write_structured(
    req: StructuredIngestRequest | StructuredIngestV2Request,
    *,
    anchored: bool,
) -> StructuredIngestResponse | StructuredIngestV2Response:
    start = datetime.now(timezone.utc)
    reference_time = req.reference_time or start
    contract_version = (
        STRUCTURED_PROVENANCE_CONTRACT_VERSION
        if anchored
        else LEGACY_STRUCTURED_CONTRACT_VERSION
    )
    episode_uuid = (
        str(req.episode_uuid)
        if anchored and isinstance(req, StructuredIngestV2Request)
        else str(uuidlib.uuid4())
    )

    try:
        graph_name = graphiti_client._graph_name_for_client(req.client_slug)
        from falkordb import FalkorDB

        db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
        )
        graph = db.select_graph(graph_name)

        now_iso = start.isoformat()
        if anchored and isinstance(req, StructuredIngestV2Request):
            episode_collision = graph.query(
                """
                MATCH (ep:Episodic {uuid: $episode_uuid, group_id: $group_id})
                RETURN ep.uuid LIMIT 1
                """,
                params={"episode_uuid": episode_uuid, "group_id": graph_name},
            )
            fact_collision = graph.query(
                """
                MATCH ()-[edge:RELATES_TO]->()
                WHERE edge.uuid IN $fact_ids AND edge.group_id = $group_id
                RETURN edge.uuid LIMIT 1
                """,
                params={
                    "fact_ids": [str(rel.fact_id) for rel in req.relationships],
                    "group_id": graph_name,
                },
            )
            if episode_collision.result_set or fact_collision.result_set:
                raise HTTPException(
                    status_code=409,
                    detail="Structured ingest identity conflict",
                )
            graph.query(
                """
                CREATE (ep:Episodic {
                    uuid: $uuid,
                    name: $name,
                    content: $content,
                    source: 'text',
                    source_description: $source_description,
                    source_id: $source_id,
                    source_type: $source_type,
                    engagement_id: $engagement_id,
                    episode_type: $episode_type,
                    anchor_mode: $anchor_mode,
                    producer_contract_version: $producer_contract_version,
                    valid_at: $valid_at,
                    created_at: $created_at,
                    group_id: $group_id
                })
                """,
                params={
                    "uuid": episode_uuid,
                    "name": req.episode_name,
                    "content": "",
                    "source_description": req.source_description,
                    "source_id": req.source_id,
                    "source_type": req.source_type,
                    "engagement_id": req.engagement_id,
                    "episode_type": req.episode_type.value,
                    "anchor_mode": req.anchor_mode.value,
                    "producer_contract_version": (
                        STRUCTURED_PROVENANCE_CONTRACT_VERSION
                    ),
                    "valid_at": reference_time.isoformat(),
                    "created_at": now_iso,
                    "group_id": graph_name,
                },
            )
        else:
            graph.query(
                """
                CREATE (ep:Episodic {
                    uuid: $uuid,
                    name: $name,
                    content: $content,
                    source: 'text',
                    source_description: $source_description,
                    anchor_mode: 'pre_chain',
                    producer_contract_version: $producer_contract_version,
                    valid_at: $valid_at,
                    created_at: $created_at,
                    group_id: $group_id
                })
                """,
                params={
                    "uuid": episode_uuid,
                    "name": req.episode_name,
                    "content": "",
                    "source_description": req.source_description,
                    "producer_contract_version": contract_version,
                    "valid_at": reference_time.isoformat(),
                    "created_at": now_iso,
                    "group_id": graph_name,
                },
            )

        name_to_uuid: dict[str, str] = {}
        entities_created = 0
        entities_merged = 0

        for ent in req.entities:
            name_norm = _normalize(ent.name)
            if not name_norm:
                continue

            match = graph.query(
                """
                MATCH (e:Entity {group_id: $group_id})
                WHERE toLower(e.name) = $name_lower
                RETURN e.uuid AS uuid LIMIT 1
                """,
                params={"group_id": graph_name, "name_lower": name_norm},
            )
            existing_uuid: str | None = None
            if match.result_set:
                existing_uuid = str(match.result_set[0][0])

            if existing_uuid:
                name_to_uuid[name_norm] = existing_uuid
                entities_merged += 1
            else:
                new_uuid = str(uuidlib.uuid4())
                summary = (ent.description or "").strip()
                label_value = ent.type.strip() if ent.type else "Entity"
                graph.query(
                    """
                    CREATE (e:Entity {
                        uuid: $uuid,
                        name: $name,
                        summary: $summary,
                        group_id: $group_id,
                        created_at: $created_at,
                        labels: $labels
                    })
                    """,
                    params={
                        "uuid": new_uuid,
                        "name": ent.name.strip(),
                        "summary": summary,
                        "group_id": graph_name,
                        "created_at": now_iso,
                        "labels": [label_value],
                    },
                )
                name_to_uuid[name_norm] = new_uuid
                entities_created += 1

            graph.query(
                """
                MATCH (ep:Episodic {uuid: $ep_uuid}), (e:Entity {uuid: $e_uuid})
                CREATE (ep)-[:MENTIONS {created_at: $created_at}]->(e)
                """,
                params={
                    "ep_uuid": episode_uuid,
                    "e_uuid": name_to_uuid[name_norm],
                    "created_at": now_iso,
                },
            )

        rels_created = 0
        rels_skipped = 0
        written_fact_ids: list[str] = []
        for rel in req.relationships:
            src_norm = _normalize(rel.source)
            tgt_norm = _normalize(rel.target)
            src_uuid = name_to_uuid.get(src_norm)
            tgt_uuid = name_to_uuid.get(tgt_norm)

            if not src_uuid:
                match = graph.query(
                    "MATCH (e:Entity {group_id: $g}) WHERE toLower(e.name) = $n RETURN e.uuid LIMIT 1",
                    params={"g": graph_name, "n": src_norm},
                )
                if match.result_set:
                    src_uuid = str(match.result_set[0][0])
            if not tgt_uuid:
                match = graph.query(
                    "MATCH (e:Entity {group_id: $g}) WHERE toLower(e.name) = $n RETURN e.uuid LIMIT 1",
                    params={"g": graph_name, "n": tgt_norm},
                )
                if match.result_set:
                    tgt_uuid = str(match.result_set[0][0])

            if not (src_uuid and tgt_uuid):
                rels_skipped += 1
                continue

            edge_uuid = (
                str(rel.fact_id)
                if anchored and isinstance(rel, StructuredRelationshipV2)
                else str(uuidlib.uuid4())
            )
            fact = rel.fact or rel.relation or ""
            if anchored and isinstance(req, StructuredIngestV2Request):
                graph.query(
                    """
                    MATCH (s:Entity {uuid: $src}), (t:Entity {uuid: $tgt})
                    CREATE (s)-[r:RELATES_TO {
                        uuid: $edge_uuid,
                        name: $name,
                        fact: $fact,
                        source_uuid: $src,
                        target_uuid: $tgt,
                        episodes: $episodes,
                        engagement_id: $engagement_id,
                        source_id: $source_id,
                        source_type: $source_type,
                        episode_type: $episode_type,
                        anchor_mode: $anchor_mode,
                        producer_contract_version: $producer_contract_version,
                        created_at: $created_at,
                        group_id: $group_id
                    }]->(t)
                    """,
                    params={
                        "src": src_uuid,
                        "tgt": tgt_uuid,
                        "edge_uuid": edge_uuid,
                        "name": rel.relation.strip(),
                        "fact": fact,
                        "episodes": [episode_uuid],
                        "engagement_id": req.engagement_id,
                        "source_id": req.source_id,
                        "source_type": req.source_type,
                        "episode_type": req.episode_type.value,
                        "anchor_mode": req.anchor_mode.value,
                        "producer_contract_version": (
                            STRUCTURED_PROVENANCE_CONTRACT_VERSION
                        ),
                        "created_at": now_iso,
                        "group_id": graph_name,
                    },
                )
            else:
                graph.query(
                    """
                    MATCH (s:Entity {uuid: $src}), (t:Entity {uuid: $tgt})
                    CREATE (s)-[r:RELATES_TO {
                        uuid: $edge_uuid,
                        name: $name,
                        fact: $fact,
                        source_uuid: $src,
                        target_uuid: $tgt,
                        episodes: $episodes,
                        anchor_mode: 'pre_chain',
                        producer_contract_version: $producer_contract_version,
                        created_at: $created_at,
                        group_id: $group_id
                    }]->(t)
                    """,
                    params={
                        "src": src_uuid,
                        "tgt": tgt_uuid,
                        "edge_uuid": edge_uuid,
                        "name": rel.relation.strip() if rel.relation else "relates_to",
                        "fact": fact,
                        "episodes": [episode_uuid],
                        "producer_contract_version": contract_version,
                        "created_at": now_iso,
                        "group_id": graph_name,
                    },
                )
            written_fact_ids.append(edge_uuid)
            rels_created += 1

        elapsed_ms = int(
            (datetime.now(timezone.utc) - start).total_seconds() * 1000
        )
        logger.info(
            "[graphiti] structured ingest to %s contract=%s: %s new + %s "
            "merged entities, %s rels (%s skipped) in %sms",
            graph_name,
            contract_version,
            entities_created,
            entities_merged,
            rels_created,
            rels_skipped,
            elapsed_ms,
        )

        common = dict(
            graph_name=graph_name,
            episode_uuid=episode_uuid,
            entities_created=entities_created,
            entities_merged=entities_merged,
            relationships_created=rels_created,
            relationships_skipped=rels_skipped,
            elapsed_ms=elapsed_ms,
        )
        if anchored:
            return StructuredIngestV2Response(
                **common,
                fact_ids=written_fact_ids,
            )
        return StructuredIngestResponse(**common)

    except HTTPException:
        raise
    except Exception as error:
        logger.error(
            "[graphiti] structured ingest failed contract=%s error_type=%s",
            contract_version,
            type(error).__name__,
        )
        raise HTTPException(status_code=500, detail="Structured ingest failed")


@router.post("/structured", response_model=StructuredIngestResponse)
async def ingest_structured(req: StructuredIngestRequest):
    """Preserved legacy endpoint. Every resulting fact is pre-chain."""

    return await _write_structured(req, anchored=False)


@router.post("/structured/v2", response_model=StructuredIngestV2Response)
async def ingest_structured_v2(req: StructuredIngestV2Request):
    """Write strict anchors without asserting relational source authority.

    This compatibility surface name-merges graph-local entities. It is barred
    from governed bulk projection, which belongs to ``/ingest/projection/v2``.
    """

    return await _write_structured(req, anchored=True)
