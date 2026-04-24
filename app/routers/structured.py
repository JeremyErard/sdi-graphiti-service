"""Structured ingest — accepts pre-extracted entities+relationships and writes directly.

Bypasses graphiti_core's per-episode LLM reconciliation loop. Intended for batch
backfill paths where extraction happens client-side with a single LLM call per
document and consolidation happens here as cheap graph writes.

Schema is kept compatible with what `graphiti_core` produces so downstream
readers (/graph/nodes-and-edges, /search/*) continue to work across mixed
ingestion paths.
"""

import logging
import uuid as uuidlib
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services import graphiti_client

logger = logging.getLogger("graphiti_service")

router = APIRouter()


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
    client_slug: str
    episode_name: str
    source_description: str = ""
    reference_time: datetime | None = None
    entities: list[StructuredEntity] = Field(default_factory=list)
    relationships: list[StructuredRelationship] = Field(default_factory=list)


class StructuredIngestResponse(BaseModel):
    graph_name: str
    episode_uuid: str
    entities_created: int
    entities_merged: int
    relationships_created: int
    relationships_skipped: int
    elapsed_ms: int


def _normalize(name: str) -> str:
    return name.strip().lower()


@router.post("/structured", response_model=StructuredIngestResponse)
async def ingest_structured(req: StructuredIngestRequest):
    """Ingest pre-extracted entities + relationships into a client's graph.

    For each entity: MERGE on (group_id, lower(name)) to match existing nodes
    written either by graphiti_core or by prior structured ingests.
    For each relationship: MATCH source + target, CREATE RELATES_TO edge.
    Also creates an Episodic node and MENTIONS edges so provenance is preserved.

    Fast — pure graph writes, no LLM calls. Typically <2s per episode regardless
    of graph size (unlike graphiti_core's per-episode LLM reconciliation chain).
    """
    start = datetime.now(timezone.utc)
    graph_name = graphiti_client._graph_name_for_client(req.client_slug)
    reference_time = req.reference_time or start

    try:
        from falkordb import FalkorDB

        db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
        )
        graph = db.select_graph(graph_name)

        now_iso = start.isoformat()
        episode_uuid = str(uuidlib.uuid4())

        # ── 1. Create the Episodic node ──
        graph.query(
            """
            CREATE (ep:Episodic {
                uuid: $uuid,
                name: $name,
                content: $content,
                source: 'text',
                source_description: $source_description,
                valid_at: $valid_at,
                created_at: $created_at,
                group_id: $group_id
            })
            """,
            params={
                "uuid": episode_uuid,
                "name": req.episode_name,
                "content": "",  # structured path doesn't store raw content
                "source_description": req.source_description,
                "valid_at": reference_time.isoformat(),
                "created_at": now_iso,
                "group_id": graph_name,
            },
        )

        # ── 2. Upsert entities ──
        # Build a name → uuid map so relationships can look up by name after.
        name_to_uuid: dict[str, str] = {}
        entities_created = 0
        entities_merged = 0

        for ent in req.entities:
            name_norm = _normalize(ent.name)
            if not name_norm:
                continue

            # Try to find existing entity by case-insensitive name match
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
                # summary is either the description we were handed, or an empty string
                summary = (ent.description or "").strip()
                # labels: store the domain-type string in the 'labels' property,
                # matching what graphiti_core writes (which downstream graph.py
                # inspects for type mapping).
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
                        "labels": f"['{label_value}']",
                    },
                )
                name_to_uuid[name_norm] = new_uuid
                entities_created += 1

            # MENTIONS edge from episode to entity
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

        # ── 3. Create relationships ──
        rels_created = 0
        rels_skipped = 0
        for rel in req.relationships:
            src_norm = _normalize(rel.source)
            tgt_norm = _normalize(rel.target)
            src_uuid = name_to_uuid.get(src_norm)
            tgt_uuid = name_to_uuid.get(tgt_norm)

            # If source or target isn't in this episode's entity list, try
            # to locate them in the existing graph before giving up.
            if not src_uuid:
                m = graph.query(
                    "MATCH (e:Entity {group_id: $g}) WHERE toLower(e.name) = $n RETURN e.uuid LIMIT 1",
                    params={"g": graph_name, "n": src_norm},
                )
                if m.result_set:
                    src_uuid = str(m.result_set[0][0])
            if not tgt_uuid:
                m = graph.query(
                    "MATCH (e:Entity {group_id: $g}) WHERE toLower(e.name) = $n RETURN e.uuid LIMIT 1",
                    params={"g": graph_name, "n": tgt_norm},
                )
                if m.result_set:
                    tgt_uuid = str(m.result_set[0][0])

            if not (src_uuid and tgt_uuid):
                rels_skipped += 1
                continue

            edge_uuid = str(uuidlib.uuid4())
            fact = rel.fact or rel.relation or ""
            graph.query(
                """
                MATCH (s:Entity {uuid: $src}), (t:Entity {uuid: $tgt})
                CREATE (s)-[r:RELATES_TO {
                    uuid: $edge_uuid,
                    name: $name,
                    fact: $fact,
                    episodes: $episodes,
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
                    "episodes": f"['{episode_uuid}']",
                    "created_at": now_iso,
                    "group_id": graph_name,
                },
            )
            rels_created += 1

        elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

        logger.info(
            f"[graphiti] structured ingest to {graph_name}: "
            f"{entities_created} new + {entities_merged} merged entities, "
            f"{rels_created} rels ({rels_skipped} skipped) in {elapsed_ms}ms"
        )

        return StructuredIngestResponse(
            graph_name=graph_name,
            episode_uuid=episode_uuid,
            entities_created=entities_created,
            entities_merged=entities_merged,
            relationships_created=rels_created,
            relationships_skipped=rels_skipped,
            elapsed_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[graphiti] structured ingest failed for {req.client_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Structured ingest failed: {e}")
