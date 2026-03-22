"""Graph traversal endpoints for Knowledge Map visualization.

Uses lazy initialization with strict socket timeouts to prevent
startup hangs. Connection to FalkorDB only happens when the
endpoint is actually called, not at import/startup time.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("graphiti_service")

router = APIRouter()


def _graph_name_for_client(client_slug: str) -> str:
    safe_slug = "".join(c for c in client_slug if c.isalnum() or c == "_").lower()
    return f"client_{safe_slug}"


def _get_redis_connection():
    """Lazy connection with strict 5-second socket timeout.

    Never called at import time — only when the endpoint is hit.
    Fails fast and loudly if FalkorDB is unreachable.
    """
    from redis import Redis

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    password = os.getenv("FALKORDB_PASSWORD", None)

    try:
        r = Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=True,
            socket_connect_timeout=5.0,
            socket_timeout=15.0,
        )
        # Test connection immediately
        r.ping()
        logger.info(f"[graph] Connected to FalkorDB at {host}:{port}")
        return r
    except Exception as e:
        logger.error(f"[graph] FalkorDB connection failed ({host}:{port}): {e}")
        raise HTTPException(status_code=503, detail=f"Knowledge graph database unreachable: {str(e)}")


class GraphDataRequest(BaseModel):
    client_slug: str
    max_nodes: int = 500


class GraphNode(BaseModel):
    id: str
    name: str
    type: str
    properties: dict[str, Any] = {}


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    fact: str = ""


class GraphDataResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    graph_name: str
    node_count: int
    edge_count: int


def _infer_type(name: str, summary: str) -> str:
    """Infer a visualization-friendly node type from name and summary."""
    name_lower = name.lower().strip()
    summary_lower = summary.lower()

    # Department detection — strict allowlist, not keyword matching
    ACTUAL_DEPARTMENTS = [
        'internal audit', 'audit department',
        'compliance & enforcement', 'compliance and enforcement', 'compliance and enforcement department', 'c&e',
        'licensing & investigations', 'licensing and investigations', 'licensing and investigation division',
        'licensing department', 'compliance department',
        'gaming technology', 'gaming technology department', 'gtu',
        'executive', 'executive office',
        'pokagon band gaming commission', 'pbgc',
        'casino operations', 'surveillance', 'surveillance department',
        'human resources', 'hr', 'information technology', 'it department',
    ]
    if name_lower in ACTUAL_DEPARTMENTS:
        return 'department'

    # Person detection — job titles and named individuals
    person_indicators = ['director of', 'chief', 'manager', 'specialist', 'investigator', 'inspector',
                         'assistant', 'commissioner', 'auditor', 'supervisor', 'officer']
    if any(name_lower.startswith(p) for p in person_indicators):
        return 'person'
    # Named people (First Last pattern with summary mentioning "holds" or "serves")
    if any(w in summary_lower for w in ['holds this role', 'holds the position', 'serves as', 'reporting to']):
        return 'person'

    # System detection
    sys_keywords = ['permitrak', 'filemaker', 'fmp', 'teammate', 'teamrisk', 'teamschedule',
                    'key traka', 'traka', 'igt', 'table manager', 'premisys', 'sharepoint',
                    'sharefile', 'active directory', 'vmware', 'adp', 'zendesk', 'excel',
                    'casino cash trac', 'casino tables accounting', 'kambi', 'crossmatch',
                    'barracuda', 'infogenesis', 'itraq', 'kiteworks', 'powerkiosk',
                    'geocomply', 'pala interactive']
    if any(s in name_lower for s in sys_keywords):
        return 'system'
    if 'software' in name_lower or 'platform' in name_lower or 'application' in name_lower:
        return 'system'

    # Content-based inference
    if any(w in summary_lower for w in ['pain', 'problem', 'issue', 'challenge', 'friction', 'struggle', 'failure', 'crisis']):
        return 'pain_point'
    if any(w in summary_lower for w in ['opportunity', 'improve', 'potential', 'automat', 'recommend', 'should consider']):
        return 'opportunity'
    if any(w in summary_lower for w in ['process', 'workflow', 'procedure', 'step', 'lifecycle', 'policy', 'protocol']):
        return 'process'
    if any(w in summary_lower for w in ['person', 'employee', 'staff member']):
        return 'person'

    return 'entity'


@router.post("/nodes-and-edges", response_model=GraphDataResponse)
async def get_graph_data(req: GraphDataRequest):
    """Return all nodes and edges from a client's knowledge graph."""
    graph_name = _graph_name_for_client(req.client_slug)
    r = _get_redis_connection()

    try:
        # Query nodes via GRAPH.QUERY
        logger.info(f"[graph] Querying nodes from {graph_name} (limit {req.max_nodes})")
        node_result = r.execute_command(
            "GRAPH.QUERY", graph_name,
            f"MATCH (n) RETURN id(n), labels(n), n.uuid, n.name, n.summary LIMIT {req.max_nodes}"
        )

        nodes: list[GraphNode] = []
        node_ids: set[str] = set()

        # GRAPH.QUERY returns [header, data, stats]
        if len(node_result) >= 2 and node_result[1]:
            for record in node_result[1]:
                internal_id = str(record[0]) if record[0] is not None else ""
                labels = record[1] if isinstance(record[1], list) else []
                uuid = str(record[2]) if record[2] else internal_id
                name = str(record[3]) if record[3] else f"Node {internal_id}"
                summary = str(record[4]) if record[4] else ""

                if not uuid or uuid in node_ids:
                    continue

                # Skip Graphiti internal nodes
                label_str = str(labels).lower() if labels else ""
                if 'episode' in label_str or 'community' in label_str:
                    continue

                node_type = _infer_type(name, summary)
                node_ids.add(uuid)

                nodes.append(GraphNode(
                    id=uuid,
                    name=name,
                    type=node_type,
                    properties={"summary": summary[:200]} if summary else {},
                ))

        # Query edges
        logger.info(f"[graph] Querying edges from {graph_name}")
        edge_result = r.execute_command(
            "GRAPH.QUERY", graph_name,
            f"MATCH (a)-[r]->(b) RETURN a.uuid, type(r), b.uuid, r.fact LIMIT {req.max_nodes * 3}"
        )

        edges: list[GraphEdge] = []
        edge_ids: set[str] = set()

        if len(edge_result) >= 2 and edge_result[1]:
            for record in edge_result[1]:
                src_id = str(record[0]) if record[0] else ""
                rel_type = str(record[1]) if record[1] else "RELATED_TO"
                tgt_id = str(record[2]) if record[2] else ""
                fact = str(record[3]) if len(record) > 3 and record[3] else ""

                if not src_id or not tgt_id:
                    continue
                if src_id not in node_ids or tgt_id not in node_ids:
                    continue

                edge_id = f"{src_id}-{rel_type}-{tgt_id}"
                if edge_id in edge_ids:
                    continue
                edge_ids.add(edge_id)

                edges.append(GraphEdge(
                    id=edge_id,
                    source=src_id,
                    target=tgt_id,
                    label=rel_type,
                    fact=fact,
                ))

        r.close()

        logger.info(f"[graph] Result: {len(nodes)} nodes, {len(edges)} edges from {graph_name}")

        return GraphDataResponse(
            nodes=nodes,
            edges=edges,
            graph_name=graph_name,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[graph] Query error on {graph_name}: {e}")
        r.close()
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(e)}")
