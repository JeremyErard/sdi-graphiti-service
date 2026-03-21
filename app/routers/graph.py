"""Graph traversal endpoints for Knowledge Map visualization."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from redis import Redis

from app.config import settings

logger = logging.getLogger("graphiti_service")

router = APIRouter()


def _graph_name_for_client(client_slug: str) -> str:
    safe_slug = "".join(c for c in client_slug if c.isalnum() or c == "_").lower()
    return f"client_{safe_slug}"


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
    properties: dict[str, Any] = {}


class GraphDataResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    graph_name: str
    node_count: int
    edge_count: int


@router.post("/nodes-and-edges", response_model=GraphDataResponse)
async def get_graph_data(req: GraphDataRequest):
    """Return all nodes and edges from a client's knowledge graph for visualization.

    Uses the Redis protocol to query FalkorDB directly via GRAPH.QUERY command.
    """
    graph_name = _graph_name_for_client(req.client_slug)

    try:
        # Connect via Redis protocol (same as FalkorDB driver)
        r = Redis(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
            decode_responses=True,
            socket_timeout=30,
        )

        # Query all nodes
        node_query = f"MATCH (n) RETURN n LIMIT {req.max_nodes}"
        node_result = r.execute_command("GRAPH.QUERY", graph_name, node_query)

        nodes: list[GraphNode] = []
        node_ids: set[str] = set()

        # FalkorDB GRAPH.QUERY returns [header, data, stats]
        if len(node_result) >= 2:
            for record in node_result[1]:
                # Each record is a list with one node
                # Node format varies — extract properties
                node_data = record[0] if isinstance(record, list) else record

                # Parse node properties from the result
                props = {}
                name = "Unknown"
                node_id = ""
                node_type = "entity"

                if isinstance(node_data, list):
                    # FalkorDB returns [labels, properties] for nodes
                    if len(node_data) >= 3:
                        labels = node_data[0] if isinstance(node_data[0], list) else [node_data[0]]
                        # Properties are key-value pairs
                        prop_data = node_data[2] if len(node_data) > 2 else node_data[1]
                        if isinstance(prop_data, list):
                            for i in range(0, len(prop_data), 2):
                                if i + 1 < len(prop_data):
                                    props[str(prop_data[i])] = prop_data[i + 1]

                        node_id = str(props.get('uuid', props.get('id', str(id(node_data)))))
                        name = str(props.get('name', props.get('label', f'Node')))

                        if labels:
                            label = str(labels[0]).lower() if labels[0] else 'entity'
                            if 'entity' in label:
                                node_type = 'entity'
                            elif 'episode' in label:
                                node_type = 'episode'
                            elif 'community' in label:
                                node_type = 'community'

                if not node_id:
                    continue

                # Infer type from name/properties
                name_lower = name.lower()
                summary = str(props.get('summary', '')).lower()

                if any(d in name_lower for d in ['audit', 'licensing', 'compliance', 'enforcement', 'executive', 'gaming technology']):
                    if 'department' in name_lower or 'division' in name_lower or 'director' in summary:
                        node_type = 'department'
                if any(s in name_lower for s in ['permitrak', 'filemaker', 'teammate', 'key traka', 'igt', 'software']):
                    node_type = 'system'
                if node_type == 'entity':
                    if any(w in summary for w in ['pain', 'problem', 'issue', 'challenge', 'friction']):
                        node_type = 'pain_point'
                    elif any(w in summary for w in ['opportunity', 'improve', 'potential', 'automat']):
                        node_type = 'opportunity'
                    elif any(w in summary for w in ['process', 'workflow', 'procedure']):
                        node_type = 'process'
                    elif any(w in summary for w in ['person', 'director', 'manager', 'inspector']):
                        node_type = 'person'

                if node_id not in node_ids:
                    node_ids.add(node_id)
                    clean_props = {}
                    for k, v in props.items():
                        if k not in ('uuid', 'id', 'name'):
                            try:
                                clean_props[k] = str(v)[:200] if isinstance(v, str) and len(str(v)) > 200 else v
                            except:
                                pass

                    nodes.append(GraphNode(
                        id=node_id,
                        name=name,
                        type=node_type,
                        properties=clean_props,
                    ))

        # Query all edges
        edge_query = f"MATCH (a)-[r]->(b) RETURN a.uuid, type(r), b.uuid, r.fact LIMIT {req.max_nodes * 3}"
        edge_result = r.execute_command("GRAPH.QUERY", graph_name, edge_query)

        edges: list[GraphEdge] = []
        edge_ids: set[str] = set()

        if len(edge_result) >= 2:
            for record in edge_result[1]:
                if len(record) >= 3:
                    src_id = str(record[0]) if record[0] else ""
                    rel_type = str(record[1]) if record[1] else "RELATED_TO"
                    tgt_id = str(record[2]) if record[2] else ""
                    fact = str(record[3]) if len(record) > 3 and record[3] else ""

                    if not src_id or not tgt_id:
                        continue

                    edge_id = f"{src_id}-{rel_type}-{tgt_id}"
                    if edge_id in edge_ids:
                        continue
                    edge_ids.add(edge_id)

                    if src_id in node_ids and tgt_id in node_ids:
                        edges.append(GraphEdge(
                            id=edge_id,
                            source=src_id,
                            target=tgt_id,
                            label=rel_type,
                            fact=fact,
                        ))

        r.close()

        logger.info(f"[graph] Retrieved {len(nodes)} nodes, {len(edges)} edges from {graph_name}")

        return GraphDataResponse(
            nodes=nodes,
            edges=edges,
            graph_name=graph_name,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    except Exception as e:
        logger.error(f"[graph] Error retrieving graph data from {graph_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve graph data: {str(e)}")
