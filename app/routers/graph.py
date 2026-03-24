"""Graph traversal endpoints for Knowledge Map visualization."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
    """Return all nodes and edges from a client's knowledge graph for visualization."""
    graph_name = _graph_name_for_client(req.client_slug)

    try:
        from falkordb import FalkorDB
        db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
        )
        graph = db.select_graph(graph_name)

        # Query all nodes
        node_result = graph.query(
            f"MATCH (n) RETURN n LIMIT {req.max_nodes}"
        )

        nodes: list[GraphNode] = []
        node_ids: set[str] = set()

        for record in node_result.result_set:
            node = record[0]
            props = dict(node.properties) if hasattr(node, 'properties') else {}
            node_id = str(props.get('uuid', props.get('id', node.id)))
            name = str(props.get('name', props.get('label', f'Node {node.id}')))

            # Determine node type from labels
            labels = node.labels if hasattr(node, 'labels') else []
            node_type = labels[0].lower() if labels else 'entity'

            # ── Step 1: Map FalkorDB node labels to base types ──
            LABEL_MAP = {
                'entity': 'entity',
                'entitynode': 'entity',
                'episode': 'episode',
                'episodenode': 'episode',
                'community': 'community',
                'communitynode': 'community',
            }
            mapped_type = LABEL_MAP.get(node_type, node_type)

            # ── Step 2: Use Graphiti LLM-assigned entity_type (authoritative) ──
            # Graphiti stores the domain-specific type from config.yaml
            # in the entity_type property (e.g., "Stakeholder", "Department")
            graphiti_type = str(props.get('entity_type', '')).strip()

            GRAPHITI_TYPE_MAP = {
                # People
                'stakeholder': 'person',
                'consultant': 'person',
                # Organization
                'department': 'department',
                'role': 'role',
                # Process & Technology
                'process': 'process',
                'system': 'system',
                # Insights
                'finding': 'theme',
                'painpoint': 'pain_point',
                'pain_point': 'pain_point',
                # Opportunities
                'opportunity': 'opportunity',
                'goal': 'opportunity',
                # Compliance
                'regulation': 'regulation',
                'policy': 'policy',
                # Strategy
                'decision': 'process',
                'risk': 'pain_point',
                'metric': 'metric',
                # Deliverables
                'deliverable': 'entity',
            }

            if graphiti_type and graphiti_type.lower() in GRAPHITI_TYPE_MAP:
                mapped_type = GRAPHITI_TYPE_MAP[graphiti_type.lower()]
            elif mapped_type in ('entity', 'entitynode'):
                # ── Step 3: Heuristic fallback for nodes without entity_type ──
                name_lower = name.lower().strip()
                summary = str(props.get('summary', '')).lower()

                DEPARTMENTS = {'internal audit', 'audit department', 'compliance & enforcement',
                              'compliance and enforcement', 'c&e', 'compliance & enforcement (c&e)',
                              'licensing & investigations', 'licensing and investigations',
                              'gaming technology', 'gtu', 'executive', 'executive office',
                              'casino operations', 'surveillance', 'pokagon band gaming commission',
                              'pbgc', 'information technology', 'human resources'}

                SYSTEMS = {'permitrak', 'filemaker', 'fmp', 'teammate', 'teamrisk', 'teamschedule',
                          'key traka', 'traka', 'premisys', 'sharefile', 'active directory', 'vmware',
                          'adp', 'zendesk', 'kambi', 'crossmatch', 'barracuda', 'infogenesis',
                          'itraq', 'kiteworks', 'powerkiosk', 'geocomply', 'pala interactive',
                          'casino cash trac', 'table manager', 'igt'}

                if name_lower in DEPARTMENTS:
                    mapped_type = 'department'
                elif name_lower in SYSTEMS:
                    mapped_type = 'system'
                elif any(w in summary for w in ['pain', 'problem', 'crisis', 'failure', 'gap']):
                    mapped_type = 'pain_point'
                elif any(w in summary for w in ['opportunity', 'should consider', 'recommend']):
                    mapped_type = 'opportunity'
                elif any(w in summary for w in ['process', 'workflow', 'procedure', 'protocol']):
                    mapped_type = 'process'

            if node_id not in node_ids:
                node_ids.add(node_id)
                # Clean properties for JSON serialization
                clean_props = {}
                for k, v in props.items():
                    if k not in ('uuid', 'id', 'name', 'label'):
                        try:
                            clean_props[k] = str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                        except:
                            clean_props[k] = str(v)

                nodes.append(GraphNode(
                    id=node_id,
                    name=name,
                    type=mapped_type,
                    properties=clean_props,
                ))

        # Query all edges between the retrieved nodes
        edge_result = graph.query(
            f"MATCH (a)-[r]->(b) RETURN a, r, b LIMIT {req.max_nodes * 3}"
        )

        edges: list[GraphEdge] = []
        edge_ids: set[str] = set()

        for record in edge_result.result_set:
            src_node = record[0]
            rel = record[1]
            tgt_node = record[2]

            src_props = dict(src_node.properties) if hasattr(src_node, 'properties') else {}
            tgt_props = dict(tgt_node.properties) if hasattr(tgt_node, 'properties') else {}
            rel_props = dict(rel.properties) if hasattr(rel, 'properties') else {}

            src_id = str(src_props.get('uuid', src_props.get('id', src_node.id)))
            tgt_id = str(tgt_props.get('uuid', tgt_props.get('id', tgt_node.id)))
            rel_type = rel.relation if hasattr(rel, 'relation') else 'RELATED_TO'

            edge_id = f"{src_id}-{rel_type}-{tgt_id}"
            if edge_id in edge_ids:
                continue
            edge_ids.add(edge_id)

            fact = str(rel_props.get('fact', ''))

            # Only include edges where both nodes are in our node set
            if src_id in node_ids and tgt_id in node_ids:
                clean_props = {}
                for k, v in rel_props.items():
                    if k != 'fact':
                        try:
                            clean_props[k] = str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                        except:
                            clean_props[k] = str(v)

                edges.append(GraphEdge(
                    id=edge_id,
                    source=src_id,
                    target=tgt_id,
                    label=rel_type,
                    fact=fact,
                    properties=clean_props,
                ))

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
