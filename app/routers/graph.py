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

            # Map Graphiti labels to visualization types
            type_map = {
                'entity': 'entity',
                'entitynode': 'entity',
                'episode': 'episode',
                'episodenode': 'episode',
                'community': 'community',
                'communitynode': 'community',
            }
            mapped_type = type_map.get(node_type, node_type)

            # ── Improved type inference ──
            name_lower = name.lower().strip()
            summary = str(props.get('summary', '')).lower()

            # Known locations — NOT people
            LOCATIONS = {'new buffalo', 'hartford', 'dowagiac', 'south bend', 'four winds'}
            # Known documents/reports — NOT people
            DOC_INDICATORS = ['document', 'report', 'dataset', 'checklist', 'template', 'spreadsheet',
                              'analysis', '.xlsx', '.pdf', '.docx', 'combined', 'all stats']
            # Known operational concepts — NOT people
            CONCEPT_INDICATORS = ['shift', 'coverage', 'scheduling', 'staffing', 'volume', 'model',
                                  'issuance', 'activity', 'incident report', 'coin-in', 'admission',
                                  'wagering', 'drop', 'count', 'seal', 'shipment']
            # Known organizational units — departments
            DEPARTMENTS = {'internal audit', 'audit department', 'compliance & enforcement',
                          'compliance and enforcement', 'c&e', 'compliance & enforcement (c&e)',
                          'compliance and enforcement department', 'licensing & investigations',
                          'licensing and investigations', 'licensing department', 'compliance department',
                          'gaming technology', 'gaming technology department', 'gtu',
                          'executive', 'executive office', 'casino operations', 'surveillance',
                          'pokagon band gaming commission', 'pbgc', 'information technology',
                          'licensing and investigation division', 'human resources'}
            # Known systems
            SYSTEMS = ['permitrak', 'filemaker', 'fmp', 'teammate', 'teamrisk', 'teamschedule',
                       'key traka', 'traka', 'igt', 'table manager', 'premisys', 'sharefile',
                       'active directory', 'vmware', 'adp', 'zendesk', 'kambi', 'crossmatch',
                       'barracuda', 'infogenesis', 'itraq', 'kiteworks', 'powerkiosk', 'geocomply',
                       'pala interactive', 'casino cash trac', 'ai integration']

            if mapped_type == 'entity' or mapped_type == 'entitynode':
                # 1. Department (strict allowlist)
                if name_lower in DEPARTMENTS:
                    mapped_type = 'department'
                # 2. Location
                elif any(loc in name_lower for loc in LOCATIONS):
                    mapped_type = 'entity'  # keep as generic entity, not person
                # 3. Document/report
                elif any(d in name_lower for d in DOC_INDICATORS):
                    mapped_type = 'entity'
                # 4. Operational concept
                elif any(c in name_lower for c in CONCEPT_INDICATORS):
                    mapped_type = 'process'
                # 5. System
                elif any(s in name_lower for s in SYSTEMS):
                    mapped_type = 'system'
                # 6. Person — only if the name looks like a person name
                #    (First Last pattern, or explicit role title prefix)
                elif ' ' in name and len(name.split()) <= 4 and not any(c.isdigit() for c in name):
                    # Check if summary says this is a person
                    if any(w in summary for w in ['holds this role', 'holds the position', 'serves as',
                                                   'reporting to', 'is a ', 'is the ']):
                        mapped_type = 'person'
                    # Check if it's a known role title
                    elif any(name_lower.startswith(p) for p in ['director of', 'chief ', 'commissioner ']):
                        mapped_type = 'person'
                    # Otherwise check if summary strongly indicates a person (by name reference)
                    elif any(w in summary for w in [' he ', ' she ', ' his ', ' her ', ' they ']):
                        mapped_type = 'person'
                # 7. Summary-based inference for remaining entities
                if mapped_type == 'entity':
                    if any(w in summary for w in ['pain', 'problem', 'crisis', 'failure', 'gap']):
                        mapped_type = 'pain_point'
                    elif any(w in summary for w in ['opportunity', 'should consider', 'recommend', 'potential']):
                        mapped_type = 'opportunity'
                    elif any(w in summary for w in ['process', 'workflow', 'procedure', 'lifecycle', 'protocol']):
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
