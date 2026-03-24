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

        # Query entity and community nodes (skip episodes — too heavy for viz)
        node_result = graph.query(
            f"MATCH (n) WHERE NOT n:Episodic RETURN n LIMIT {req.max_nodes}"
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

            # ── Step 2: Use Graphiti LLM-assigned type (authoritative) ──
            # Graphiti stores domain-specific types from config.yaml in the
            # 'labels' property (e.g., "Stakeholder", "Department", "System")
            # This is distinct from node.labels which is the FalkorDB graph label.
            graphiti_labels = props.get('labels', '')
            # labels may be a string or list; normalize to string
            if isinstance(graphiti_labels, list):
                graphiti_type = graphiti_labels[0] if graphiti_labels else ''
            else:
                graphiti_type = str(graphiti_labels).strip()

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
                # ── Step 3: Heuristic classification ──
                # Graphiti stores all entity nodes with generic 'Entity' label,
                # so we must infer types from name + summary content.
                name_lower = name.lower().strip()
                summary = str(props.get('summary', '')).lower()

                # ── Departments (exact match) ──
                DEPARTMENTS = {'internal audit', 'audit department', 'compliance & enforcement',
                              'compliance and enforcement', 'c&e', 'compliance & enforcement (c&e)',
                              'licensing & investigations', 'licensing and investigations',
                              'licensing and investigations (l&i)', 'gaming technology', 'gtu',
                              'executive', 'executive office', 'casino operations', 'surveillance',
                              'pokagon band gaming commission', 'pbgc', 'information technology',
                              'human resources', 'gaming technology unit'}

                # ── Systems (exact match on full name) ──
                SYSTEMS = {'permitrak', 'filemaker', 'filemaker pro', 'fmp', 'teammate',
                          'teamrisk', 'teamschedule', 'teamtec', 'teamcentral',
                          'key traka', 'trakaweb', 'premisys', 'sharefile',
                          'active directory', 'vmware', 'adp', 'zendesk', 'kambi',
                          'crossmatch', 'barracuda', 'infogenesis', 'itraq', 'kiteworks',
                          'powerkiosk', 'geocomply', 'pala interactive', 'casino cash trac',
                          'table manager', 'igt', 'iris', 'merydyan prime', 'pryme',
                          'casino insight', 'advantage monitor', 'servicenow',
                          'hms', 'spasoft', 'filemakergo', 'filemaker go',
                          'igc skins', 'ez pay', 'campo'}

                # ── Regulations / compliance items (name prefix or exact) ──
                REGULATION_PREFIXES = ['sc-', 'fips ', 'fips-', 'cjis ', 'title 31',
                                       'title-31', 'nist ', 'nigc ']
                REGULATIONS = {'tribal gaming compact', 'gaming regulatory act',
                              'cjis security policy', 'nigc', 'title 31', 'title-31',
                              'bsa', 'bank secrecy act'}

                # ── Policy / configuration items (name contains) ──
                POLICY_INDICATORS = ['policy', 'lockout', 'configuration',
                                     'audit checklist', 'security checklist',
                                     'sop', 'standard operating']

                # 1. Department (exact)
                if name_lower in DEPARTMENTS:
                    mapped_type = 'department'
                # 2. System (exact)
                elif name_lower in SYSTEMS:
                    mapped_type = 'system'
                # 3. Regulation (exact or prefix — BEFORE summary-based system)
                elif name_lower in REGULATIONS or any(name_lower.startswith(p) for p in REGULATION_PREFIXES):
                    mapped_type = 'regulation'
                # 4. Policy (name contains indicator — BEFORE summary-based system)
                elif any(p in name_lower for p in POLICY_INDICATORS):
                    mapped_type = 'policy'
                # 5. Pain points (strong signals)
                elif any(w in summary for w in ['failed', 'broken', 'defect', 'error',
                        'falsified', 'breach', 'vulnerability', 'violation']):
                    mapped_type = 'pain_point'
                elif any(w in summary for w in ['pain point', 'problem', 'crisis',
                        'gap', 'missing', 'absent', 'no documentation']):
                    mapped_type = 'pain_point'
                # 6. Opportunity
                elif any(w in summary for w in ['opportunity', 'should consider',
                        'recommend', 'potential benefit', 'could improve']):
                    mapped_type = 'opportunity'
                # 7. Regulation (summary-based — BEFORE system)
                elif any(w in summary for w in ['regulation', 'compliance standard',
                        'regulatory requirement', 'compact', 'statute',
                        'cjis', 'nist', 'security policy area']):
                    mapped_type = 'regulation'
                # 8. Policy (summary-based — BEFORE system)
                elif any(w in summary for w in ['policy', 'standard operating',
                        'internal rule', 'directive', 'security control',
                        'configuration requirement']):
                    mapped_type = 'policy'
                # 9. System (summary-based)
                elif any(w in summary for w in ['software', 'platform', 'application',
                        'database', 'system used']):
                    mapped_type = 'system'
                # 10. Process
                elif any(w in summary for w in ['process', 'workflow', 'procedure',
                        'protocol', 'lifecycle', 'steps to']):
                    mapped_type = 'process'
                # 11. Department (summary-based)
                elif any(w in summary for w in ['department', 'division', 'team',
                        'unit within', 'organizational']):
                    mapped_type = 'department'

            if node_id not in node_ids:
                node_ids.add(node_id)
                # Clean properties for JSON serialization (skip heavy fields)
                SKIP_PROPS = {'uuid', 'id', 'name', 'label', 'content',
                              'name_embedding', 'entity_edges', 'source'}
                clean_props = {}
                for k, v in props.items():
                    if k not in SKIP_PROPS:
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

        # Query edges — only between non-episode nodes, return minimal data
        edge_result = graph.query(
            "MATCH (a)-[r]->(b) "
            "WHERE NOT a:Episodic AND NOT b:Episodic "
            f"RETURN a.uuid, a.id, id(a), r, b.uuid, b.id, id(b) LIMIT {req.max_nodes * 2}"
        )

        edges: list[GraphEdge] = []
        edge_ids: set[str] = set()

        for record in edge_result.result_set:
            # Extract source/target IDs from returned properties
            src_id = str(record[0] or record[1] or record[2])
            rel = record[3]
            tgt_id = str(record[4] or record[5] or record[6])

            rel_props = dict(rel.properties) if hasattr(rel, 'properties') else {}
            rel_type = rel.relation if hasattr(rel, 'relation') else 'RELATED_TO'

            edge_id = f"{src_id}-{rel_type}-{tgt_id}"
            if edge_id in edge_ids:
                continue
            edge_ids.add(edge_id)

            fact = str(rel_props.get('fact', ''))

            # Only include edges where both nodes are in our node set
            if src_id in node_ids and tgt_id in node_ids:
                edges.append(GraphEdge(
                    id=edge_id,
                    source=src_id,
                    target=tgt_id,
                    label=rel_type,
                    fact=fact,
                    properties={},
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
