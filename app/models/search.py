"""Pydantic models for search requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SearchContextRequest(BaseModel):
    client_slug: str = Field(..., description="Client identifier for graph isolation")
    engagement_id: str = Field(..., description="Engagement identifier")
    query: str = Field(..., description="Natural language search query")
    entity_types: list[str] | None = Field(
        default=None, description="Filter by entity type names"
    )
    max_results: int = Field(default=10, ge=1, le=50)
    include_segment: bool = Field(
        default=True, description="Also search segment knowledge graph"
    )


class EntityResult(BaseModel):
    name: str
    entity_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0
    mention_count: int = 0


class FactResult(BaseModel):
    subject: str
    predicate: str
    object: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 0.0
    source_description: str = ""


class ContradictionResult(BaseModel):
    fact_a: str
    fact_b: str
    source_a: str
    source_b: str
    detected_at: datetime


class SearchContextResponse(BaseModel):
    nodes: list[EntityResult] = Field(default_factory=list)
    facts: list[FactResult] = Field(default_factory=list)
    contradictions: list[ContradictionResult] = Field(default_factory=list)
    segment_insights: list[str] = Field(default_factory=list)
    graph_name: str = ""
    search_time_ms: float = 0.0
