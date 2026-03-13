"""Pydantic models for search requests and responses."""

from datetime import datetime

from pydantic import BaseModel, Field


class SearchContextRequest(BaseModel):
    client_slug: str = Field(..., description="Client identifier for graph isolation")
    engagement_id: str = Field(..., description="Engagement identifier")
    query: str = Field(..., description="Natural language search query")
    max_results: int = Field(default=10, ge=1, le=50)
    include_segment: bool = Field(
        default=True, description="Also search segment knowledge graph"
    )


class FactResult(BaseModel):
    """A temporal fact from the knowledge graph (Graphiti EntityEdge)."""

    subject: str = Field(description="Source entity UUID")
    predicate: str = Field(description="Relationship name")
    object: str = Field(description="Target entity UUID")
    fact: str = Field(default="", description="Human-readable fact statement")
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    expired_at: datetime | None = None


class SearchContextResponse(BaseModel):
    facts: list[FactResult] = Field(default_factory=list)
    segment_insights: list[str] = Field(default_factory=list)
    graph_name: str = ""
    search_time_ms: float = 0.0
