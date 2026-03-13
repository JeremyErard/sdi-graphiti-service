"""Pydantic models for episode ingestion requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EpisodeType(str, Enum):
    INTERVIEW_SYNTHESIS = "interview_synthesis"
    CROSS_ANALYSIS = "cross_analysis"
    DOCUMENT_ANALYSIS = "document_analysis"
    PROCESS_EXTRACTION = "process_extraction"
    PROCESS_REFINEMENT = "process_refinement"
    FUTURE_STATE = "future_state"
    SOP_GENERATION = "sop_generation"
    STATUS_UPDATE = "status_update"
    ENGAGEMENT_SETUP = "engagement_setup"
    BOOTSTRAP = "bootstrap"


class IngestEpisodeRequest(BaseModel):
    client_slug: str = Field(..., description="Client identifier for graph isolation")
    engagement_id: str = Field(..., description="Engagement identifier for group filtering")
    episode_type: EpisodeType
    content: str = Field(..., description="Text content to ingest")
    source_id: str = Field(..., description="ID of the source entity (interview, document, etc.)")
    source_type: str = Field(..., description="Type of source: interview, document, process, etc.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IngestEpisodeResponse(BaseModel):
    episode_id: str
    entities_extracted: int
    facts_created: int
    graph_name: str


class BootstrapRequest(BaseModel):
    client_slug: str
    engagement_id: str
    documents: list[dict[str, str]] = Field(
        ...,
        description="List of {name, content, category} to seed the graph",
    )
