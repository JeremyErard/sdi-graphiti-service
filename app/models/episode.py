"""Pydantic models for episode ingestion requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.provenance_contract import EPISODE_PROVENANCE_CONTRACT_VERSION


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
    INSIGHT_REPORT = "insight_report"
    # Insight Engine Upgrade 2026-04-23 — per-perspective archive + approved-artifact tracking.
    # These types let the client graph accumulate the structural knowledge that the SOP/map
    # pipelines produce, and preserve each reduce-phase model's independent view before reconciliation.
    SOP_APPROVED = "sop_approved"
    SOP_REGENERATION = "sop_regeneration"
    PROCESS_MAP_APPROVED = "process_map_approved"
    PROCESS_FEEDBACK_RESOLVED = "process_feedback_resolved"
    INSIGHT_OPUS_PERSPECTIVE = "insight_opus_perspective"
    INSIGHT_GEMINI_PERSPECTIVE = "insight_gemini_perspective"
    INSIGHT_GPT_PERSPECTIVE = "insight_gpt_perspective"
    INSIGHT_RECONCILIATION = "insight_reconciliation"


class EpisodeAnchorMode(str, Enum):
    TYPED_SOURCE = "typed_source"
    ENGAGEMENT = "engagement"


class IngestEpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_slug: str = Field(..., description="Client identifier for graph isolation")
    engagement_id: str = Field(..., description="Engagement identifier for group filtering")
    episode_type: EpisodeType
    content: str = Field(..., description="Text content to ingest")
    source_id: str = Field(..., description="ID of the source entity (interview, document, etc.)")
    source_type: str = Field(..., description="Type of source: interview, document, process, etc.")
    anchor_mode: EpisodeAnchorMode | None = Field(
        default=None,
        description="Explicit source granularity for v2 producers",
    )
    producer_contract_version: (
        Literal[EPISODE_PROVENANCE_CONTRACT_VERSION] | None
    ) = Field(
        default=None,
        description="Versioned producer signature; required with anchor_mode",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_anchor_pair(self):
        if (self.anchor_mode is None) != (self.producer_contract_version is None):
            raise ValueError(
                "anchor_mode and producer_contract_version must be supplied together"
            )
        if (
            self.producer_contract_version
            == EPISODE_PROVENANCE_CONTRACT_VERSION
            and self.anchor_mode == EpisodeAnchorMode.ENGAGEMENT
            and self.source_id != self.engagement_id
        ):
            raise ValueError(
                "engagement anchors require source_id to equal engagement_id"
            )
        if (
            self.producer_contract_version
            == EPISODE_PROVENANCE_CONTRACT_VERSION
            and self.source_type == "engagement"
            and (
                self.anchor_mode != EpisodeAnchorMode.ENGAGEMENT
                or self.source_id != self.engagement_id
            )
        ):
            raise ValueError(
                "engagement sources require an exact engagement anchor"
            )
        return self


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
