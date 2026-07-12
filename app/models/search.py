"""Pydantic models for provenance-aware search requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.provenance_contract import (
    PROVENANCE_SUMMARY_CONTRACT_VERSION,
    PROVENANCE_SHADOW_CONTRACT_VERSION,
    SEARCH_CONTEXT_CONTRACT_VERSION,
)


class SearchContextRequest(BaseModel):
    client_slug: str = Field(..., description="Client identifier for graph isolation")
    engagement_id: str = Field(..., description="Engagement identifier")
    query: str = Field(..., description="Natural language search query")
    max_results: int = Field(default=10, ge=1, le=50)
    include_segment: bool = Field(
        default=False,
        description=(
            "Reserved segment channel. P1 rejects activation until a governed "
            "pattern contract exists."
        ),
    )


class ChainStatus(str, Enum):
    """Graph-structural provenance only; Postgres resolution belongs upstream."""

    CHAINED = "chained"
    PRE_CHAIN = "pre_chain"


class FactSource(BaseModel):
    """Episode/source anchors returned by Graphiti without authority claims."""

    episode_uuid: str
    episode_name: str
    source_description: str
    source_type: str
    source_id: str
    engagement_id: str
    episode_type: str
    anchor_mode: str
    producer_contract_version: str
    valid_at: datetime | None = None


class FactResult(BaseModel):
    """A structurally chained temporal fact from either retrieval producer."""

    fact_id: str = Field(description="Stable RELATES_TO edge UUID")
    subject: str = Field(description="Source entity UUID")
    subject_name: str = Field(description="Source entity display name")
    predicate: str = Field(description="Relationship name")
    object: str = Field(description="Target entity UUID")
    object_name: str = Field(description="Target entity display name")
    fact: str = Field(default="", description="Human-readable fact statement")
    episodes: list[str] = Field(default_factory=list)
    sources: list[FactSource] = Field(default_factory=list)
    chain_status: Literal["chained"] = ChainStatus.CHAINED.value
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    expired_at: datetime | None = None


class ProvenanceSummary(BaseModel):
    """Graphiti-owned portion of the frozen provenance suppression algebra.

    Backend PR-B owns reference/content resolution and the final A4 buckets. This
    service reports only the terminal decisions it can make from graph state.
    """

    contract_version: Literal[PROVENANCE_SUMMARY_CONTRACT_VERSION] = (
        PROVENANCE_SUMMARY_CONTRACT_VERSION
    )
    candidates: int = 0
    service_forwarded: int = 0
    malformed_item_suppressed: int = 0
    expired_suppressed: int = 0
    pre_chain_suppressed: int = 0
    cross_engagement_suppressed: int = 0
    malformed_response_events: int = 0
    retrieval_path: Literal["fast", "hybrid_fallback"]
    requested_results: int
    overfetch_limit: int
    starved_at_service: bool = False


class SearchContextResponse(BaseModel):
    contract_version: Literal[SEARCH_CONTEXT_CONTRACT_VERSION] = (
        SEARCH_CONTEXT_CONTRACT_VERSION
    )
    facts: list[FactResult] = Field(default_factory=list)
    # Retained for additive response compatibility. P1 never populates this
    # channel without a separately governed pattern contract.
    segment_insights: list[str] = Field(default_factory=list)
    graph_name: str = ""
    search_time_ms: float = 0.0
    provenance_summary: ProvenanceSummary


class LegacyFactResult(BaseModel):
    """Exact pre-P1 fact wire retained for rolling compatibility."""

    subject: str
    predicate: str
    object: str
    fact: str = ""
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    expired_at: datetime | None = None


class LegacySearchContextResponse(BaseModel):
    """Pre-P1 response: no contract version and therefore no P1 claim."""

    facts: list[LegacyFactResult] = Field(default_factory=list)
    segment_insights: list[str] = Field(default_factory=list)
    graph_name: str = ""
    search_time_ms: float = 0.0


class ProvenanceShadow(BaseModel):
    """Non-enforcing v3 preview nested under the compatibility response."""

    contract_version: Literal[PROVENANCE_SHADOW_CONTRACT_VERSION] = (
        PROVENANCE_SHADOW_CONTRACT_VERSION
    )
    enforcement_applied: Literal[False] = False
    facts: list[FactResult] = Field(default_factory=list)
    provenance_summary: ProvenanceSummary


class ShadowSearchContextResponse(LegacySearchContextResponse):
    provenance_shadow: ProvenanceShadow
