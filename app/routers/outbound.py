"""HTTP endpoints for the gtm_outbound graph.

Exposed at /outbound/* via app.main. Designed for the Thrive LLM proxy
(Mac Studio) to call as the outcome-ingestion + recipient-retrieval surface.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import outbound_graph

logger = logging.getLogger("graphiti_service")
router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HypothesisIn(BaseModel):
    id: str | None = None
    predicted_reply_rate: float
    rationale: str = ""
    model_version: str = "thrive_llm_unknown"


class RecipientIn(BaseModel):
    id: str | None = None
    full_name: str | None = None
    linkedin_urn: str | None = None
    primary_email: str | None = None
    company: str | None = None
    company_id: str | None = None
    title: str | None = None
    seniority: str | None = None
    function: str | None = None
    sector: str | None = None
    tier: int | None = None
    geo: str | None = None
    dossier_path: str | None = None


class SendIn(BaseModel):
    id: str | None = None
    recipient_id: str
    channel: str = "email"
    drafted_at: str | None = None
    sent_at: str | None = None
    status: str = "drafted"
    variant_id: str | None = None
    framing: str | None = None
    hook_type: str | None = None
    length_tokens: int | None = None
    sub_segment: str | None = None
    body_hash: str | None = None
    body_preview: str | None = None
    drafted_by: str | None = None
    promoted_via: str | None = None
    client_context: str | None = None
    signal_basis: list[str] | None = None
    hypothesis: HypothesisIn | None = None


class OutcomeIn(BaseModel):
    id: str | None = None
    send_id: str
    kind: str  # delivered|opened|replied|meeting_booked|opportunity|closed_won|closed_lost|bounced|unsubscribed
    occurred_at: str | None = None
    latency_seconds: int | None = None
    reply_sentiment: str | None = None  # warm|neutral|negative|auto_reply
    reply_intent: str | None = None
    reply_text_hash: str | None = None
    meeting_id: str | None = None
    deal_id: str | None = None
    attribution_confidence: float | None = None
    observed_by: str | None = None


class CoachingEventIn(BaseModel):
    id: str | None = None
    send_id: str | None = None
    action: str = "REWRITE"  # KEEP|REWRITE|REJECT
    reviewer: str = "jeremy"
    raw_feedback: str = ""
    extracted_rule: str = ""
    scope: str = "general"
    occurred_at: str | None = None


class CohortFilters(BaseModel):
    tier: int | None = None
    function: str | None = None
    sector: str | None = None
    sub_segment: str | None = None
    limit: int = 50


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/init")
async def init_schema():
    """Initialize the gtm_outbound graph: indexes + label seeds. Idempotent."""
    return outbound_graph.init_schema()


@router.get("/health")
async def health():
    """Health check + node count snapshot."""
    return outbound_graph.health()


@router.get("/stats")
async def stats():
    """Per-label node counts."""
    return outbound_graph.graph_stats()


@router.post("/recipient")
async def upsert_recipient(payload: RecipientIn):
    try:
        return outbound_graph.upsert_recipient(payload.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/send")
async def record_send(payload: SendIn):
    try:
        return outbound_graph.record_send(payload.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/outcome")
async def record_outcome(payload: OutcomeIn):
    try:
        return outbound_graph.record_outcome(payload.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/coaching")
async def record_coaching(payload: CoachingEventIn):
    try:
        return outbound_graph.record_coaching_event(payload.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/recipient/{recipient_id}/history")
async def recipient_history(recipient_id: str, limit: int = 10):
    return outbound_graph.get_recipient_history(recipient_id, limit=limit)


@router.post("/cohort/stats")
async def cohort(filters: CohortFilters):
    return outbound_graph.cohort_stats(
        filters={k: v for k, v in filters.model_dump().items() if k != "limit" and v is not None},
        limit=filters.limit,
    )
