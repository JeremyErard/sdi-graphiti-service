"""Vectors for projected records: POST /ingest/projection/v2/embed.

Kept out of app/routers/projection.py on purpose. That module applies governed
envelopes and imports nothing but the graph driver and this service's own
models, so it has no path to an embedding provider or anything else. Giving a
projected record its vector is a separate, derived step, and it lives here.
"""

import logging
import uuid as uuidlib

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.auth import GraphPrincipal
from app.config import settings
from app.routers import projection
from app.services import projection_records

logger = logging.getLogger("graphiti_service")

router = APIRouter()


class ProjectionEmbedRequest(BaseModel):
    """Give the current projected records their vectors. Carries no content."""

    model_config = ConfigDict(extra="forbid")

    client_slug: str = Field(pattern=r"^[a-z0-9-]+$", min_length=1, max_length=128)
    engagement_id: str | None = Field(default=None, min_length=1, max_length=256)
    limit: int = Field(
        default=projection_records.MAX_EMBED_PER_CALL,
        ge=1,
        le=projection_records.MAX_EMBED_PER_CALL,
    )


class ProjectionEmbedResponse(BaseModel):
    graph_name: str
    pending: int
    written: int
    # Statements the embedding provider refused even one at a time; they stay pending.
    failed: int = 0
    remaining: bool


@router.post("/projection/v2/embed", response_model=ProjectionEmbedResponse)
async def embed_projection_records(
    request: ProjectionEmbedRequest,
    principal: GraphPrincipal = Depends(projection.INGEST_PRINCIPAL),
):
    """Embed the statement of every current projected record that has no vector.

    The apply restates a row's whole property map, so a reprojected row loses
    its vector with everything else and is pending again; the caller runs this
    after a projection and repeats while `remaining` is true. It writes only
    the derived vector and its marker, runs no language model, and is safe to
    call at any time: with nothing pending it does nothing.
    """
    projection._guard_lane(principal, request.client_slug)
    # The pass embeds records of either lane that writes them: the process structure, or the law.
    if not (settings.projection_v2_allow_process_structure or settings.projection_v2_allow_law):
        raise projection._fail(
            422,
            "record projection is not authorized on this deployment in this phase; set "
            "PROJECTION_V2_ALLOW_PROCESS_STRUCTURE or PROJECTION_V2_ALLOW_LAW once a lane "
            "is switched on",
        )
    graph_name = projection._derive_graph_name(request.client_slug)
    correlation_id = uuidlib.uuid4().hex
    try:
        outcome = await projection_records.embed_pending(
            client_slug=request.client_slug,
            engagement_id=request.engagement_id,
            limit=request.limit,
            graph_name=graph_name,
        )
    except projection_records.EmbeddingUnavailable as exc:
        # The provider is not answering. Say so, rather than a success that tells the caller to try again.
        logger.warning(
            "[graphiti] record embedding stopped correlation_id=%s graph=%s: %s",
            correlation_id,
            graph_name,
            exc,
        )
        raise projection._fail(
            502,
            {
                "code": "projection_embed_provider_unavailable",
                "message": "the embedding provider is not answering; records stay pending and are findable by keyword",
                "correlation_id": correlation_id,
            },
        )
    except Exception as exc:
        logger.exception(
            "[graphiti] record embedding failed correlation_id=%s graph=%s: %s",
            correlation_id,
            graph_name,
            exc,
        )
        raise projection._fail(
            500,
            {
                "code": "projection_embed_failed",
                "message": "record embedding failed; see the service log for this correlation id",
                "correlation_id": correlation_id,
            },
        )
    return ProjectionEmbedResponse(**outcome)
