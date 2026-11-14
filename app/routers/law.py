"""Sections of a law, read exactly by identity (no search).

POST /law/sections answers, for each (law, section number) asked for, the
section letter for letter as stored, or a plain "not found": at the version
in force, or at a version the caller pins (the `version_id` an earlier answer
gave, which is the version id the write path stores on every record).
POST /law/sources lists the laws in force a caller may ask about.

Both take the tenant and engagement scope POST /search/records takes, under the
same signed "search" scope, and are open on the same terms as that route: one
of the lanes that write records is on (the process structure or the law) and
the signed perimeter is closed. The law switch governs writing, not reading, as
it does for the law search: with it off and the process lane on, a law already
written is still read. Unlike the search, which answers empty when it cannot
answer (it is an addition to another search), these routes say so with an
error: a reader that shows "the sections this procedure cites" must never show
a section as missing because a read failed or reads were switched off.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.law import (
    LawSectionResult,
    LawSectionsRequest,
    LawSectionsResponse,
    LawSourceResult,
    LawSourcesRequest,
    LawSourcesResponse,
)
from app.services import law_sections

logger = logging.getLogger("graphiti_service")

router = APIRouter()


def _require_reads_open() -> None:
    """The condition POST /search/records serves records under, refused out loud instead of answered empty."""
    if not (
        (settings.projection_v2_allow_process_structure or settings.projection_v2_allow_law)
        and settings.graphiti_auth_mode == "required"
    ):
        raise HTTPException(status_code=503, detail="Law reads are not enabled on this service")


@router.post("/sections", response_model=LawSectionsResponse)
async def law_sections_by_identity(req: LawSectionsRequest):
    """Each section asked for, exactly as stored, or not found. No fuzzy match, no nearest section."""
    _require_reads_open()
    try:
        graph_name, found = await law_sections.fetch_sections(
            req.client_slug,
            req.engagement_order(),
            [(item.law, item.section, item.version) for item in req.items],
        )
    except law_sections.LawReadFailed:
        raise HTTPException(status_code=500, detail="Law read failed")
    return LawSectionsResponse(sections=[LawSectionResult(**entry) for entry in found], graph_name=graph_name)


@router.post("/sources", response_model=LawSourcesResponse)
async def law_sources(req: LawSourcesRequest):
    """The laws in force for the engagements read: id, title, version, and how many sections are stored."""
    _require_reads_open()
    try:
        graph_name, sources, truncated = await law_sections.list_sources(req.client_slug, req.engagement_order())
    except law_sections.LawReadFailed:
        raise HTTPException(status_code=500, detail="Law read failed")
    return LawSourcesResponse(
        sources=[LawSourceResult(**entry) for entry in sources], truncated=truncated, graph_name=graph_name
    )
