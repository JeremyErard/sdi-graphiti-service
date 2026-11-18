"""Request and response models for reading sections of a law exactly, by identity.

No search is involved: a section is asked for by its law and its section
number, and comes back as stored, or is reported as not found.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LAW_SECTIONS_CONTRACT_VERSION = "graphiti_law_sections_v1"
LAW_SOURCES_CONTRACT_VERSION = "graphiti_law_sources_v1"

# The most sections one request may ask for.
MAX_LAW_SECTION_ITEMS = 50


class _Scope(BaseModel):
    """The same tenant and engagement scope as POST /search/records."""

    model_config = ConfigDict(extra="forbid")

    client_slug: str = Field(..., pattern=r"^[a-z0-9][a-z0-9-]{0,127}$")
    engagement_id: str = Field(..., min_length=1, max_length=256)
    readable_engagement_ids: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        default_factory=list,
        max_length=50,
        description="Other engagements of this client the caller may read",
    )

    def engagement_order(self) -> list[str]:
        """The engagements this request may read, the requesting one first, each once."""
        return list(dict.fromkeys([self.engagement_id, *self.readable_engagement_ids]))


class LawSectionRef(BaseModel):
    """One section asked for: its law's id (as POST /law/sources lists it) and its section number as written."""

    model_config = ConfigDict(extra="forbid")

    law: str = Field(..., min_length=1, max_length=256, description="The law's id, as POST /law/sources lists it")
    section: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description=(
            "The section number as written (for example 1.01). Matched exactly after trimming whitespace at "
            "either end (Python str.strip: Unicode white space). No case folding and no Unicode normalization."
        ),
    )

    version: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "A version of the law's text to read, as an earlier answer gave it in `version_id`. Matched exactly "
            "after trimming whitespace. Omitted, the version in force is read."
        ),
    )

    @field_validator("law", "section", "version")
    @classmethod
    def _trimmed_and_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Whitespace at either end is trimmed and nothing else is changed: no case folding, no number parsing,
        # no Unicode normalization. "1.1" is not "1.10", and "1.01a" is not "1.01".
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("must not be blank")
        # The graph driver cannot carry a NUL in a query parameter, and the write path cannot store one, so no
        # stored id or number holds one. Refused here, so one such item cannot fail a whole request as a read error.
        if "\x00" in trimmed:
            raise ValueError("must not contain a NUL character")
        return trimmed


class LawSectionsRequest(_Scope):
    """Fetch sections of the laws in force exactly, by law and section number."""

    items: list[LawSectionRef] = Field(..., min_length=1)

    @field_validator("items")
    @classmethod
    def _within_the_cap(cls, value: list[LawSectionRef]) -> list[LawSectionRef]:
        if len(value) > MAX_LAW_SECTION_ITEMS:
            raise ValueError(
                f"at most {MAX_LAW_SECTION_ITEMS} sections may be asked for in one request "
                f"({len(value)} were); send the rest in another request"
            )
        return value


class LawSectionPart(BaseModel):
    """One stored record of a section. A long section is stored as several parts, in order."""

    record_id: str = Field(description="Exact projected node id")
    part: int | None = Field(default=None, description="This part's number (1-based), when the record states one")
    text: str | None = Field(description="The record's words exactly as stored; null only when none are stored")
    source_id: str = ""
    content_hash: str = ""


class LawSectionResult(BaseModel):
    law: str = Field(description="The law's id, as asked for (trimmed)")
    section: str = Field(description="The section number, as asked for (trimmed)")
    version: str | None = Field(default=None, description="The version pinned, as asked for (trimmed); null when none was")
    status: Literal["found", "not_found", "incomplete"] = Field(
        description=(
            "found: every part of the section is here. not_found: no section with exactly this number is stored "
            "for this law at the version asked for (the version in force when none is pinned) in the engagements "
            "read. incomplete: parts are stored, but not all of the parts "
            "the records say there are (or a part has no words), so the text must not be shown as whole."
        )
    )
    engagement_id: str = Field(default="", description="The engagement the section was read from")
    law_title: str = ""
    heading: str = ""
    chapter: str = ""
    version_id: str = Field(
        default="",
        description=(
            "The version of the law's text these parts belong to, as the write path stores it on each record. "
            "Record it with the law and section to pin this text; send it back as `version` to read it again."
        ),
    )
    current_version_id: str = Field(default="", description="The version of the law's text in force now")
    in_force: bool = Field(default=False, description="True when version_id is the version in force now")
    parts_expected: int | None = Field(default=None, description="How many parts the records say the section has")
    parts: list[LawSectionPart] = Field(default_factory=list)


class LawSectionsResponse(BaseModel):
    contract_version: Literal[LAW_SECTIONS_CONTRACT_VERSION] = LAW_SECTIONS_CONTRACT_VERSION
    # One entry per item asked for, in the order asked.
    sections: list[LawSectionResult] = Field(default_factory=list)
    graph_name: str = ""


class LawSourcesRequest(_Scope):
    """List the laws in force for the engagements a caller may read."""


class LawSourceResult(BaseModel):
    law: str = Field(description="The law's id: what POST /law/sections takes as `law`")
    engagement_id: str
    title: str = ""
    version_id: str = Field(default="", description="The version of the law's text in force")
    sections: int = Field(default=0, description="Distinct section numbers in force, counted from the stored records")
    parts: int = Field(default=0, description="Records in force, counted from the stored records")


class LawSourcesResponse(BaseModel):
    contract_version: Literal[LAW_SOURCES_CONTRACT_VERSION] = LAW_SOURCES_CONTRACT_VERSION
    sources: list[LawSourceResult] = Field(default_factory=list)
    truncated: bool = Field(default=False, description="True when there were more laws than one answer lists")
    graph_name: str = ""
