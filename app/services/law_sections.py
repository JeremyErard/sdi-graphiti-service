"""Sections of a law, read exactly by identity: no search, no ranking, no nearest match.

A reader that shows the sections a procedure cites needs those sections letter
for letter, or a plain "not there". This module answers that and nothing else.

What a section is, as the projection writes it (/ingest/projection/v2, kinds
LAW_SOURCE and LAW_SECTION, edge SECTION_OF):

* the law is one node (LAW_SOURCE, base label ProjectionNode) whose id is
  stable across versions of its text, and whose `current_law_version_id` says
  which version is in force;
* each section is one or more nodes (LAW_SECTION, base label LawRecordNode),
  one per part of the section, each naming its law in `workflow_node_id`, its
  version in `projection_source_version_id`, its section number in `section`,
  its heading in `title`, and, when the section is cut into parts, `part` and
  `parts`. Its words are under `law_text` (the write moves them there from
  `text`).

By default a section is read at the version in force: the same rule the record
and law searches use (projection_records._IN_FORCE), its law in the same tenant
and engagement pointing at its version. A caller may instead pin a version it
recorded earlier (the `version_id` an answer carries, which is the version id
the write path stores on every record of the section); that version is read
exactly, and the answer says whether it is still the one in force. Nothing here
makes up an identifier: every id returned is one the write path stored.

Every statement is a constant: labels, kinds and property names come from the
model's own tables, and every value from a request travels as a query
parameter, never as Cypher text.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import settings
from app.models.projection import (
    LAW_SECTION_BASE_LABEL,
    LAW_TEXT_PROPERTY,
    POINTER_ANCHOR_KIND,
    PROJECTION_NODE_LABEL,
    RECORD_WORKFLOW_PROPERTY,
    ProjectionSourceKind,
)
from app.services import graphiti_client
from app.services.projection_records import LAW_KINDS, _read

logger = logging.getLogger("graphiti_service")

# The pointer a law carries, read off the model: the one whose anchor is a law.
_LAW_POINTERS = [p for p, anchor in POINTER_ANCHOR_KIND.items() if anchor is ProjectionSourceKind.LAW_SOURCE]
if len(_LAW_POINTERS) != 1:  # pragma: no cover - a guard on the model
    raise RuntimeError("a law carries exactly one pointer; the exact read is written for one")
LAW_POINTER = _LAW_POINTERS[0]
_LAW_SOURCE_KIND = ProjectionSourceKind.LAW_SOURCE.value
_LAW_KIND_LIST = "[" + ", ".join(f"'{kind}'" for kind in LAW_KINDS) + "]"

# A non-vector read of a few hundred nodes at most; the keyword read's limit is the nearest kind of read.
LAW_READ_TIMEOUT_MS = settings.search_bm25_timeout_ms
# The most laws one listing returns.
MAX_LAW_SOURCES = 200

# The law named by each item, in one of the engagements read; then every record of one version of it whose section
# number is exactly the one asked for. The version is the one the item pins (`item.version`), or, when it pins
# none, the one the law's pointer holds (the version in force). A pinned version is read whether or not it is still
# in force: an older version stays on record, and a caller that recorded "this section at this version" gets that
# text back, told whether it is still in force. `n.section = item.section` is plain equality on the stored value: a
# stored number is never equal to a string, and nothing is trimmed, folded or parsed here. The section is an
# OPTIONAL MATCH so that every engagement holding the law answers a row, with or without the section: which
# engagement's law an item is read from is decided by who holds the law, not by who holds the section.
SECTIONS_QUERY = (
    "UNWIND $items AS item "
    f"MATCH (w:{PROJECTION_NODE_LABEL} {{uuid: item.law}}) "
    "WHERE w.group_id = $group_id AND w.engagement_id IN $engagement_ids "
    f"AND w.projection_source_kind = '{_LAW_SOURCE_KIND}' "
    f"OPTIONAL MATCH (n:{LAW_SECTION_BASE_LABEL}) "
    "WHERE n.group_id = $group_id AND n.engagement_id = w.engagement_id "
    f"AND n.{RECORD_WORKFLOW_PROPERTY} = w.uuid "
    f"AND n.projection_source_kind IN {_LAW_KIND_LIST} "
    f"AND n.projection_source_version_id = coalesce(item.version, w.{LAW_POINTER}) "
    "AND n.section = item.section "
    "RETURN item.index AS item, w.engagement_id AS engagement_id, w.name AS law_name, "
    "n.projection_source_version_id AS version_id, n.uuid AS record_id, n.title AS heading, n.chapter AS chapter, "
    f"n.law_title AS law_title, n.part AS part, n.parts AS parts, n.{LAW_TEXT_PROPERTY} AS text, "
    "n.projection_source_id AS source_id, n.projection_source_content_hash AS content_hash, "
    f"w.{LAW_POINTER} AS current_version_id"
)

# Every law with a version in force in the engagements read, and what of that version is stored.
SOURCES_QUERY = (
    f"MATCH (w:{PROJECTION_NODE_LABEL}) "
    "WHERE w.group_id = $group_id AND w.engagement_id IN $engagement_ids "
    f"AND w.projection_source_kind = '{_LAW_SOURCE_KIND}' AND w.{LAW_POINTER} IS NOT NULL "
    f"OPTIONAL MATCH (n:{LAW_SECTION_BASE_LABEL}) "
    "WHERE n.group_id = $group_id AND n.engagement_id = w.engagement_id "
    f"AND n.{RECORD_WORKFLOW_PROPERTY} = w.uuid "
    f"AND n.projection_source_kind IN {_LAW_KIND_LIST} "
    f"AND n.projection_source_version_id = w.{LAW_POINTER} "
    f"WITH w, count(n) AS parts, count(DISTINCT n.section) AS sections "
    f"RETURN w.uuid AS law, w.engagement_id AS engagement_id, w.name AS title, w.{LAW_POINTER} AS version_id, "
    "sections, parts "
    "ORDER BY law ASC, engagement_id ASC "
    f"LIMIT {MAX_LAW_SOURCES + 1}"
)


class LawReadFailed(Exception):
    """The read did not answer. Never reported as "not found": a reader must not show a missing section for it."""


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _whole_number(value: Any) -> int | None:
    """An int as stored, or None. A bool is not a number here, and nothing is parsed from text."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


async def _run(graph: Any, name: str, statement: str, params: dict[str, Any]) -> list[list[Any]]:
    """One read. A tenant with no graph has no law: that is an empty answer. Any other failure is LawReadFailed."""
    try:
        result = await asyncio.to_thread(_read, graph, statement, params, LAW_READ_TIMEOUT_MS)
        return list(result.result_set or [])
    except Exception as error:
        try:
            exists = name in (graphiti_client.get_falkor_db().list_graphs() or [])
        except Exception:
            exists = True
        if not exists:
            return []
        logger.warning("[graphiti] law read failed on %s error_type=%s", name, type(error).__name__)
        raise LawReadFailed() from error


def _section_from_rows(
    law: str, section: str, version: str | None, rows: list[list[Any]], engagement_order: list[str]
) -> dict[str, Any]:
    """One item's answer from the rows the read found for it.

    The same law can be loaded under more than one engagement. An item is read from the first engagement, in the
    order the caller named them (the requesting one first), that holds the law, and only from that one: so parts of
    two loads are never mixed, and a section that engagement's law does not have (repealed by a newer version, say)
    is not found, rather than served from another engagement's older load as if it were in force.
    """
    by_engagement: dict[str, list[list[Any]]] = {}
    for row in rows:
        by_engagement.setdefault(_text(row[1]), []).append(row)
    if not by_engagement:
        return {"law": law, "section": section, "version": version, "status": "not_found"}
    engagement = next(e for e in [*engagement_order, *sorted(by_engagement)] if e in by_engagement)
    # A row with no record id is the law with no section of that number at that version.
    chosen = [row for row in by_engagement[engagement] if row[4] is not None]
    if not chosen:
        return {"law": law, "section": section, "version": version, "status": "not_found"}

    def order(row: list[Any]) -> tuple[bool, int, str]:
        part = _whole_number(row[8])
        return (part is None, part if part is not None else 0, _text(row[4]))

    chosen = sorted(chosen, key=order)
    parts = [
        {
            "record_id": _text(row[4]),
            "part": _whole_number(row[8]),
            # Exactly as stored. A value that is not a string is not words a reader can show.
            "text": row[10] if isinstance(row[10], str) else None,
            "source_id": _text(row[11]),
            "content_hash": _text(row[12]),
        }
        for row in chosen
    ]

    # Whole only when the parts are exactly 1..N and every part states the same N, or when the section is one
    # record that states neither its part nor a count (a section kept in one record). Anything else (a part
    # missing, a part twice, counts that disagree) is incomplete: the parts are returned, never passed off as whole.
    declared = {_whole_number(row[9]) for row in chosen}
    numbers = [p["part"] for p in parts]
    count = next(iter(declared)) if len(declared) == 1 else None
    if count is not None and count >= 1:
        expected: int | None = count
        # Compared through the number of parts found, so no list is ever sized by a stored value.
        whole = count == len(numbers) and numbers == list(range(1, len(numbers) + 1))
    elif declared == {None} and numbers == [None]:
        expected, whole = 1, True
    else:
        expected, whole = None, False
    # A part number or count that is stored but is not a whole number (text, a boolean) says nothing that can be
    # checked, so it is not taken as absent either.
    unreadable = any(
        value is not None and _whole_number(value) is None for row in chosen for value in (row[8], row[9])
    )
    # A part with no words (none stored, not a string, or only white space) is not text a reader can show whole.
    # The write path refuses blank text; this holds whatever wrote the record.
    whole = whole and not unreadable and all(p["text"] is not None and p["text"].strip() for p in parts)

    first = chosen[0]
    return {
        "law": law,
        "section": section,
        "version": version,
        "status": "found" if whole else "incomplete",
        "engagement_id": engagement,
        "law_title": _text(first[2]) or _text(first[7]),
        "heading": _text(first[5]),
        "chapter": _text(first[6]),
        "version_id": _text(first[3]),
        "current_version_id": _text(first[13]),
        "in_force": first[13] is not None and first[3] == first[13],
        "parts_expected": expected,
        "parts": parts,
    }


async def fetch_sections(
    client_slug: str, engagement_ids: list[str], items: list[tuple[str, str, str | None]]
) -> tuple[str, list[dict[str, Any]]]:
    """The sections asked for, one answer per item in the order asked: found, incomplete, or not found.

    `items` are (law id, section number, pinned version or None), already trimmed. Raises LawReadFailed when the
    read does not answer.
    """
    name = graphiti_client._graph_name_for_client(client_slug)
    graph = graphiti_client.get_falkor_db().select_graph(name)
    params = {
        "group_id": name,
        "engagement_ids": list(engagement_ids),
        "items": [
            {"index": index, "law": law, "section": section, "version": version}
            for index, (law, section, version) in enumerate(items)
        ],
    }
    rows = await _run(graph, name, SECTIONS_QUERY, params)
    by_item: dict[int, list[list[Any]]] = {}
    for row in rows:
        index = _whole_number(row[0]) if row else None
        if index is not None and 0 <= index < len(items):
            by_item.setdefault(index, []).append(row)
    return name, [
        _section_from_rows(law, section, version, by_item.get(index, []), list(engagement_ids))
        for index, (law, section, version) in enumerate(items)
    ]


async def list_sources(client_slug: str, engagement_ids: list[str]) -> tuple[str, list[dict[str, Any]], bool]:
    """The laws in force for the engagements read, with what of each is stored. Raises LawReadFailed on failure."""
    name = graphiti_client._graph_name_for_client(client_slug)
    graph = graphiti_client.get_falkor_db().select_graph(name)
    rows = await _run(graph, name, SOURCES_QUERY, {"group_id": name, "engagement_ids": list(engagement_ids)})
    sources = [
        {
            "law": _text(row[0]),
            "engagement_id": _text(row[1]),
            "title": _text(row[2]),
            "version_id": _text(row[3]),
            "sections": _whole_number(row[4]) or 0,
            "parts": _whole_number(row[5]) or 0,
        }
        for row in rows
        if row and row[0] is not None
    ]
    return name, sources[:MAX_LAW_SOURCES], len(sources) > MAX_LAW_SOURCES
