"""Shared test setup.

`graphiti_client` holds ONE FalkorDB handle per process — that is the whole
point of the change that introduced it, since building a pool per request
exhausted FalkorDB's client limit in production on 2026-08-27.

A process-wide handle is shared state, and tests inject their own fake
FalkorDB. Without this reset the first test to touch it pins its fake for the
rest of the session and later tests silently exercise the wrong object.
"""

import pytest

from app.routers import projection
from app.services import graphiti_client


@pytest.fixture(autouse=True)
def _reset_shared_falkordb():
    graphiti_client.reset_falkor_db()
    projection._reset_falkor_db()
    yield
    graphiti_client.reset_falkor_db()
    projection._reset_falkor_db()


# --- Cypher WITH-scope checker, shared by the dedup and search tests ---------
import re as _re

CLAUSE_HEAD = _re.compile(r"\b(WHERE|RETURN|ORDER BY|LIMIT|MATCH|CALL|MERGE|UNWIND|CREATE)\b")
REBINDING = _re.compile(r"\b(MATCH|CALL|MERGE|UNWIND|CREATE)\b")


def _split_top_level(text: str) -> list[str]:
    """Split a projection list on commas outside parentheses."""
    items, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(text[start:i]); start = i + 1
    items.append(text[start:])
    return items


def assert_with_scopes_are_sound(q: str) -> None:
    """Cypher scope rules a fake database cannot enforce (FalkorDB rejected a
    live query for the first one on 2026-09-14, "'n' not defined"):

    - a WITH projection may read only the incoming scope: an alias the WITH
      defines is not in scope inside that WITH, dotted or bare, unless the
      item re-aliases the same incoming name (`node AS node`);
    - the clauses after a WITH, up to the next clause that binds new
      variables (MATCH, CALL, MERGE, UNWIND, CREATE) or the next WITH, may
      dereference only what the WITH projected. Anything after such a
      clause is not checked (a CALL's YIELD fields are not tracked); no
      query in this service has a WITH followed by a CALL.
    """
    for segment in _re.split(r"\bWITH\b", q)[1:]:
        head = CLAUSE_HEAD.split(segment, 1)[0]
        head = _re.sub(r"^\s*DISTINCT\b", "", head)  # `WITH DISTINCT n` projects n
        items = [i.strip() for i in _split_top_level(head)]
        projected: set[str] = set()
        exprs: list[str] = []
        self_aliases: set[str] = set()
        for item in items:
            m = _re.search(r"\bAS (\w+)\s*$", item)
            if m:
                alias, expr = m.group(1), item[: m.start()].strip()
                projected.add(alias)
                exprs.append(expr)
                if expr == alias:
                    self_aliases.add(alias)
            elif _re.fullmatch(r"\w+", item):
                projected.add(item)
                self_aliases.add(item)
        forbidden = projected - self_aliases
        for expr in exprs:
            # Identifiers only: not property names after a dot (`n.uuid AS uuid`)
            # and not function names before a parenthesis (`size(e.episodes) AS size`).
            for ident in _re.findall(r"(?<![.\w])([A-Za-z_]\w*)\b(?!\s*\()", expr):
                assert ident not in forbidden, f"{ident} used inside the WITH that defines it: {head.strip()[:120]}"
        tail = segment[len(head):]
        tail = REBINDING.split(tail, 1)[0]
        for deref in _re.findall(r"\b(\w+)\.", tail):
            assert deref in projected, f"{deref} dereferenced after a WITH that did not project it: {tail.strip()[:120]}"
