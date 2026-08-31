#!/usr/bin/env python3
"""Report what vector indexes actually exist in FalkorDB, per graph.

WHY. Production logs show the Entity.name_embedding index being CREATED
repeatedly for the same graph -- client_thrive on 2026-08-30 at 12:56, 14:42 and
16:34 -- and never once reporting "already exists". Both ensure paths log
"created" only when the CREATE raised nothing, and log "not created" when it
did. Across seven days every line says created. An index that persisted would
have collided with itself on the second attempt.

At the same time, `db.idx.vector.queryNodes('Entity', 'name_embedding', ...)`
fails with "Invalid arguments for procedure" in the SAME SECOND as a successful
create, and the search falls back to graphiti's O(N) cosine scan -- the scan the
index exists to remove.

The unit tests cannot see any of this. They assert on the Cypher we EMIT against
a fake executor, so they pass whether or not FalkorDB accepts it. Code and test
share the assumption; only the real server can refute it.

So this asks FalkorDB directly, and it is a READ: `db.indexes()` per graph, plus
one bounded probe query whose failure is caught and reported. Nothing is
created, altered or deleted.

    render jobs create <graphiti-serviceId> --confirm \
      --start-command "python scripts/check_vector_indexes.py"
"""
from __future__ import annotations

import sys
from typing import Any


def summarize_indexes(rows: list[Any]) -> list[str]:
    """Render `CALL db.indexes()` rows as one readable line each.

    Kept pure so the shape handling is testable without a FalkorDB: the driver
    returns positional rows whose arity has changed across versions, which is
    itself a candidate explanation for the failure this script investigates.
    """
    out: list[str] = []
    for row in rows or []:
        if isinstance(row, dict):
            label = row.get("label") or row.get("entityType") or "?"
            props = row.get("properties") or row.get("fields") or "?"
            types = row.get("types") or row.get("indexType") or "?"
            out.append(f"label={label} fields={props} types={types}")
        elif isinstance(row, (list, tuple)):
            out.append(" | ".join(str(cell) for cell in row))
        else:
            out.append(str(row))
    return out


def classify_probe(error: str | None) -> str:
    """Turn the probe outcome into the distinction that decides the fix.

    "Invalid arguments" means the procedure exists but rejects this call -- a
    signature or version mismatch, and the query has to change. Anything naming
    the index means the index is genuinely absent, and the CREATE is the thing
    that is not working. Conflating the two sends the fix to the wrong file.
    """
    if error is None:
        return "OK — the index answered a query, so it exists and is usable"
    lowered = error.lower()
    if "invalid arguments" in lowered:
        return "SIGNATURE — procedure exists but rejects this call; the QUERY is wrong for this build"
    if "unknown procedure" in lowered or "unknown function" in lowered:
        return "UNSUPPORTED — this FalkorDB build has no vector query procedure at all"
    if "index" in lowered and ("not" in lowered or "no such" in lowered):
        return "ABSENT — no such index; the CREATE is what is not working"
    return f"OTHER — {error}"


def main() -> int:
    from app.config import settings
    from app.services.graphiti_client import get_falkor_db

    db = get_falkor_db()
    try:
        graphs = db.list_graphs()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"could not list graphs: {exc}")
        return 1

    print(f"\n######## VECTOR INDEX STATE ({len(graphs)} graph(s)) ########")
    dim = int(settings.embedding_dim)
    for name in sorted(graphs):
        print(f"\n=== {name} ===")
        graph = db.select_graph(name)
        try:
            result = graph.query("CALL db.indexes()")
            for line in summarize_indexes(getattr(result, "result_set", []) or []):
                print(f"  {line}")
        except Exception as exc:  # noqa: BLE001
            print(f"  db.indexes() failed: {exc}")

        # Bounded probe: k=1 against a zero vector. Reads only, and its ERROR is
        # the actual diagnostic — a result set is not what we are after.
        error: str | None = None
        try:
            graph.query(
                "CALL db.idx.vector.queryNodes('Entity', 'name_embedding', 1, vecf32($v)) "
                "YIELD node RETURN count(node)",
                params={"v": [0.0] * dim},
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        print(f"  probe: {classify_probe(error)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
