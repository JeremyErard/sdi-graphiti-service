#!/usr/bin/env python3
"""Dormant P1 provenance audit/backfill CLI.

Default mode is metadata-only audit. An apply request requires both ``--apply``
and one exact, positional client slug, then fails closed before database access
until its cardinality guard is separately proven. This script is never imported
by application startup and never selects more than one tenant graph.
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.provenance_ops import (
    APPLY_BLOCKED_CODE,
    ApplyBlockedError,
    ProvenanceAuditReadError,
    run_provenance_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one tenant provenance graph",
        allow_abbrev=False,
    )
    parser.add_argument("client_slug", help="exact [a-z0-9-]+ tenant slug")
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "request unambiguous repairs; currently activation-blocked pending "
            "a disposable Falkor cardinality-guard proof"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_provenance_audit(
            args.client_slug,
            apply=args.apply,
        )
    except ApplyBlockedError:
        result = {
            "mode": "apply",
            "counts": {},
            "codes": {APPLY_BLOCKED_CODE: 1},
        }
        print(json.dumps(result, sort_keys=True))
        return 2
    except ProvenanceAuditReadError as error:
        result = {
            "mode": "apply" if args.apply else "audit",
            "counts": {},
            "codes": {error.code: 1},
        }
        print(json.dumps(result, sort_keys=True))
        return 2
    except Exception as error:
        result = {
            "mode": "apply" if args.apply else "audit",
            "counts": {},
            "codes": {f"AUDIT_FAILED_{type(error).__name__.upper()}": 1},
        }
        print(json.dumps(result, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    apply_conflicts = result.get("counts", {}).get("apply_conflicts", 0)
    unresolved = any(
        "UNRESOLVED" in code and count
        for code, count in result.get("codes", {}).items()
    )
    return 2 if args.apply and (apply_conflicts or unresolved) else 0


if __name__ == "__main__":
    sys.exit(main())
