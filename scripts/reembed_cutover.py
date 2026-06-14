#!/usr/bin/env python3
"""Driver for the OpenAI -> Voyage embedder cutover.

Loops POST /admin/reembed-graph until the graph is fully migrated, then proves
retrieval works with a real /search/context call. Safe to re-run (the endpoint
is idempotent/convergent). Read-only against content: re-embed only rewrites
vector properties, so node/edge totals MUST be unchanged — this driver asserts
that invariant and aborts loudly if it ever shifts.

Usage:
  python scripts/reembed_cutover.py <base_url> <client_slug> [options]
    --dry-run            Report what would be re-embedded; write nothing.
    --backup PATH        Dump the entity subgraph (content snapshot) to PATH first.
    --query "TEXT"       Verification query for /search/context (default provided).
    --max-items N        Stale items per call (default 1500).
    --max-rounds N       Safety cap on calls (default 60).

Example (staged):
  python scripts/reembed_cutover.py https://sdi-graphiti-service.onrender.com \
      michiganrestaurantlodgingassociation --query "compliance training gaps"
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

CONFIRM = "I understand this overwrites all embeddings"

# Exit-code contract (so an operator/automation can tell apart failure classes):
#   0 success/verified | 1 transient (max_rounds — re-run) | 2 search empty (deploy
#   but retrieval limited) | 3 CRITICAL (content invariant / no progress — do not
#   blindly re-run) | 4 HTTP/infra error (inspect, often a deploy in flight)
EXIT_OK, EXIT_TRANSIENT, EXIT_EMPTY, EXIT_CRITICAL, EXIT_HTTP = 0, 1, 2, 3, 4


def die(msg: str, code: int):
    print(msg, file=sys.stderr)
    sys.exit(code)


def _post(base: str, path: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:500]
        die(f"HTTP {e.code} on {path}: {detail}", EXIT_HTTP)


def backup_content(base: str, slug: str, path: str) -> tuple[int, int]:
    print(f"[backup] dumping entity subgraph for {slug} -> {path}")
    data = _post(base, "/graph/nodes-and-edges", {"client_slug": slug, "max_nodes": 20000})
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"[backup] saved {data['node_count']} nodes / {data['edge_count']} edges")
    return data["node_count"], data["edge_count"]


def reembed_until_done(base: str, slug: str, dry_run: bool, max_items: int, max_rounds: int) -> dict:
    rounds = 0
    total_n = total_e = 0
    totals_lock: tuple[int, int] | None = None  # (nodes_total, edges_total) content invariant
    last = {}
    while True:
        rounds += 1
        if rounds > max_rounds:
            die(f"[abort] exceeded max_rounds={max_rounds}; last={last}", EXIT_TRANSIENT)
        last = _post(
            base,
            "/admin/reembed-graph",
            {"client_slug": slug, "confirm": CONFIRM, "dry_run": dry_run, "max_items": max_items},
        )
        n, e = last["nodes_reembedded"], last["edges_reembedded"]
        total_n += n
        total_e += e
        # Content invariant: the count of embeddable nodes/edges must never change
        # across the run (re-embed touches vectors only, never content).
        cur = (last["nodes_total"], last["edges_total"])
        if totals_lock is None:
            totals_lock = cur
        elif cur != totals_lock:
            die(
                f"[abort] content invariant violated: totals changed {totals_lock} -> {cur} "
                "(re-embed must not add/remove nodes or edges)",
                EXIT_CRITICAL,
            )
        print(
            f"  round {rounds:>2}: +{n} nodes +{e} edges | "
            f"stale n={last['stale_nodes_remaining']} e={last['stale_edges_remaining']} | "
            f"failures={last['failures']} done={last['done']} ({last['elapsed_ms']:.0f}ms)"
        )
        if last["done"]:
            break
        if dry_run:
            print("  [dry-run] stopping after one pass")
            break
        if n + e == 0:
            die(
                f"[abort] no progress this round but not done "
                f"(stale n={last['stale_nodes_remaining']} e={last['stale_edges_remaining']}, "
                f"failures={last['failures']}) — likely persistent write/embed failure",
                EXIT_CRITICAL,
            )
    print(
        f"[reembed] {slug}: re-embedded {total_n} nodes + {total_e} edges in {rounds} round(s); "
        f"marker={last.get('marker')} totals={totals_lock} done={last['done']} dry_run={dry_run}"
    )
    return last


def search_check(base: str, slug: str, query: str) -> bool:
    print(f"[verify] /search/context {slug!r} query={query!r}")
    r = _post(
        base,
        "/search/context",
        {"client_slug": slug, "query": query, "max_results": 5, "include_segment": False},
    )
    facts = r.get("facts", [])
    print(f"[verify] returned {len(facts)} facts in {r.get('search_time_ms', 0):.0f}ms")
    for f in facts[:3]:
        print(f"    - {str(f.get('fact', ''))[:110]}")
    ok = len(facts) > 0
    print("[verify] " + ("PASS — retrieval works with new embeddings" if ok else "EMPTY — investigate before declaring success"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("client_slug")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backup")
    ap.add_argument("--query", default="key risks and compliance gaps")
    ap.add_argument("--max-items", type=int, default=1500)
    ap.add_argument("--max-rounds", type=int, default=60)
    args = ap.parse_args()

    pre = None
    if args.backup:
        pre = backup_content(args.base_url, args.client_slug, args.backup)

    t0 = time.time()
    last = reembed_until_done(
        args.base_url, args.client_slug, args.dry_run, args.max_items, args.max_rounds
    )

    if args.dry_run:
        print(f"[done] dry-run complete in {time.time()-t0:.0f}s")
        return

    if last["failures"]:
        print(f"[warn] {last['failures']} item(s) failed on the final round — re-run to converge")

    ok = search_check(args.base_url, args.client_slug, args.query)

    if pre is not None:
        post = (last["nodes_total"], last["edges_total"])
        print(f"[content] entity totals from endpoint: {post} (backup viz snapshot: {pre[0]} nodes / {pre[1]} edges)")

    print(f"[done] {args.client_slug} cutover {'VERIFIED' if ok else 'NEEDS REVIEW'} in {time.time()-t0:.0f}s")
    sys.exit(EXIT_OK if ok else EXIT_EMPTY)


if __name__ == "__main__":
    main()
