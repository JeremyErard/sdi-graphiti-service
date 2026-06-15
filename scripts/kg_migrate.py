#!/usr/bin/env python3
"""Full-fidelity KG export / restore for the durability remediation.

export  : dump every node (all labels) + every edge (all types), all props except
          embeddings (regenerated on restore), to JSONL.
restore : recreate a graph from those JSONL files (init indices -> import nodes ->
          import edges -> re-embed), then verify counts + a live search.

Usage:
  python kg_migrate.py export  <base_url> <slug> <dir>
  python kg_migrate.py restore <base_url> <slug> <dir> [--target SLUG] [--query Q]
  python kg_migrate.py verify  <base_url> <slug> <dir> [--target SLUG]
"""
import json
import sys
import urllib.error
import urllib.request

CONFIRM_IMPORT = "import"


def _post(base, path, body, timeout=180):
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
        raise SystemExit(f"HTTP {e.code} on {path}: {e.read().decode()[:400]}")


def _dump(base, slug, kind, path, limit=400):
    off = total = written = 0
    with open(path, "w") as f:
        while True:
            r = _post(base, "/admin/export-graph", {"client_slug": slug, "kind": kind, "offset": off, "limit": limit})
            total = r["total"]
            for row in r["rows"]:
                f.write(json.dumps(row) + "\n")
                written += 1
            off += r["count"]
            print(f"  {kind}: {written}/{total}")
            if r["done"] or r["count"] == 0:
                break
    if written != total:
        raise SystemExit(f"[abort] {kind} incomplete: {written}/{total}")
    return total


def export(base, slug, out_dir):
    n = _dump(base, slug, "all_nodes", f"{out_dir}/{slug}__allnodes.jsonl")
    e = _dump(base, slug, "all_edges", f"{out_dir}/{slug}__alledges.jsonl")
    print(f"[export] {slug}: {n} nodes + {e} edges -> {out_dir}")
    return n, e


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _batched_import(base, target, kind, rows, batch=400):
    done = 0
    imported = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        r = _post(base, "/admin/import-graph", {"client_slug": target, "kind": kind, "rows": chunk, "confirm": CONFIRM_IMPORT})
        imported += r.get("imported", 0)
        done += len(chunk)
        print(f"  import {kind}: {done}/{len(rows)} (created {imported})")
    return imported


def restore(base, slug, in_dir, target=None, query="key risks and compliance gaps"):
    target = target or slug
    nodes = _read_jsonl(f"{in_dir}/{slug}__allnodes.jsonl")
    edges = _read_jsonl(f"{in_dir}/{slug}__alledges.jsonl")
    print(f"[restore] {slug} -> {target}: {len(nodes)} nodes + {len(edges)} edges")

    print("  init-graph (range + fulltext + vector indices)")
    _post(base, "/admin/init-graph", {"client_slug": target})

    n_imp = _batched_import(base, target, "nodes", nodes)
    e_imp = _batched_import(base, target, "edges", edges)

    print("  re-embed (regenerate name/fact embeddings)")
    rounds = 0
    while True:
        rounds += 1
        if rounds > 80:
            raise SystemExit("[abort] re-embed exceeded max rounds")
        r = _post(base, "/admin/reembed-graph", {"client_slug": target, "confirm": "I understand this overwrites all embeddings", "max_items": 2000})
        print(f"    round {rounds}: +{r['nodes_reembedded']}n +{r['edges_reembedded']}e stale n={r['stale_nodes_remaining']} e={r['stale_edges_remaining']} done={r['done']}")
        if r["done"]:
            break
        if r["nodes_reembedded"] + r["edges_reembedded"] == 0:
            raise SystemExit(f"[abort] re-embed no progress (failures={r['failures']})")

    # Verify
    ok = _verify(base, target, len(nodes), len(edges), query)
    print(f"[restore] {target}: {'VERIFIED' if ok else 'NEEDS REVIEW'} (imported {n_imp}n/{e_imp}e)")
    return ok


def _verify(base, target, exp_nodes, exp_edges, query):
    # Entity/RELATES_TO counts via export totals (cheap re-query through export endpoint).
    nr = _post(base, "/admin/export-graph", {"client_slug": target, "kind": "all_nodes", "offset": 0, "limit": 1})
    er = _post(base, "/admin/export-graph", {"client_slug": target, "kind": "all_edges", "offset": 0, "limit": 1})
    print(f"  verify counts: nodes {nr['total']}/{exp_nodes}  edges {er['total']}/{exp_edges}")
    counts_ok = nr["total"] == exp_nodes and er["total"] == exp_edges
    s = _post(base, "/search/context", {"client_slug": target, "engagement_id": "verify", "query": query, "max_results": 5, "include_segment": False})
    facts = s.get("facts", [])
    print(f"  verify search: {len(facts)} facts in {s.get('search_time_ms', 0):.0f}ms")
    for f in facts[:3]:
        print("    -", str(f.get("fact", ""))[:95])
    if not counts_ok:
        print("  [warn] COUNT MISMATCH")
    return counts_ok and len(facts) > 0


def main():
    cmd, base, slug, d = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    target = None
    query = "key risks and compliance gaps"
    for i, a in enumerate(sys.argv):
        if a == "--target":
            target = sys.argv[i + 1]
        if a == "--query":
            query = sys.argv[i + 1]
    if cmd == "export":
        export(base, slug, d)
    elif cmd == "restore":
        ok = restore(base, slug, d, target, query)
        sys.exit(0 if ok else 2)
    elif cmd == "verify":
        t = target or slug
        nodes = _read_jsonl(f"{d}/{slug}__allnodes.jsonl")
        edges = _read_jsonl(f"{d}/{slug}__alledges.jsonl")
        ok = _verify(base, t, len(nodes), len(edges), query)
        sys.exit(0 if ok else 2)
    else:
        raise SystemExit("cmd must be export|restore|verify")


if __name__ == "__main__":
    main()
