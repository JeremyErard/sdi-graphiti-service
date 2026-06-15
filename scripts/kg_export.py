#!/usr/bin/env python3
"""Safety-net export of a graph's Entity nodes + RELATES_TO edges to JSONL.

Content only (embeddings omitted — regenerable verbatim via re-embed). Enough to
reconstruct the searchable substrate if a restore is ever needed.

Usage: python scripts/kg_export.py <base_url> <client_slug> <out_dir>
Writes <out_dir>/<graph>__nodes.jsonl and <graph>__edges.jsonl, prints counts.
"""

import json
import sys
import urllib.error
import urllib.request


def _post(base, path, body, timeout=120):
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
        raise SystemExit(f"HTTP {e.code} on {path}: {e.read().decode()[:300]}")


def dump(base, slug, kind, path, limit=500):
    off = total = 0
    written = 0
    graph = ""
    with open(path, "w") as f:
        while True:
            r = _post(base, "/admin/export-graph", {"client_slug": slug, "kind": kind, "offset": off, "limit": limit})
            graph = r["graph_name"]
            total = r["total"]
            for row in r["rows"]:
                f.write(json.dumps(row) + "\n")
                written += 1
            off += r["count"]
            print(f"  {kind}: {written}/{total}")
            if r["done"] or r["count"] == 0:
                break
    if written != total:
        raise SystemExit(f"[abort] {kind} export incomplete: wrote {written} of {total}")
    return graph, total


def main():
    base, slug, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    g, n = dump(base, slug, "nodes", f"{out_dir}/{slug}__nodes.jsonl")
    _, e = dump(base, slug, "edges", f"{out_dir}/{slug}__edges.jsonl")
    print(f"[done] {g}: {n} nodes + {e} edges exported to {out_dir}")


if __name__ == "__main__":
    main()
