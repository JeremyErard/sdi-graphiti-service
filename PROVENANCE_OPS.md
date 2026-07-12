# P1 provenance ops companion

These tools are dormant. Nothing here runs at service startup, and this change
does not authorize a live graph operation.

## Audit and backfill

Audit one exact tenant by default (read-only):

```sh
python scripts/provenance_audit.py <client-slug>
```

Mutation is designed to require the explicit flag below and, once separately
unblocked, would apply only cardinality-guarded, unambiguous repairs:

```sh
python scripts/provenance_audit.py <client-slug> --apply
```

In this dormant companion, that command exits with
`APPLY_BLOCKED_CARDINALITY_GUARD_UNVERIFIED` before database access. Activation
requires a separately authorized disposable-Falkor compatibility proof of the
singleton-conditional mutation query; unit query-shape tests do not unlock it.

The parser accepts only the exact historical pair emitted by the episode route:

```text
<episode_type>: <source_type>/<source_id>
Engagement <engagement_id> — <episode_type> from <source_type>
```

Both lines must match one another. Near misses, duplicate identities, conflicts,
and unprovable episode lists remain unresolved. Output contains counts and codes,
never graph values.

## Graph statistics

The existing signed `POST /admin/graph-stats` behavior is unchanged by default.
Provenance queries and additive aggregates require this signed-body opt-in:

```json
{"client_slug":"<client-slug>","include_provenance":true}
```

Aggregates contain structural status, episode type, engagement ID, and counts.
No fact text, names, descriptions, or source content are returned. A fact with
valid sources in more than one dimension may appear once in each dimension; the
structural-status totals remain stable-fact counts.

## Read-only pinned probes

The harness requires all three inputs and follows no redirects:

```sh
python scripts/provenance_probe.py \
  --manifest <manifest.json> \
  --service-url <exact-https-origin> \
  --auth-secret-env GRAPHITI_SEARCH_SECRET
```

`GRAPHITI_AUTH_MODE=required` and a valid `GRAPHITI_SEARCH_SECRET` are mandatory.
The strict `graphiti_p1_probe_manifest_v1` JSON object pins:

- the exact service origin, tenant, engagement, and search auth input;
- one or more non-empty queries with positive result minima;
- expected stable fact UUID, episode UUID, source type, and source ID tuples.

The only network operation implemented is signed `POST /search/context` with
`include_segment=false`. There is no ingest/admin/model/backfill route and no
client-controlled fallback switch. Output contains pass/failure counts and codes
only.
