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
singleton-conditional mutation query; unit query-shape or syntax tests do not
unlock it. The following activation gates remain **HOLD** even if the current
query text parses:

- stale-evidence protection/CAS for every name, description, anchor, endpoint,
  and episode-list value used to construct a plan;
- endpoint tenant/group binding on both matched endpoint nodes; and
- a proven serialization or transaction/recovery design that cannot report a
  partially applied multi-record plan as complete.

The parser accepts only the exact historical pair emitted by the episode route:

```text
<episode_type>: <source_type>/<source_id>
Engagement <engagement_id> — <episode_type> from <source_type>
```

Both lines must match one another. Near misses, duplicate identities, conflicts,
and unprovable episode lists remain unresolved. Output contains counts and codes,
never graph values. Audit first proves exact `GRAPH.LIST` membership, then uses
only `GRAPH.RO_QUERY`; a missing graph is an error and is never selected or
created. Episode and edge observations each use a 100,001-row sentinel and hard
fail above 100,000 without constructing a partial plan.

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

The opt-in requires one exact `client_slug`, proves that its central mapped graph
name is present in `GRAPH.LIST` before selection, and uses only
`GRAPH.RO_QUERY`. Episode and edge reads each stop at a 100,001-row sentinel;
more than 100,000 rows or more than 256 engagement buckets fails with a fixed
code and returns no partial aggregate. The sentinel is intentionally one bounded
read per record class rather than offset paging: separate Falkor commands do not
provide a shared snapshot, so concurrent graph changes could otherwise cause
page skips/duplicates and break source-aggregate parity. The one-command result
is bounded before any plan/aggregate construction.

## Read-only pinned probes

The harness requires all three inputs and follows no redirects:

```sh
python scripts/provenance_probe.py \
  --manifest <manifest.json> \
  --service-url <exact-https-origin> \
  --auth-secret-env GRAPHITI_SEARCH_SECRET
```

A dedicated service process must set all three of:

```text
GRAPHITI_ACCEPTANCE_PROBE_MODE=true
GRAPHITI_PROVENANCE_MODE=enforce
GRAPHITI_AUTH_MODE=required
```

The harness requires a valid local `GRAPHITI_SEARCH_SECRET` to sign the request.
That local environment value is not proof that the remote process actually
rejects unsigned traffic; remote required-auth/negative-auth evidence remains a
separate rollout check.

The strict `graphiti_p1_probe_manifest_v1` JSON object pins:

- the exact service origin, tenant, engagement, and search auth input;
- one or more non-empty queries with positive result minima and the `fast`
  retrieval path; and
- expected stable fact UUID, episode UUID, source type, source ID, episode type,
  anchor mode, and producer contract version tuples.

The only network operation implemented is signed `POST /search/context` with
`include_segment=false` and `acceptance_probe=true`; both values are inside the
signed body. A probe-mode process rejects a request without that fence, while a
normal process rejects a request with it. In probe mode the service proves exact
graph membership, skips lazy index creation, uses `GRAPH.RO_QUERY` for indexed
retrieval and provenance resolution, and fails closed rather than initializing a
Graphiti client or falling back.

“Read-only probe” is scoped precisely: HMAC replay protection writes the nonce as
security state, and fast retrieval makes a query-embedding provider call. The
claim is no tenant-graph mutation/index creation, no Graphiti client
initialization, and no generative/extraction call. Output contains pass/failure
counts and codes only, and redirects remain rejected.
