# Provenance repair: operating record

## What the repair does

`POST /admin/provenance-audit` plans, and with `apply: true` writes, the
provenance anchor fields (`source_id`, `source_type`, `engagement_id`,
`episode_type`, `anchor_mode`, `producer_contract_version`) onto episodes that
were written before the contract existed, and repairs edge endpoints and
episode lists. Two legacy formats are read: the ingest serializer's
(`<type>: <source>/<id>` with `Engagement <id> — <type> from <source>`) and
the 2026-04-24 batch extraction's (`SOP: … vN` / `Map: …` / `Report: …` with
`Approved artifact (batch-extracted): <kind>:<id>`), the latter anchored under
the v2 contract with an engagement the operator supplies after verifying the
ids in the tenant database. Every write is singleton-conditional (only while
exactly one row carries the uuid and its existing values agree with the plan)
and idempotent.

## The proof (2026-09-11)

Service commit 778ac15 on the deployed FalkorDB. `client_pokagon` (4,284
nodes, 15,637 edges, 7,757 facts) was exported through `/admin/export-graph`
and imported into `scratch_pokagon_provenance` through `/admin/import-graph`
with `group_id` retagged to the scratch name. Batch engagement:
`cmmnn76l60000lqzwaua2zed7` (Phase 1), verified: 98 of 99 SOP ids are
`SOPDocument` rows and the map id a `ProcessVersion` row under that
engagement; the 6 report ids are `InsightReport` rows under it.

| step | result |
|---|---|
| audit (dry run) | 343 episodes, 7,757 edges; 329 anchors planned, 7,757 endpoint updates, 2,716 episode-list updates; 14 already under the v2 contract |
| apply | 8,086 attempted, 8,086 succeeded, 0 conflicts |
| audit again | 0 planned; 329 already anchored, 7,757 endpoints already repaired, 7,757 lists already normalized |
| apply again | 0 attempted |
| graph-stats before | chained 513, pre_chain 7,244 |
| graph-stats after | chained 6,930, pre_chain 827 (legacy engagement-sourced episodes, unchainable under the stats rule of the time; rule aligned with the backend contract in the following change) |
| tenant graph | untouched: chained 513, pre_chain 7,244 |
| scratch copy | deleted through `/admin/delete-graph` |

Runner: `scratchpad/kg-proof.ts` in the operator's session (signed admin calls
only; no graph values leave the service except as counts).

## Operating notes

- Run the audit (dry) first and read the codes; `EPISODE_UNRESOLVED_*`
  counts are the episodes the repair will not touch and why.
- Apply is additive metadata. It never rewrites names, descriptions or fact
  text.
- A scratch copy (`scratch_[a-z0-9_]+`) can be made and removed with the
  export, import and delete routes; graph-stats and the audit accept the
  scratch name so a change can be rehearsed before it touches a tenant graph.
