# Graphiti service-auth rollout

This change is intentionally deployable without interrupting current Engage
traffic. It does not mutate graph data, route bodies, or successful response
contracts. It also closes the legacy tenant-name collision. The two populated
hyphenated tenants discovered in the July 9 live inventory retain their exact
legacy graph names through an explicit compatibility registry; other future
hyphenated slugs use a lossless underscore encoding.

## Pre-deployment graph-name gate

Re-run the live client/graph inventory immediately before deployment and compare
it with `LEGACY_TENANT_GRAPH_NAMES` in `app/graph_names.py`. The July 9 inventory
contains `michigan-restaurant-lodging-association` and `test-provision`; both are
registered and preserve their populated graph names. If any additional live
hyphenated slug has legacy data, stop and either add its exact compatibility
mapping or migrate it with the export/import tooling before promotion. Never let
the service create an empty replacement graph for an existing tenant.

Freeze all new client provisioning from the first Graphiti deployment until
both services have remained in `required` mode through two clean verification
cycles and that build becomes the new rollback baseline. A tenant created only
in the v2 namespace would not be readable by the pre-v2 mapper during a code
rollback. Existing clients and engagements continue normally during this short
provisioning freeze.

## Credentials

Create three independent random secrets of at least 32 characters and store them
only in the deployment secret managers for both services:

- `GRAPHITI_SEARCH_SECRET`
- `GRAPHITI_INGEST_SECRET`
- `GRAPHITI_ADMIN_SECRET`

Do not reuse `JWT_SECRET`, database credentials, or provider API keys.

## Promotion sequence

1. Confirm new client provisioning is frozen, then deploy the Graphiti service code with all three secrets and
   `GRAPHITI_AUTH_MODE=optional`.
2. Confirm `/health` and `/ready` remain available and existing unsigned calls
   still work.
3. Deploy the Engage backend code with the same three secrets and
   `GRAPHITI_AUTH_MODE=required`. This makes every backend graph call signed.
4. Verify successful search, graph visualization, ingestion, and admin-init
   calls. Confirm Graphiti logs no signature, tenant-claim, or scope denials for
   valid Engage traffic.
5. Change the Graphiti service to `GRAPHITI_AUTH_MODE=required` without changing
   the credentials.
6. Prove the perimeter:
   - unsigned `/search`, `/graph`, `/ingest`, and `/admin` requests return 401;
   - a search credential cannot call ingest/admin routes;
   - a signed tenant claim cannot be paired with another `client_slug`;
   - expired or body-modified requests return 401;
   - an identical signed request replay returns 409;
   - `/health` and `/ready` remain public for platform health checks.

Production is not considered secured until step 5 is complete.

Signed requests use the v2 contract and include a cryptographically random
nonce. Nonces are atomically consumed in FalkorDB's Redis layer; if that replay
store is unavailable, signed traffic fails closed with 503 rather than accepting
a potentially replayed write.

## Rollback

If signed traffic fails before step 5, leave Graphiti in `optional` mode while
the backend signer is corrected or rolled back. If a problem appears after step
5, return Graphiti to `optional` before rolling back the backend. Do not use
`off` in production; `optional` is the bounded compatibility state.

Do not lift the provisioning freeze while a pre-v2 Graphiti image remains the
rollback target. After two clean verification cycles, retain the v2 tenant
resolver in every forward and rollback artifact before provisioning resumes.

No rollback step mutates FalkorDB or client databases.

## Rotation

The current contract accepts one credential per scope. Rotate during a short
coordinated window: put Graphiti in `optional`, update the credential on both
services, verify signed traffic, then restore `required`. Dual-key overlap can
be added later if rotation without any compatibility window becomes necessary.
