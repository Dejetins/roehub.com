# Roehub local-platform information architecture v2

## Status and authority

- Status: `accepted_target_architecture`.
- Scope: authenticated self-hosted Web platform only.
- Product-owner acceptance: `conversation:2026-08-03#g0-product-completeness-package-v2-acceptance`.
- Supersession: replaces v1 by the product-owner decision accepting candidate
  SHA-256
  `16a6d72563166e9811021ae3fd73915b8a38ae993d344b2724d6ab8ed168dc40`.
- Completeness source: `roehub-ui-surface-inventory-v2.md`.
- Machine companions:
  `roehub-ui-surface-registry-v2.json`,
  `roehub-local-platform-screen-registry-v2.json`, and
  `roehub-local-platform-access-and-route-contract-v2.json`.
- Public `roehub.com` and mobile-product information architecture are excluded.

This revision resolves the product-completeness gaps identified after G1
revision `1`. It does not design screen composition, assign families or waves,
select responsive anchors, or authorize implementation.

## Navigation model

| Group | Canonical destinations | Purpose |
|---|---|---|
| Overview | `/dashboard` | Installation and organization summary, readiness, recent work, and guided next actions. |
| Research | `/data`, `/artifacts`, `/strategies`, `/backtests`, `/models` | Data selection and ingestion, reusable evidence, strategies, experiments, backtests, and models. |
| Operations | `/live`, `/jobs`, `/monitoring`, `/connections` | Execution, cross-capability work, service health, safe recovery, and exchange connectivity. |
| System | `/docs`, `/settings/*`, `/admin/*` | Local documentation, personal preferences, organization administration, installation resources, extensions, updates, and recovery. |

The stable application shell retains organization context, command search,
notifications, preferences, locale, theme, and session state. Shell controls do
not grant server capabilities.

## Canonical route structure

### Existing v1 destinations retained

`/login`, `/setup/{step}`, `/dashboard`, `/data`, `/strategies`,
`/strategies/new`, `/strategies/{strategy_id}`, `/backtests`,
`/backtests/new`, `/backtests/{job_id}`, `/models`, `/models/{model_id}`,
`/live`, `/monitoring`, `/connections`, all four `/settings/*`, all five
`/admin/*`, `/docs`, and `/docs/operator/runbooks/{runbook_id}/` retain their
v1 meanings.

### Data ingestion

- `/data` owns catalog discovery and organization selection.
- `/data/ingestion` owns source choice, requested history, current coverage,
  gaps, freshness, backfill/provisioning state, and safe retry eligibility.
- Raw scheduler, worker, ClickHouse, or Redis controls are not exposed as
  user-facing ingestion actions.

### Artifacts

- `/artifacts` is the organization-filtered catalog across supported artifact
  types and consumers.
- `/artifacts/{bundle_id}/{version}` exposes immutable manifest identity,
  provenance, compatibility, storage/capacity state, and downstream use.
- Blob paths, backend secrets, direct GC, and raw storage mutation are not
  product controls.

### Jobs and queues

- `/jobs` provides a bounded cross-capability queue and terminal history.
- `/jobs/{job_id}` owns attempt state, measured progress, queue/execution ETA,
  inputs, outputs, resource admission, cancellation, failure, and safe retry.
- Backtests, models, artifacts, imports, reports, and maintenance retain their
  domain result screens; Jobs does not replace them.

### Live execution and manual position control

- `/live` remains the primary paper/live operations workspace with positions,
  risk, freshness, current activity, and safe-stop context.
- `/live/activity` provides bounded search and filtering for signals, intents,
  orders, fills, rejections, notifications, and reconciliation state.
- `/live/executions/{intent_id}` exposes the durable source-to-intent-to-order-
  to-fill chain and unknown/reconciliation handling.
- `screen.live.manual_position_flow` is a hosted overlay opened from an
  eligible strategy or position. It supports both manual entry and manual exit,
  requires explicit direction and size/close scope, passes the same risk and
  recent-auth gates as automatic execution, never retries blindly, and links
  to the resulting execution identity.

### Monitoring and datastores

- `/monitoring` remains the grouped health overview for platform core, data,
  computation, trading, security, and extensions.
- `/monitoring/{service_id}` is the detail identity for any registered service
  or datastore, including PostgreSQL, ClickHouse, Redis, OpenBao, workers,
  exchange execution, notification delivery, and artifact storage.
- Detail exposes redacted readiness, availability, freshness, dependencies,
  user impact, bounded metrics and backlog, incidents, related jobs, runbooks,
  and role-allowed safe actions.
- Redis, PostgreSQL, ClickHouse, and OpenBao do not receive navigation entries
  or standalone product routes.

## Journey identities

The twelve v1 journey identities are retained. Six identities are added:

1. `journey.data.manage_ingestion_and_coverage`;
2. `journey.artifacts.inspect_and_use`;
3. `journey.jobs.inspect_and_operate`;
4. `journey.live.manual_position_control`;
5. `journey.live.inspect_and_reconcile`;
6. `journey.monitoring.diagnose_service_or_datastore`.

G1 records only these identities and exact source references. Entry,
intermediate, alternate, failure, recovery, and terminal graphs remain G2-owned.

## Classification and count target

The v2 screen registry contains `44` entries: `33` route screens, `1` route
flow, `1` persistent shell, `1` overlay, `1` system-state family, `6`
internal/non-visual entries, and `1` historical exclusion. It reconciles `43`
surface records, `28` retained current route patterns, `8` additive target
route patterns, and `18` journey identities.

## Explicit exclusions

- No public-site screen enters this platform program.
- `mobile_scope` remains `unauthorized`; no mobile screen, bottom navigation,
  mobile drawer, or touch-first information architecture is created.
- Redis and other datastores remain monitoring detail identities, not product
  navigation.
- Historical React, Penpot, Figma, and design-tool visuals provide no active
  topology or visual authority.
- Backtests `v23` remains family-only accepted visual evidence.

## Proof boundary

This candidate proves a deterministic source-level product blueprint once its
machine companions reconcile. It does not prove implementation, browser
behavior, server authorization, persistence, service health, performance,
recovery, accessibility runtime, visual acceptance, release, or deployment.
