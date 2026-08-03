# Roehub UI surface inventory v2

## Status and authority

- Status: `accepted_target_architecture`.
- Scope: authenticated self-hosted Web platform only.
- Product-owner review trigger: `2026-08-03` rejection of G1 revision `1` for
  incomplete product coverage.
- Product-owner acceptance: `conversation:2026-08-03#g0-product-completeness-package-v2-acceptance`.
- Supersedes `roehub-ui-surface-inventory-v1.md` by the product-owner decision
  accepting candidate SHA-256
  `0425f0a7a5f5059971292dc40920a1a0114d0517555fefd28261aaba1812bd6c`.
- Public-site and mobile-product surfaces are excluded. `mobile_scope` remains
  `unauthorized`.

The owner decision requires a product-completeness review across execution,
monitoring, datastore visibility, artifacts, data ingestion, and jobs/queues.
It also requires manual position open and close for a strategy. Redis remains
inside the common monitoring model; it does not receive a route, navigation
item, screen family, or standalone screen identity.

## Source baseline

| Source | Authority | SHA-256 |
|---|---|---|
| `roehub-local-platform-screen-registry-v1.json` | accepted target screen baseline | `0be443dcbd9907792d766aa32f6ba2e77865703e540c739ab2e36abe3477a661` |
| `roehub-ui-surface-registry-v1.json` | accepted surface and route evidence baseline | `973ce8ef42de1ff7a19c5ba8f9f29f21c52b12ef5213bd6c2d9ca96967e0796d` |
| `roehub-ui-surface-inventory-v1.md` | accepted surface inventory baseline | `ef866f556ee660332e56887a74d830ee6288ac4b81e3297143c7122e1fffdc9b` |
| `roehub-local-platform-access-and-route-contract-v1.json` | accepted role, capability, and target-route baseline | `49a0a7adb4c0414135e39f8a442f11bb74ccd68a31ae5ef98a09df3e82580ea2` |
| `roehub-product-transformation-requirements-v1.md` | accepted product requirements | current repository source |
| `live-execution-universal-order-gateway-v1.md` | target execution, order, fill, position, and reconciliation architecture | current repository source |
| `artifact-store-v1.md` | accepted artifact catalog and storage lifecycle contract | current repository source |
| `isolated-job-runtime-v1.md` | accepted general job and attempt lifecycle contract | current repository source |

The accepted Backtests `v23` HTML and receipt remain visual authority only for
the Backtests family. They do not authorize topology, composition, tokens, or
screen identities for the additions in this inventory.

## Completeness method

Each product task is resolved to exactly one primary screen identity and may
name supporting screens. Lifecycle uses the normalized verbs `discover`,
`inspect`, `create`, `configure`, `start`, `observe`, `control`, `recover`,
`archive`, and `consume`. A task without a screen, an explicit non-visual
disposition, or an external boundary is a blocking gap.

Roles use the accepted organization identities `owner`, `admin`, `operator`,
`trader`, and `viewer`. Physical installation actions remain under the separate
`installation_owner` authority overlay.

## Role × task × entity × lifecycle matrix

| Roles | Task | Entity | Lifecycle | Primary screen | Supporting screen / boundary |
|---|---|---|---|---|---|
| anonymous, all authenticated roles | Sign in, recover, and continue safely | session | create, recover | `screen.auth.login` | `screen.system.error_and_recovery` |
| installation_owner | Initialize owner, organization, access, and initial selection | installation | create, configure, recover | `screen.setup.first_launch` | `screen.data.catalog` |
| all authenticated roles | Understand installation and organization status | dashboard projection | discover, inspect | `screen.dashboard.overview` | monitoring and jobs deep links |
| all authenticated roles | Browse exchange, market, provider segment, instrument, history, freshness, and coverage | market-data catalog | discover, inspect | `screen.data.catalog` | `screen.data.ingestion` |
| owner, admin; delegated trader | Change organization instrument and history selection | instrument selection | configure, archive | `screen.data.catalog` | `screen.data.ingestion` |
| all authenticated roles | Observe ingestion, backfill, coverage, gaps, freshness, and source choice | ingestion requirement | inspect, observe | `screen.data.ingestion` | `screen.operations.jobs` |
| owner, admin, operator | Retry an allowlisted failed or stalled ingestion operation | ingestion operation | control, recover | `screen.data.ingestion` | `screen.operations.job_detail` |
| owner, admin; trader for own resources | Create and edit a strategy | strategy | discover, create, configure, archive | `screen.strategies.library`, `screen.strategies.editor` | `screen.strategies.detail` |
| all authenticated roles, server-filtered | Inspect strategy provenance and runtime | strategy | inspect, observe | `screen.strategies.detail` | Backtests and Live links |
| owner, delegated admin, owning trader; operator stop-only | Start, stop, or restart an eligible strategy | strategy run | start, control, recover | `screen.strategies.detail` | `screen.live.workspace` |
| owner, delegated admin, owning trader | Open or close a strategy position manually | manual execution request | configure, start, observe, recover | `screen.live.manual_position_flow` | `screen.strategies.detail`, `screen.live.workspace` |
| all authenticated roles, server-filtered | Observe paper/live positions, orders, fills, PnL, risk, and freshness | execution projection | discover, inspect, observe | `screen.live.workspace` | `screen.live.execution_activity` |
| all authenticated roles, server-filtered | Search and filter signals, intents, orders, fills, rejections, and notifications | execution ledger projection | discover, inspect | `screen.live.execution_activity` | `screen.live.execution_detail` |
| owner, admin, operator; owning trader for non-escalating scope | Inspect an execution chain and reconcile unknown state | execution intent/order/fill/reconciliation | inspect, control, recover | `screen.live.execution_detail` | `screen.docs.runbook` |
| all authenticated roles | Read redacted exchange and integration readiness | exchange connection | discover, inspect | `screen.connections.workspace` | strategy binding context |
| owner, admin; delegated trader binding | Create, validate, bind, rotate, recheck, disconnect, or archive a connection | exchange connection | create, configure, control, archive | `screen.connections.workspace` | recent-auth boundary |
| owner, admin, operator | Inspect general job queues across supported capabilities | job | discover, inspect, observe | `screen.operations.jobs` | domain-specific result screen |
| owner, admin, operator; creator for own job | Inspect attempts, progress, inputs, outputs, failure, cancellation, and retry eligibility | job attempt | inspect, control, recover | `screen.operations.job_detail` | `screen.artifacts.detail` |
| owner, admin, operator; creator within own scope | Cancel or retry an allowlisted job without mutating completed evidence | job attempt | control, recover | `screen.operations.job_detail` | resource admission boundary |
| all authenticated roles, server-filtered | Browse artifacts usable by data, backtests, models, reports, and jobs | artifact bundle | discover, inspect, consume | `screen.artifacts.catalog` | `screen.artifacts.detail` |
| all authenticated roles, server-filtered | Inspect manifest identity, version, provenance, compatibility, storage state, quota impact, and consumers | artifact manifest | inspect, consume | `screen.artifacts.detail` | Backtest, model, or job destination |
| installation_owner, owner/admin read; operator diagnostics | Observe artifact storage health, capacity, GC, and backup state | artifact store | inspect, observe, recover | `screen.monitoring.service_detail` | `screen.admin.resources`, `screen.admin.maintenance` |
| all authenticated roles, server-filtered | Observe grouped platform, data, compute, trading, security, and extension health | service group | discover, observe | `screen.monitoring.workspace` | `screen.monitoring.service_detail` |
| owner, admin, operator; other roles server-filtered | Diagnose a service or datastore, including PostgreSQL, ClickHouse, Redis, OpenBao, workers, and artifact storage | service/datastore projection | inspect, observe, recover | `screen.monitoring.service_detail` | `screen.docs.runbook` |
| owner, admin, operator | Perform only an allowlisted safe service action | service | control, recover | `screen.monitoring.service_detail` | audit and runbook boundary |
| owner, admin; own or server-filtered read for others | Configure and inspect backtests | backtest job/result | create, configure, start, observe, recover, archive | `screen.backtests.library`, `screen.backtests.builder`, `screen.backtests.detail` | artifacts and jobs links |
| all authenticated roles, server-filtered | Inspect models, experiments, readiness, promotion, and rollback | model/experiment | discover, inspect, configure, start, recover | `screen.models.registry`, `screen.models.detail` | jobs and artifacts links |
| all authenticated roles | Manage personal identity, preferences, notifications, sessions, and personal audit | personal settings | inspect, configure, recover | `screen.settings.profile`, `screen.settings.preferences`, `screen.settings.notifications`, `screen.settings.security` | application shell |
| owner, admin, operator read | Inspect organization and installation administration | administration projection | discover, inspect | `screen.admin.overview` | members, resources, extensions, maintenance |
| owner, admin with owner guards | Manage organization membership | membership | create, configure, archive | `screen.admin.members` | recent-auth and audit boundary |
| installation_owner; owner/admin/operator read as filtered | Configure installation ceilings and inspect allocation | resource policy | inspect, configure | `screen.admin.resources` | jobs and artifact capacity links |
| owner, admin | Validate, install, update, or roll back extensions/providers | extension/provider | discover, inspect, create, configure, recover, archive | `screen.admin.extensions` | monitoring detail |
| installation_owner | Backup, restore, update, roll back, and recover the installation | maintenance operation | inspect, start, observe, recover | `screen.admin.maintenance` | `screen.operations.job_detail`, monitoring detail |
| all authenticated roles, server-filtered | Read installed documentation and follow permitted runbooks | local documentation | discover, inspect, recover | `screen.docs.index`, `screen.docs.runbook` | monitoring/execution incident context |

## Gap decisions

| Gap | Resolution in v2 | Consequence |
|---|---|---|
| Manual trading was only a capability/state | Add `screen.live.manual_position_flow` as a hosted overlay flow and `journey.live.manual_position_control`; both manual entry and manual exit are required. | G1 must include the overlay exactly once. |
| Execution detail was compressed into `/live` | Add `/live/activity` and `/live/executions/{intent_id}` for bounded ledger inspection and reconciliation. | Orders, fills, rejections, and unknown outcomes gain stable identities. |
| Service monitoring lacked drilldown | Add `/monitoring/{service_id}`. Datastores are service identities inside this route. | No `/redis`, `/postgres`, or `/clickhouse` product routes are created. |
| Artifact lifecycle was represented only by Backtest modes and budgets | Add `/artifacts` and `/artifacts/{bundle_id}/{version}` as read/consume workspaces. | GC, raw storage operations, and secret-bearing backend configuration remain system/operator concerns. |
| Data ingestion was hidden behind catalog selection | Add `/data/ingestion` for coverage, freshness, backfill, gaps, and safe recovery. | Selection stays in `/data`; raw worker controls are not exposed. |
| General jobs and attempts had no product surface | Add `/jobs` and `/jobs/{job_id}`. | Domain result screens remain canonical; Jobs owns cross-capability execution state. |
| Redis could be mistaken for a product workspace | Keep Redis under `screen.monitoring.service_detail` with redacted health, impact, dependencies, and safe diagnostics. | Redis is never a navigation destination or money source of truth. |

## Reconciliation target

- Screen entries: `44`.
- Route screens: `33`.
- Route flows: `1`.
- Persistent shells: `1`.
- Route-backed transients: `0`.
- Overlays: `1`.
- System-state families: `1`.
- Internal/non-visual: `6`.
- Historical exclusions: `1`.
- Surface records: `43` (`33` retained + `10` added).
- Current observed route patterns: `28` retained without reinterpretation.
- Added target route patterns: `8`.
- Journey identities: `18` (`12` retained + `6` added).

## Remaining review boundary

There are no unresolved identities inside the named completeness gaps. The
entire v2 package remains unresolved as product authority until the owner
accepts the exact hashes of the v2 inventory, surface registry, information
architecture, screen registry, and access/route contract. No G2 family, wave,
representative, responsive anchor, component, visual, browser, or runtime
decision is made here.

## Compatibility and proof boundary

- Product route contract: `compatible-change`; eight additive target routes,
  no removal or reinterpretation of an accepted canonical route.
- Capability contract: `compatible-change` at target-document level; new read
  and safe-operation projections do not broaden existing runtime grants.
- Runtime/API implementation: unchanged and unproven.
- Browser/design proof: not performed.
- Public site: excluded.
- Mobile product: unauthorized.
