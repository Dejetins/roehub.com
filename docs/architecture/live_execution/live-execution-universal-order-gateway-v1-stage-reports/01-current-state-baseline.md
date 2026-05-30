# Stage 01: Current-State Baseline Inventory

Stage 01 records the current Backtest, Strategy, exchange-connection, Redis, and ops surfaces before live-execution implementation starts.

Date: 2026-05-30.

Status: accepted for Stage 01 inventory. The first local-runtime blocker was
superseded by a follow-up readonly SSH check against `macstudio`: API, SQL,
Redis, exchange-control, Monit and most Prometheus targets are available there.
Authenticated browser inventory was completed against `https://roehub.com`.
The `backtest-artifact-publisher` target remains down, but the Mac Studio
runbook excludes that service from automatic launchd/Monit startup; this is
recorded as current ops state, not a Stage 01 live-execution blocker.

## Scope

Stage 01 is inventory-only. No product code, schema, config, API behavior, Redis contract, browser UI, or runtime process was changed.

Included surfaces:

- Backtest API and result/variant surfaces.
- Strategy immutable CRUD, clone, run and stop surfaces.
- Strategy live-runner and Redis stream contracts.
- Exchange connections and strategy exchange bindings.
- Current persistence migrations.
- Native ops and monitoring configuration.
- Explicit missing live-execution components.

Out of scope:

- No mainnet order submission.
- No testnet order submission.
- No `live_execution` implementation.
- No `exchange-execution` process.
- No direct exchange SDK/API access.
- No broad refactor or UI redesign.

## Prerequisite

| Requirement | Evidence | Verdict |
|---|---|---|
| No previous stage is required. | The iteration ledger existed and Stage `01` was `planned` before this update. | Pass. |
| Work on `main`; no stage branch or PR. | `git status --short --branch` reported `## main...origin/main` before edits. | Pass. |
| Runtime evidence is mandatory for acceptance. | Stage prompt and ledger both require API, SQL, Redis, ops, and browser/runtime evidence. | Pass as requirement; Mac Studio runtime evidence was collected after the initial local check. |

## Observed Static Current State

Static inventory is source/doc evidence only. It does not replace real-boundary acceptance.

| Surface | Present today | Evidence |
|---|---|---|
| Backtest API | Authenticated routes exist for runtime defaults, preflight, job create/list/get/cancel/delete, summary, top variants, variant detail, equity, drawdown, stats, trades, lazy trades materialization, and CSV export. | `apps/api/routes/backtests.py`; route decorator inventory. |
| Backtest variant detail | `GET /backtests/jobs/{job_id}/variants/{variant_key}` exists and returns a top-variant DTO. | `apps/api/routes/backtests.py`. |
| Create strategy from backtest variant | Not present as a canonical API/use case. | No route or use-case match for create-from-backtest-variant in the inspected route inventory. |
| Strategy API | Authenticated immutable strategy create, clone, list, get, run, stop, and soft-delete routes exist. | `apps/api/routes/strategies.py`. |
| Strategy run | `POST /strategies/{strategy_id}/run` exists and maps to `RunStrategyUseCase`. | `apps/api/routes/strategies.py`. |
| Strategy stop | `POST /strategies/{strategy_id}/stop` exists and maps to `StopStrategyUseCase`. | `apps/api/routes/strategies.py`. |
| Strategy restart | Not present as a separate user contract. | Route inventory found `run` and `stop`, but no `restart` route. |
| Strategy live-runner | Present as a worker that polls active runs, reads `md.candles.1m.<instrument_key>`, updates `strategy_runs.checkpoint_ts_open`, handles warmup/rollup/repair, and publishes strategy realtime output. | `src/trading/contexts/strategy/application/services/live_runner.py`; `apps/worker/strategy_live_runner/`. |
| Strategy signal journal | Not present as a durable `StrategySignal` or `ExecutionSourceEvent` journal. | No migration/source route for `execution_source_events`; live-runner currently records run progress/events, not source-event execution signals. |
| Strategy realtime Redis output | Present as per-user Redis streams `strategy.metrics.v1.user.<user_id>` and `strategy.events.v1.user.<user_id>`. | `configs/dev/strategy.yaml`; `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md`. |
| Exchange connections | Present via account facade routes for list/create/rotate/disable/archive/validate. | `apps/api/routes/ui_account.py`; `migrations/postgres/0008_exchange_connections_v1.sql`. |
| Strategy exchange bindings | Present via account facade routes for list/create/disable and table `strategy_exchange_bindings`. | `apps/api/routes/ui_account.py`; `migrations/postgres/0010_strategy_exchange_bindings_v1.sql`. |
| Metrics endpoint | API exposes `/health` and `/metrics`; native monitoring config also tracks worker and exchange-control metrics ports. | `apps/api/routes/operations.py`; `infra/macos/prometheus/prometheus.prod.yml`. |
| Monit config | Monit snippets exist for market-data workers, backtest runner/artifact publisher, exchange-control, OpenBao, and Keycloak. | `infra/scripts/monit/*.monitrc`. |

## Persistence Inventory

Static migration inventory shows these current tables or table families:

| Area | Tables present in migrations | Missing for live execution |
|---|---|---|
| Identity | `identity_users`, `identity_sessions`, `identity_exchange_keys`, `identity_user_preferences`, `identity_user_profile_overrides`, `identity_integrations`, `identity_notification_preferences`, `identity_audit_events`. | None for Stage 01; execution work must not reuse identity settings as money ledger. |
| Exchange connections | `exchange_connections`, `exchange_credential_versions`. | Exchange account state projection, positions, open orders, config guard snapshots. |
| Strategy | Strategy storage is implemented through strategy migrations and Postgres adapters; `strategy_exchange_bindings` is present in migration `0010`. | `strategy_live_profiles`, restart operations, strategy signal journal, position ownership, capital reservations, paper accounting. |
| Backtest | Backtest job repositories and artifact-backed result surfaces are present. | Owner-scoped launch provenance from backtest variant to immutable live strategy is not present. |
| Live execution | None. | `execution_source_events`, `execution_intents`, order/fill/funding/reconciliation ledgers, notification outbox, partition/retention/PITR scaffolding. |

Mac Studio SQL runtime inventory confirmed the same core existing tables and
missing execution tables:

| Runtime table | Status |
|---|---|
| `identity_users`, `identity_sessions`, `identity_exchange_keys`, `identity_user_preferences`, `identity_user_profile_overrides`, `identity_integrations`, `identity_notification_preferences`, `identity_audit_events` | present |
| `exchange_connections`, `exchange_credential_versions` | present |
| `strategy_strategies`, `strategy_runs`, `strategy_events`, `strategy_exchange_bindings` | present |
| `strategy_live_profiles`, `execution_source_events`, `execution_intents`, `execution_orders`, `execution_fills`, `execution_reconciliation_runs`, `strategy_position_ownership`, `strategy_capital_reservations`, `paper_execution_orders`, `paper_execution_fills`, `strategy_accounting_snapshots`, `execution_notification_outbox` | missing |

## Required Presence And Absence Matrix

| Required item | Current finding | Acceptance status |
|---|---|---|
| create-from-backtest-variant | Absent. Existing backtest variant detail endpoints are read/result surfaces only. | API/openapi runtime inventory found no create-from-variant route. |
| live profile | Absent. No `strategy_live_profiles` table/API found. | SQL runtime inventory confirmed missing table. |
| restart | Absent as explicit command/API. Existing `run` and `stop` remain separate. | `POST /strategies/{id}/restart` returned `404` on Mac Studio. |
| signal journal | Absent as durable source-event/signal journal. | SQL runtime inventory confirmed `execution_source_events` missing; no signal journal API found. |
| execution source events | Absent. | SQL runtime inventory confirmed `execution_source_events` missing. |
| Redis execution dispatch | Absent. Current Redis stream sample has market-data streams and no execution streams. | Redis scan found `md.candles.1m.*` streams and zero `execution*`, `live_execution*`, or `exchange.execution*` streams. |
| exchange-execution | Absent. No `apps/exchange_execution` package or supervised service config found. | `GET /exchange-execution/health` returned `404`. |
| order adapters | Absent for live execution. | No exchange submit path or execution app found; no exchange submit attempted. |
| reconciliation | Absent for live execution. | SQL runtime inventory confirmed reconciliation tables missing. |
| notifications | Account notification preferences exist; live-execution notification outbox is absent. | SQL runtime inventory confirmed `execution_notification_outbox` missing. |

## Runtime Evidence

The first local endpoint probe was invalid for target acceptance because it
checked this checkout session rather than the Mac Studio runtime. Follow-up
SSH evidence against `macstudio` produced the current target facts below.

| Boundary | Command / check | Result | Classification |
|---|---|---|---|
| API health | `ssh macstudio ... curl http://127.0.0.1:8000/health` | `{"status":"ok"}`. | Pass. |
| API route inventory | Mac Studio `/openapi.json` parsed through Python. | Current routes include backtests jobs/result/variant surfaces, strategy CRUD/clone/run/stop, exchange keys, exchange connections, strategy exchange bindings, UI backtests dashboard and UI strategies dashboard. No restart or execution submit routes found. | Pass. |
| API route status calls | Unauthenticated probes against current and missing routes. | Current authenticated routes returned `401`; `POST /strategies/{id}/restart`, `POST /execution/intents`, `POST /live-execution/intents`, and `GET /exchange-execution/health` returned `404`. | Pass. |
| API metrics | `curl http://127.0.0.1:8000/metrics` | Prometheus exposition available with `http_requests_total` and `http_request_duration_seconds`. | Pass. |
| Exchange-control | `curl http://127.0.0.1:9205/health/ready` and `/metrics`. | Ready with service identity, external exchange validation and Transit cipher checks ready; metrics expose `exchange_control_active 1.0` and exchange-connection counters/gauges. | Pass. |
| SQL metadata | Python `psycopg` metadata query using host env without printing DSNs. | Existing identity/exchange/strategy tables present; live profile and live-execution ledger/accounting tables missing. | Pass. |
| Redis metadata | Python `redis` scan/XINFO using host env without printing credentials. | `md.candles.1m.*` streams exist; sample streams include Binance futures symbols; no `strategy.metrics.v1.user.*`, `strategy.events.v1.user.*`, `execution*`, `live_execution*`, or `exchange.execution*` streams in sample. | Pass with note: strategy realtime streams may be absent because no active run emitted records at sample time. |
| launchd | `launchctl list` filtered for Roehub services. | `com.roehub.api`, `market-data-ws-worker`, `market-data-scheduler`, `backtest-job-runner`, `exchange-control`, `openbao`, `keycloak` listed. | Pass. |
| Monit | `monit summary` filtered for Roehub services. | `roehub_openbao`, market-data workers, `roehub_keycloak`, `roehub_exchange_control`, `roehub_backtest_job_runner` all `OK`. | Pass. |
| Prometheus | `/api/v1/targets` and `up{job=...}` queries. | `exchange-control`, Redis exporter, Postgres exporter, market-data workers, backtest-job-runner and OpenBao are `up=1`; `backtest-artifact-publisher` is `up=0`. | Pass for current-state inventory: records a known excluded service target, not a live-execution blocker. |
| Ports | `nc -z -w 1` on Mac Studio. | `8000`, `9201`, `9202`, `9204`, `9205`, `9090`, `6379`, `5432` open; `8010` and `9203` closed. | Pass for current-state inventory: public Web is served through `https://roehub.com`; artifact publisher metrics are closed because the service is excluded from automatic startup. |
| `backtest-artifact-publisher` classification | launchd, Monit, port, metrics and runbook inventory. | No launchd service, no Monit service, port `9203` closed, Prometheus target `up=0`; `docs/runbooks/mac-studio-native-backend-operations.md` states the service is excluded from automatic reload/deploy bootstrap. | Pass for Stage 01: do not start the service as part of inventory-only acceptance. |

## Browser Inventory

Authenticated browser-visible inventory was completed through Playwright against
`https://roehub.com`:

| Page | Observed current state | Acceptance status |
|---|---|---|
| `/backtests` | Authenticated page loaded with title `Backtests | Roehub`, `data-page="backtests"`, configure/results workstation text, instrument/indicator controls, preflight and run-optimization actions. No launch or restart action was visible. | Pass. |
| `/strategies` | Authenticated page loaded with title `Strategies | Roehub`, `data-page="strategies"`, saved strategy list and `Run`, `Stop`, `Manage` actions. No restart, live profile, signal journal or execution outcome surface was visible. | Pass. |
| `/settings` profile | Authenticated page loaded with title `Settings | Roehub`, `data-page="settings"`, profile, limits and event-log surfaces. | Pass. |
| `/settings#api` | API tab showed connected exchange API inventory with masked key suffix, validation status, capability, market, environment, and actions. No secret values were visible in page text. | Pass. |
| `/settings#integrations` | Integrations tab showed Telegram/Discord/Slack and notification toggles. No live-execution console or order-submit surface was visible. | Pass. |

Static UI facts from the main plan remain the handoff baseline:

- `/backtests` has workstation/results/variant detail surfaces, but no launch action.
- `/strategies` has selected strategy and run/stop surfaces, but no live profile, restart, latest signals, paper accounting, or execution outcome links.
- `/settings` has exchange connections and is not the live-trading console.

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `none` | No API code or DTO was changed. |
| Persistence | `none` | No migration or repository code was changed. |
| Redis | `none` | No stream, consumer group, config, or Redis adapter was changed. |
| Config | `none` | No env/YAML/default changed. |
| Runtime / ops | `none` | No service, launchd, Monit, Prometheus, or runbook config was changed. |
| UI / browser | `none` | No templates, CSS, JS, or browser behavior changed. |
| Docs | `compatible-change` | Added a stage report and updated the stage ledger to record the accepted baseline and handoff facts. |

## Logging And Redaction

No secrets, cookies, Authorization headers, API keys, private keys, signed payloads, exchange provider responses, passphrases, or ciphertext were written to repository files intentionally.

Evidence collected in this report is limited to:

- route names;
- table names;
- service names;
- local port availability;
- authenticated page titles, tab names, masked-key visibility and absence of live-execution controls;
- bounded static file paths.

## Quality Gates

| Gate | Result |
|---|---|
| `uv run ruff check .` | Passed. |
| `uv run pyright` | Passed: `0 errors, 0 warnings, 0 informations`. |
| `uv run pytest -q -ra` | Passed: `998 passed, 3 warnings`. Warnings are existing `httpx` per-request cookie deprecation warnings in web route tests. |
| `python -m tools.docs.generate_docs_index --check` | Passed after docs index regeneration: `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.` |
| `git diff --check` | Passed. |
| Sensitive-value grep over changed stage docs | Passed: no matches for the provided auth credential values or browser-visible user identifiers. Generic policy words such as `secret` and `token` remain in the redaction rules. |

## Rollback

Rollback is documentation-only:

- remove this stage report;
- revert the ledger update;
- regenerate `docs/architecture/README.md` if it was updated.

No data, runtime process, schema, Redis stream, or API behavior needs rollback.

## Next-Stage Handoff

Stage `02` may start only after this accepted Stage `01` report is delivered on
`main` and the direct-main publish/deploy handoff completes.

Follow-up items are not Stage 01 blockers:

- Prometheus still has a `backtest-artifact-publisher` target with `up=0` while the runbook excludes the service from automatic startup. Treat this as ops documentation/monitoring cleanup unless later stages require the publisher.
- Web is not listening on Mac Studio port `8010`; authenticated browser proof was taken from the public Roehub route.
