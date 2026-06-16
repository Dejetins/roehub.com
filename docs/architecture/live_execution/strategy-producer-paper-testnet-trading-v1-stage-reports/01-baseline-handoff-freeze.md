---
doc: strategy-producer-paper-testnet-trading-v1-stage-01-baseline-handoff-freeze
stage: "01"
status: blocked
plan: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
ledger: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
collected_at: "2026-06-16"
---

# Stage 01: Baseline And Handoff Freeze

Статус: `blocked`.

Runtime baseline для нового цикла `strategy-producer-paper-testnet-trading-v1`
собран и reconciled с accepted Stage `17` foundation. Блокер не в runtime
evidence, а в delivery contract: этот docs-only stage еще не доставлен в
`origin/main`, поэтому stage нельзя помечать `accepted`.

User required before start: nothing.

## Scope

Stage `01` является inventory/evidence-only. Feature code, schema, API
поведение, Redis contracts, Monit/Prometheus config и browser UI не менялись.

Перед редактированием broad expected directory был сужен до конкретного списка:

| Planned file | Action |
|---|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/01-baseline-handoff-freeze.md` | create |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify |
| `docs/architecture/README.md` | modify only if docs index generator requires sync |

Out of scope:

- no paper/testnet strategy launch implementation;
- no exchange submit, cancel, or account mutation;
- no mainnet enablement;
- no new launchd/Monit service;
- no feature flags or runtime config changes.

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `01-baseline-handoff-freeze.md` | - | - | Stage `01` evidence report. | `compatible-change` docs/report semantics only |
| - | `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | - | Record Stage `01` result, validation evidence, blocker, and next-stage handoff. | `compatible-change` docs/ledger semantics only |
| - | `docs/architecture/README.md` | - | Keep architecture docs index synchronized after new Markdown report. | `compatible-change` docs index only |

## Foundation Reconciliation

The new plan starts after accepted
`live-execution-universal-order-gateway-v1` Stage `17`, not as old Stage `18`.
Current foundation facts are:

| Area | Current baseline |
|---|---|
| Old gateway status | Stage `17` is `accepted`; bounded Bybit spot testnet canary is recorded with one filled testnet order, reconciliation, notification, Redis pending `0`, DLQ baseline `2`, and public `/strategies` proof. |
| Source/event path | `execution_source_events`, `execution_intents`, Redis `execution.requests.v1`, `exchange-execution`, order/fill/reconciliation ledgers and notification outbox exist. |
| UI/API path | create-from-variant, live profile/readiness, run/stop/restart, execution source-events/intents, execution notifications, `/ui/backtests/workstation`, and `/ui/strategies/dashboard` routes exist. |
| Strategy producer shape | Code-level `apps/worker/strategy_live_runner` and strategy execution producer port exist; no current Mac Studio launchd/Monit service for `strategy_live_runner` was observed. Stage `06` must make the reused producer supervised. |
| Mainnet boundary | Mainnet submit remains blocked; new cycle scope stays `paper` + `testnet`. |

Drift note: the old gateway Stage `01` current-state baseline is historical and
stale by design. It recorded many live-execution tables/routes as absent before
Stages `02`-`17`; current runtime correctly contains those accepted foundation
surfaces.

## Evidence

All commands were read-only except creating/revoking a disposable browser smoke
session. No credentials, cookies, raw Authorization headers, API keys, signed
payloads, raw provider payloads, or ciphertext were written to this report.

### Git And Host

| Boundary | Command | Evidence summary |
|---|---|---|
| Local git | `git status --short --branch` | `## main...origin/main`; local checkout was clean before docs edits. |
| Mac Studio git | `ssh macstudio 'git -C /Users/daniildegtyarev/Projects/roehub.com status --short --branch; git -C ... rev-parse --short HEAD; git -C ... rev-parse --short origin/main'` | `main...origin/main`; `HEAD=3117fae9`, `origin/main=3117fae9`. |
| Mac Studio smoke | `ssh macstudio 'bash /opt/roehub/app/scripts/macos/smoke_prod.sh'` | Passed: core Homebrew services running, launchd Roehub services listed, API auth probe returned expected unauthenticated `401`, ClickHouse/Redis/Postgres checks passed, Tailscale running. |

### API

| Boundary | Command / call | Evidence summary |
|---|---|---|
| API health | `curl -fsS http://127.0.0.1:8000/health` on `macstudio` | `{"status":"ok"}`. |
| Exchange-control readiness | `curl -fsS http://127.0.0.1:9205/health/ready` | `status=ready`; service identity, external exchange validation, and Transit cipher checks ready. |
| Exchange-execution readiness | `curl -fsS http://127.0.0.1:9206/health/ready` | `status=ready`, `adapter_mode=testnet`, `submit_enabled=1`, `pending_count=0`, `dlq_stream_length=2`, `ledger_pitr=pitr_restore_verified`. |
| Route inventory | `curl /openapi.json | python3 ...` | Routes include `/backtests/jobs/{job_id}/variants/{variant_key}/strategies`, `/strategies/{id}/live-profile`, `/strategies/{id}/restart`, `/ui/execution/source-events`, `/ui/execution/intents`, `/ui/execution/notifications`, `/ui/backtests/workstation`, `/ui/strategies/dashboard`, and settings/account routes. |
| Authenticated UI APIs | Playwright `page.evaluate(fetch(..., credentials="include"))` with disposable session | `200` for `/api/auth/current-user`, `/api/ui/account/profile`, `/api/ui/account/exchange-connections`, `/api/ui/backtests/workstation?refresh=initial`, and `/api/ui/strategies/dashboard?refresh=initial&state=all`. |

### Postgres

| Boundary | Command | Evidence summary |
|---|---|---|
| DB identity/version | `psql ... "select current_database(), current_user, (select version_num from alembic_version limit 1)"` | `roehub|roehub|20260603_0030`. |
| Foundation tables | `information_schema.tables` query for strategy/live-execution tables | Present: `strategy_strategies`, `strategy_runs`, `strategy_live_profiles`, `strategy_signals`, `strategy_variant_compatibility_checks`, `market_data_subscription_requirements`, `exchange_account_snapshots`, `strategy_capital_reservations`, `paper_orders`, `paper_fills`, `strategy_paper_accounting`, `execution_source_events`, `execution_intents`, `execution_orders`, `execution_fills`, `execution_reconciliation_runs`, `execution_notification_outbox`. |
| Ledger counts | SQL count over execution tables | `execution_orders=11`, `execution_fills=2`, `execution_reconciliation_runs=4`, `execution_notification_outbox=9`, `execution_source_events=62`, `execution_intents=56`. |
| Paper counts | SQL count over paper tables | `paper_orders=2`, `paper_fills=2`, `strategy_paper_accounting=2`. |
| Mainnet guard rows | `select environment, status, status_reason, count(*) from execution_orders group by ...` | Mainnet rows are only `guard_rejected/mainnet_hard_block=2`. Testnet rows include adapter errors, guard rejects, cancelled orders, and one `status_checked/filled`. |

### Redis

| Boundary | Command | Evidence summary |
|---|---|---|
| Redis health | `redis-cli -h 127.0.0.1 -p 6379 PING` | `PONG`. |
| Execution streams | `XINFO STREAM execution.requests.v1`, retry and DLQ streams | `execution.requests.v1 length=15`, `execution.requests.retry.v1 length=1`, `execution.requests.dlq.v1 length=2`. |
| Consumer pending | `XPENDING execution.requests.v1 exchange-execution.v1` | `0`. |
| Market data streams | `XINFO STREAM md.candles.1m.binance:spot:BTCUSDT`, `binance:futures:BTCUSDT`, `bybit:spot:BTCUSDT` | BTCUSDT streams exist with lengths around `10085`-`10091`. |
| Strategy realtime streams | `redis-cli --scan --pattern "strategy.*"` | Existing `strategy.events.v1.user.*` and `strategy.metrics.v1.user.*` streams were observed. |

### Monit, Launchd, Prometheus

| Boundary | Command | Evidence summary |
|---|---|---|
| Monit | `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary` | `roehub_openbao`, market-data workers, `roehub_keycloak`, `roehub_exchange_execution`, and `roehub_exchange_control` are `OK`; `roehub_backtest_job_runner` is `Not monitored`. |
| launchd | `launchctl list | grep -E "roehub|strategy|exchange|market-data|backtest"` | API, market-data scheduler/worker, backtest job runner, exchange-control, exchange-execution, OpenBao, Keycloak, exporters and Tailscale services are listed. No `strategy_live_runner` launchd service observed. |
| Strategy runner process | `ps aux | grep -E "strategy_live_runner|apps.worker.strategy"` | No active `strategy_live_runner` process observed. |
| Prometheus targets | Prometheus `/api/v1/query` for `up{job=~...}` | `up=1` for OpenBao, backtest-job-runner, Redis exporter, Postgres exporter, market-data workers, exchange-control, exchange-execution; `backtest-artifact-publisher up=0`. |
| Exchange metrics | Prometheus `/api/v1/query` | `exchange_execution_redis_pending=0`; `exchange_execution_testnet_order_total` returned no current sample in this scrape, while `/health/ready` and DB ledgers prove current exchange-execution readiness/history. |

Ops notes:

- `roehub_backtest_job_runner` is running under launchd and Prometheus `up=1`,
  but Monit currently reports it as `Not monitored`. This is current runtime
  inventory, not a Stage `01` blocker for the new strategy-producer cycle.
- `backtest-artifact-publisher` remains `up=0` in Prometheus, matching the
  known excluded/closed publisher state from the earlier foundation notes.
- `strategy_live_runner` exists in code/config but is not currently a supervised
  Mac Studio service. Stage `06` must either reuse and supervise it or document
  a blocker before introducing a separate producer process.

### Browser

Browser QA used the pinned Playwright CLI wrapper and a disposable local
`roehub_session_id` session for the stage smoke subject. The session was revoked
after capture; active sessions for the subject were `0`.

| Page | Runtime evidence | Screenshot |
|---|---|---|
| `/settings` | Authenticated `Settings | Roehub` page rendered profile, limits, event log, tabs `Profile/API/Integrations/Security`; API calls for current user, profile and exchange connections returned `200`. | `output/playwright/stage01-freeze-settings.png` |
| `/backtests` | Authenticated `Backtests | Roehub` page rendered configure/results workstation, instrument universe, indicators, preflight/run optimization controls; `/api/ui/backtests/workstation...` request returned `200`; console had `0` errors and `0` warnings. | `output/playwright/stage01-freeze-backtests.png` |
| `/strategies` | Authenticated `Strategies | Roehub` page rendered strategy analytics, live profile, account readiness, paper accounting, execution outcomes, saved strategies, latest signals and trades sections; `/api/ui/strategies/dashboard...` returned `200`; console had `0` errors and `0` warnings. The disposable subject had no saved strategies, so dashboard status was degraded/empty with `selected_strategy_not_found`, which is expected for this smoke user. | `output/playwright/stage01-freeze-strategies.png` |

## Mainnet Submit Check

No mainnet submit path is enabled for this cycle.

Evidence:

- deployed `/opt/roehub/app/configs/prod/exchange_execution.yaml` has
  `adapter_mode: "testnet"` and `cancel_after_submit: true`;
- `exchange-execution` readiness reports `adapter_mode=testnet`;
- DB mainnet execution rows are only `guard_rejected/mainnet_hard_block`;
- code-level native adapters still contain `mainnet_hard_block` guard paths;
- the new plan keeps mainnet outside scope until a separate approved plan.

The Mac Studio env contains redacted `ROEHUB_E2E_BYBIT_MAINNET_TRADE_*` keys,
but current runtime evidence shows they do not enable mainnet submit.

## Drift And Blockers

| Item | Classification | Impact | Next action |
|---|---|---|---|
| Old gateway Stage `01` baseline says live-execution tables/routes are absent. | `drift` | Historical doc is stale relative to accepted Stages `02`-`17`; not a runtime blocker. | Treat old Stage `01` as historical only; use Stage `17` report/ledger and this report as current foundation. |
| `roehub_backtest_job_runner` is `Not monitored` in Monit while launchd and Prometheus show it running/up. | `drift` | Monitoring-state mismatch for backtest runtime. Not a blocker for freezing strategy-producer foundation, but should be cleaned before relying on Monit for runner SLOs. | Ops follow-up or a later backtest/ops stage. |
| `strategy_live_runner` is not a current launchd/Monit service. | `expected gap` | The code path exists but the future strategy producer is not yet supervised. | Stage `06` must make the reused producer supervised or document a blocker before creating a new process. |
| Stage docs are not delivered to `origin/main`. | `blocker` | Prompt forbids `accepted` without main-branch delivery evidence. | Publish/merge this docs stage and record main SHA; docs-only runtime sync can be `N/A` with reason. |

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API route, status, error, or payload changed. |
| Port contract | `none` | No protocol/interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or table semantics changed. |
| Config schema/defaults | `none` | No env/YAML/default changed. |
| Request hash / cache key / persistence identity | `none` | No identity, hash, or cache semantics changed. |
| Service-call auth/timeout/retry/error semantics | `none` | Runtime calls were read-only evidence; no service behavior changed. |
| External side-effect/idempotency/unknown-state semantics | `none` | No exchange submit/cancel/account mutation was performed. |
| Logs/metrics/traces/audit/ledger/report semantics | `compatible-change` | Adds sanitized docs evidence and ledger handoff only. |
| Alert/runbook semantics | `none` | No Monit/Prometheus/runbook config changed. |
| Benchmark/rollout gate impact | `none` | No benchmark threshold or gate changed. |
| Performance risk on verified hot path | `none` | No runtime code changed. |
| Browser-visible behavior | `none` | Browser was observed only; UI files were not changed. |

## Quality Gates

| Gate | Result |
|---|---|
| `python -m tools.docs.generate_docs_index --check` | Passed after regenerating `docs/architecture/README.md`: docs index is up to date. |

Tests-only acceptance was not used. Stage `01` used target runtime and browser
runtime evidence.

## Publish / Deploy

| Item | Status |
|---|---|
| GitHub publish | Not completed in this report yet. |
| Main branch delivery | Missing; `delivery_pending_main_host_sync`. |
| Runtime sync | `N/A` for docs-only stage after main delivery; no runtime/code changes require `/opt/roehub/app` update. |
| Stage status | `blocked` until main delivery evidence is recorded. |

## Next-Stage Handoff

Stage `02` must not start from this report until delivery is unblocked and the
ledger row changes from `blocked` to `accepted`.

When unblocked, Stage `02` starts from these facts:

- use accepted gateway Stage `17` foundation as the money-boundary baseline;
- do not redesign `exchange-execution`;
- keep mainnet out of scope;
- use `/backtests` -> create-from-variant -> `/strategies` APIs/UI as current
  launch foundation;
- expect empty/degraded strategy dashboard states for users without saved
  strategies and handle them as valid read-model states;
- Stage `06` owns the supervised `strategy_live_runner` producer service gap.
