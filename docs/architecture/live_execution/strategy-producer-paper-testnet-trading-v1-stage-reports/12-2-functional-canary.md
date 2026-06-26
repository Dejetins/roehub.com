# Stage 12.2: Functional Canary

Статус: `accepted`.

Дата принятого rerun: `2026-06-26`.

## Pre-Start

User required before start: none beyond the provided prompt. The rerun used existing Mac Studio/runtime/browser access, the already configured selected Testnet subject, and no secrets from chat.

Stage `12.1` gate: `accepted` in `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`. The rerun reused the selected Stage `12.1` subject and did not broaden producer allowlists:

| Field | Value |
|---|---|
| owner `user_id` | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| strategy `id` | `ee15e181-309f-478e-8726-04a299f1292f` |
| profile `id` | `5103b2db-5211-4f62-9e0e-a23605de9b41` |
| Stage `12.1` run `id` | `bb5d31ca-3f82-4237-bfb1-25cc9d6bf156` |
| accepted rerun `id` | `d87917a1-1d72-49a8-b5c5-e40290bd3096` |
| connection `id` | `8cec780c-c19c-4781-bd22-2af1d592039d` |
| mode / exchange / market | `testnet` / `binance` / `spot` |
| instrument | `binance:spot:BTCUSDT` |
| strategy spec | `MA(3,5)`, `1m`, `spot`, `BTCUSDT` |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `output/playwright/stage12-2-functional-canary-rerun/*` | none | none | Browser/API evidence artifacts for the accepted rerun. | `none`; evidence artifacts only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-2-functional-canary.md` | none | Replace the previous blocked decision with accepted rerun evidence while retaining the blocker/root-cause explanation. | `compatible-change`; docs/status handoff only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Mark Stage `12.2 accepted`, open Stage `12.3`, and record repair evidence. | `compatible-change`; staged workflow status only. |
| none | `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist` | none | Re-applied scoped host-runtime producer enablement/allowlist after runtime drift showed producer disabled. | `compatible-change`; host runtime config stayed scoped to selected `paper,testnet` subject. |
| `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist.stage12-2-20260626T190736Z.bak` | none | none | Recovery copy before the host-runtime LaunchAgent update. | `none`; host-local backup. |

## Decision

Final decision: `accepted`.

The accepted canary window ran for `32m03s`, from `2026-06-26T19:28:10.541433Z` to `2026-06-26T20:00:13.589485Z`. During that declared window the selected active Testnet strategy produced in-window DB rows and live producer metrics:

| Surface | Baseline | Final | Delta / result |
|---|---:|---:|---:|
| strategy signals | `9` | `41` | `+32` |
| actionable `signal` outcomes | `1` | `8` | `+7` |
| execution source events | `9` | `41` | `+32` |
| source events `recorded` | `1` | `8` | `+7` |
| execution intents | `0` | `0` | `0` |
| execution orders in window | `0` | `0` | `0` |
| mainnet orders in window | `0` | `0` | `0` |
| Redis candle consumer pending | `0` | `0` | no growth |
| Redis candle consumer lag | `0` | `0` | no growth |
| `strategy_live_runner_iterations_total` | `169.0` | `779.0` | `+610` |
| `strategy_live_runner_iteration_errors_total` | `0.0` | `0.0` | no errors |
| `strategy_producer_polled_runs` | `1.0` | `1.0` | selected run polled |
| `strategy_producer_active_instruments` | `1.0` | `1.0` | selected instrument active |

Stage `12.3` may start after this Stage `12.2` handoff is delivered.

## Бизнес-Читаемое Резюме

Первый Stage `12.2` на `2026-06-21` правильно заблокировал продолжение: UI/API показывали активную стратегию, но внутри заявленного окна не появлялись новые signals/source events. Это был именно тот false-positive, который Stage `12.2` должен был поймать до burst/soak.

Rerun `2026-06-26` исправил runtime-состояние и доказал живую работу producer-а: каждую минуту появлялись новые `strategy_signals` и связанные `execution_source_events`, checkpoint двигался по текущим свечам, Redis consumer group не накапливала долг, а mainnet/orders/intents не появились. Это достаточно для перехода к Stage `12.3` burst/resource gate.

## Repair Evidence Before Rerun

The previous blocked state was caused by a combination of runtime drift and stale stream/checkpoint state:

| Finding | Evidence / action |
|---|---|
| Stage `12.1` run was no longer active | Prior accepted run `bb5d31ca-...` had failed before rerun; no active runs were available at the start of repair. |
| Producer was disabled by host runtime drift | `/health/ready` showed producer disabled and empty allowlists. |
| Scoped producer allowlist was re-applied | `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED=true`, `allow_all=false`, modes `paper,testnet`, allowed owner/strategy counts `1/1`. |
| Account projection was stale before restart | API start correctly failed closed with `capital_projection_stale`. |
| Account projection refreshed through existing exchange-control path | Snapshot `7ca08303-2d0b-4ee0-b3a1-38c4c436fc2d`, observed `2026-06-26T19:25:24.980554Z`, status `fresh/account_state_read_ok`. |
| Intermediate run exposed the real blocker | Run `dceeae42-...` stuck in gap repair because checkpoint was far behind and Redis consumer group delivered current messages; `run_once()` spent the first cycle in repair sleep/backoff. |
| Runtime recovery was scoped | Stopped only the selected run, stopped `strategy-live-runner`, destroyed only Redis consumer group `strategy.live_runner.v1` on `md.candles.1m.binance:spot:BTCUSDT`, then restarted the service. Redis stream/candle data and ClickHouse data were not deleted. |
| Accepted rerun was created through the API | Run `d87917a1-1d72-49a8-b5c5-e40290bd3096` reached `running` and advanced from current candles. |

The recovery did not enable mainnet and did not broaden producer scope beyond the selected owner/strategy.

## Canary Window

| Field | Value |
|---|---|
| declared start UTC | `2026-06-26T19:28:10.541433+00:00` |
| declared end UTC | `2026-06-26T20:00:13.589485+00:00` |
| declared start Moscow | `2026-06-26 22:28:10+03` |
| declared end Moscow | `2026-06-26 23:00:13+03` |
| duration | `1923s` / `32m03s` |
| run under observation | `d87917a1-1d72-49a8-b5c5-e40290bd3096` |
| run state at final sample | `running` |
| checkpoint at final sample | `2026-06-26 22:59:00+03` |
| `last_error` | empty |

Final monitor sample:

| Surface | Final sample value |
|---|---:|
| signals total on run | `41` |
| source events total on run | `41` |
| source outcomes | `no_intent/ma_cross_no_change=28`, `recorded/source_event_recorded=8`, `warmup_not_satisfied=4`, `ma_cross_baseline_ready=1` |
| Redis group | `consumers=1`, `pending=0`, `lag=0`, `entries-read=136388` |
| producer metrics | `iterations_total=779.0`, `iteration_errors_total=0.0`, `polled_runs=1.0`, `active_instruments=1.0`, `signal_lag_seconds=0.861814022064209` |

Independent DB/Redis/metrics post-snapshot at `2026-06-26T20:08:21Z` confirmed the in-window rows without relying on terminal monitor state:

| Surface | In-window value |
|---|---:|
| `strategy_signals` rows | `32` |
| actionable signal rows | `7` |
| `execution_source_events` rows joined to selected run | `32` |
| source outcomes | `no_intent/ma_cross_no_change=25`, `recorded/source_event_recorded=7` |
| `execution_intents` rows | `0` |
| `execution_orders` rows | `0` |
| mainnet orders | `0` |
| active runs for selected strategy | `1` |
| Redis group after post-snapshot | `pending=0`, `lag=0`, `entries-read=136396` |
| current run checkpoint after post-snapshot | `2026-06-26 23:07:00+03` |

The post-snapshot also showed the run continued after the declared window (`signals_total=49`, `actionable=10`) with `last_error=""`, which supports handoff continuity for Stage `12.3`.

## Browser and API Evidence

Production browser/API proof was collected against `https://roehub.com/strategies?strategy_id=ee15e181-309f-478e-8726-04a299f1292f` with a temporary authenticated session for the selected owner. The session value was not written to artifacts, logs, the report, or the ledger; the session was revoked after proof and active recent proof sessions returned `0`.

| Artifact | Result |
|---|---|
| `output/playwright/stage12-2-functional-canary-rerun/strategies-prod-rerun-20260626T2000Z.png` | Screenshot captured `/strategies` showing the selected `BTCUSDT 1m [MA(3,5)]` strategy as `live`, producer/run `running`, `testnet`, readiness `ready`, and latest signals after `23:00` Moscow. |
| `output/playwright/stage12-2-functional-canary-rerun/strategies-prod-rerun.snapshot.txt` | Text snapshot includes selected strategy, `running`, `testnet`, `testnet_ready_recent_auth_and_connection`, `binance:spot:BTCUSDT`, and latest signal/source rows through `06/26, 11:20 PM`. |
| `output/playwright/stage12-2-functional-canary-rerun/strategies-prod-rerun.console-network.json` | Dashboard API requests returned `200`; console errors `0`, warnings `0`, failed requests `0`. |
| `output/playwright/stage12-2-functional-canary-rerun/strategies-prod-rerun.api-summary.json` | Manual dashboard API fetch returned `200`; selected strategy id/status/runtime state matched the page. |

## Safety Boundaries

| Boundary | Result |
|---|---|
| Mainnet submit | Passed: no mainnet orders in the declared window. |
| Execution dispatch | Passed: no execution intents/orders in the declared window; current Stage `12.2` acceptance is source-event/signal canary, not real-order proof. |
| Redis backlog | Passed: candle consumer pending and lag stayed `0`. |
| Retry/DLQ growth | No new growth observed for the canary scope. |
| Secrets/redaction | Passed by inspection: no cookies, tokens, DSNs, exchange keys, raw provider payloads, or session values are recorded. |
| Producer scope | Passed: producer remains scoped to `paper,testnet` and selected owner/strategy; `allow_all=false`. |
| Backtest short/funding downstream policy | Not directly exercised: selected subject is historical `long/short` spot Testnet canary evidence; no new short-like launch was created during the rerun. Future new short-like launches still require futures-only policy. |

## Contract Impact

| Dimension | Impact | Notes |
|---|---:|---|
| Public API contract | `none` | Existing strategy run/dashboard APIs were used. |
| Port contract | `none` | No interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or schema change. |
| Config schema/defaults | `none` | No repo config schema/default changed. |
| Host runtime config | `compatible-change` | Re-applied existing scoped producer enablement/allowlist for one selected `paper,testnet` subject. |
| Request hash / cache key / persistence identity | `none` | No identity semantics changed. |
| Service-call semantics | `none` | Existing API, exchange-control, DB, Redis, health, metrics, and browser paths were used. |
| External side effects / unknown-state semantics | `compatible-change` | Refreshed a Testnet account projection, created/stopped repair runs, reset only a Redis consumer group, and started one accepted Testnet strategy run. |
| Logs / metrics / audit / report semantics | `compatible-change` | Stage report and ledger now record accepted rerun evidence. |
| Browser-visible behavior | `compatible-change` | `/strategies` visibly shows the active Testnet strategy and latest in-window/post-window signals. |

## Conditional Service-Call Coverage

This stage touched only runtime verification and recovery paths needed for the accepted rerun. Surfaces outside Stage `12.2` are explicitly marked `N/A`.

| Caller / callee | Purpose | Evidence | Failure behavior |
|---|---|---|---|
| Authenticated API -> Postgres | Create replacement run and read dashboard state for the selected strategy. | Run `d87917a1-...` reached `running`; dashboard API returned `200`. | Initial start was fail-closed on stale capital projection until refreshed. |
| Runtime proof script -> exchange-control | Refresh selected Binance Spot Testnet account projection before API run creation. | Snapshot `7ca08303-...` became `fresh/account_state_read_ok`. | Run creation remains blocked by `capital_projection_stale`. |
| `strategy-live-runner` -> Redis market-data stream | Consume closed `BTCUSDT 1m` candles for the selected active run. | Redis group `strategy.live_runner.v1` stayed `pending=0`, `lag=0`; checkpoint advanced to current bars. | Gap/backlog mismatch can stall in repair/backoff; this was the repaired blocker. |
| `strategy-live-runner` -> Postgres / live_execution port | Persist signals and execution source events. | `+32` in-window `strategy_signals` and `+32` joined `execution_source_events`. | If no in-window rows appear, Stage `12.2` blocks and Stage `12.3` stays closed. |
| Browser/Web -> API | Prove `/strategies` renders selected active Testnet strategy and latest signals. | Production page title `Strategies | Roehub`; dashboard API `200`; console/network clean. | Browser-visible stage remains blocked without this proof. |
| N/A -> exchange order submit | Real order submit is outside Stage `12.2`; Stage `09` owns representative real testnet orders. | `execution_intents=0`, `execution_orders=0`, `mainnet_orders=0` in the canary window. | Any unexpected order/mainnet side effect would block acceptance. |
| N/A -> notification delivery | Notification delivery is outside Stage `12.2`. | N/A | Covered by later notification/runbook stages. |

## Quality Gates

| Gate | Result | Evidence |
|---|---:|---|
| Runtime canary | passed | `32m03s` window produced `+32` signals and `+32` source events. |
| Producer metrics | passed | `iterations_total` advanced from `169.0` to `779.0`; `iteration_errors_total=0.0`; selected run/instrument metrics stayed active. |
| DB evidence | passed | Independent post-snapshot confirmed `32` in-window signal/source-event rows. |
| Redis evidence | passed | Consumer group pending/lag stayed `0`; no stream data was deleted. |
| Mainnet boundary | passed | `mainnet_orders=0` in the declared window. |
| Browser/API proof | passed | Production `/strategies` page title `Strategies | Roehub`, dashboard API `200`, console/network clean, temporary session revoked. |
| `python -m tools.docs.generate_docs_index --check` | passed | `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.` |
| Cold-head artifact review | passed | Cold self-review fallback completed after docs-index check; stale blocked ledger validation row was fixed before handoff. |

## Handoff to Stage 12.3

Stage `12.3` may start from run `d87917a1-1d72-49a8-b5c5-e40290bd3096` if it is still `running` and producer health remains enabled/scoped at start. The next executor must not reinterpret this Stage `12.2` as real-order proof: the canary proves producer polling, source events, signals, Redis/metrics health, and no-mainnet/no-backlog safety. Burst/resource acceptance remains Stage `12.3`.

Before Stage `12.3`, re-check:

| Check | Required result |
|---|---|
| Stage ledger | `12.2 accepted`, `current_stage=12.3`. |
| Active selected strategy | exactly one active run for `ee15e181-309f-478e-8726-04a299f1292f`, or explicitly create/record a replacement before burst. |
| Producer scope | enabled, `allow_all=false`, modes `paper,testnet`, owner/strategy allowlist counts `1/1`. |
| Redis candle group | pending/lag acceptable before burst. |
| Account projection | fresh enough for any API action that reserves capital. |

## Delivery Status

Runtime validation and direct-main delivery are complete. The exact final SHA is recorded in the executor handoff; this report records the delivery surfaces and constraints.

| Surface | Status |
|---|---|
| Local scoped files | delivered: report, ledger, and docs index were committed through direct `main` delivery. |
| GitHub auth | passed after re-check: `gh auth status` authenticated account `Dejetins`; no token value is recorded. |
| GitHub CI/deploy | passed for the final direct-main delivery path; CI, app image publish, backend deploy, and web deploy completed successfully. |
| Mac Studio checkout | synced to `origin/main`; unrelated RL worktree was preserved in a named stash before fast-forward. |
| Runtime code deploy | `N/A` for code impact because no repo runtime code changed; production smoke still passed after delivery. |
| Producer runtime override | re-applied after deploy because the service restarted with producer disabled; final health shows `enabled=true`, `allow_all=false`, modes `paper,testnet`, allowlist counts `1/1`. |

Stage `12.3` may start after a fresh active-run/producer/Redis preflight.

## Cold-Head Review

| Field | Result |
|---|---|
| Cold-head review | completed |
| Mode | cold self-review fallback; no independent subagent was available in the active tool set. |
| Verdict | Release after fixes. Stage `12.2` accepted rerun evidence is coherent across report, ledger status row, blocker table, next prompt, docs index, and browser/API artifacts. |
| Blockers fixed | Added conditional service-call coverage; regenerated and checked `docs/architecture/README.md`; updated stale ledger validation row from blocked to accepted; kept old `2026-06-21` blocked row only as historical evidence. |
| Residual risks | Stage `12.3` must re-check that run `d87917a1-...` is still active and scoped before burst. This stage proves producer signals/source events, not real order execution. Mac Studio unrelated RL changes are preserved in a named stash and are intentionally not part of this Stage `12.2` delivery. |
