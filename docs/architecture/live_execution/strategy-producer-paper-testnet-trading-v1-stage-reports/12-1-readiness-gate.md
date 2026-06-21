# Stage 12.1: Readiness Gate

Статус: `accepted`.

Дата проверки: `2026-06-21`.

## Pre-Start

User required before start: пользователь выбрал Testnet-вариант и подтвердил, что на тестовом аккаунте есть рабочие spot/futures demo keys. Секреты в чат не требовались и не запрашивались.

Stage `11` gate: `accepted` в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`. Ledger row for Stage `11` records direct `main` delivery `c4993cc9`, Mac Studio checkout sync, prod smoke `0`, and host-side controlled load `passed=true`, `violations=[]`.

Старый монолитный Stage `12` остается `superseded` historical negative evidence and is not acceptance for this gate.

Stage `12.1` is a readiness gate only. Functional canary, burst, and 6h soak were not started.

## Concrete File List Before Edits

| Path | Planned action | Reason |
|---|---:|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-1-readiness-gate.md` | update | Replace blocked evidence with accepted Testnet readiness proof. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | update | Mark `12.1 accepted`; open `12.2`. |
| `docs/architecture/README.md` | generated update/check | Required docs index consistency after Markdown changes. |
| `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist` | host-runtime update | Scoped producer enablement and allowlist for the selected Testnet subject. |
| `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist.stage12-1-backup` | host-runtime backup | Recovery copy before changing the LaunchAgent environment variables. |

## Readiness Verdict

Final decision: `accepted`.

The Stage `12.1` blocker was repaired with a scoped Testnet subject:

| Field | Value |
|---|---|
| selected `owner_user_id` | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| selected `strategy_id` | `ee15e181-309f-478e-8726-04a299f1292f` |
| selected `profile_id` | `5103b2db-5211-4f62-9e0e-a23605de9b41` |
| selected `run_id` | `bb5d31ca-3f82-4237-bfb1-25cc9d6bf156` |
| selected mode | `testnet` |
| selected connection | `8cec780c-c19c-4781-bd22-2af1d592039d` |
| selected exchange / market | `binance` / `spot` |
| selected instrument | `binance:spot:BTCUSDT` |
| strategy spec | `MA(3,5)`, `1m`, `spot`, `BTCUSDT` |

Producer runtime is enabled only for the selected scope:

| Setting | Runtime value |
|---|---:|
| `enabled` | `true` |
| `allow_all` | `false` |
| `allowed_modes` | `paper,testnet` |
| `allowed_user_count` | `1` |
| `allowed_strategy_count` | `1` |

## Business Impact

| Layer | Impact |
|---|---|
| Operator confidence | Stage `12.1` no longer observes an idle producer. It has one real, selected Testnet strategy run visible through SQL and the `/strategies` dashboard API. |
| Release risk | Stage `12.2` may start from a concrete Testnet subject instead of inventing scope during the canary. The producer remains `allow_all=false` with one allowed user and one allowed strategy. |
| Money safety | The proof uses Testnet only, `$50` sizing, and `exchange-execution` in `adapter_mode=testnet`; recent mainnet order count stayed `0`. |
| Customer-visible behavior | `/strategies` can now show an active `live`/`running` Testnet strategy state with fresh account projection, rather than only blocked or empty states. |
| Operations | The LaunchAgent override is host-local and backed up, so rollback is bounded to restoring the previous plist and restarting `roehub_strategy_live_runner`. |

## What Was Fixed

The previous blocked report observed two invalid testnet profiles pointing to placeholder connection `00000000-0000-0000-0000-00000000e008`. The accepted Testnet repair did not reuse those profiles.

Instead, the gate used the existing `smoke_e2e_keycloak` owner and active Testnet exchange connections. The selected run was created through the authenticated API path, with a temporary DB session used only for the smoke window and revoked after each call. No session cookie or secret value was printed or written.

Account-state projections were refreshed through exchange-control for:

| Connection | Exchange | Market | Environment | Projection result |
|---|---|---|---|---|
| `8cec780c-c19c-4781-bd22-2af1d592039d` | `binance` | `spot` | `testnet` | `fresh`, `account_state_read_ok`, USDT free `>= 50` |
| `18e2de29-27e5-489b-94d3-f681e7e12e2c` | `binance` | `futures` | `testnet` | `fresh`, `account_state_read_ok`, USDT free `>= 50` |

The runtime also has active `testnet` connections for Binance spot/futures and Bybit spot/futures under the selected owner.

## Runtime Evidence

All commands used host-local env on Mac Studio where needed. No DSN, cookie, password, token, exchange key, raw credential, or signed payload is recorded.

| Surface | Command / query | Result |
|---|---|---|
| Mac Studio source checkout | `git -C /Users/daniildegtyarev/Projects/roehub.com ...` | Runtime checks used Mac Studio paths according to `.codex/AGENTS.md`; repo-code deploy/runtime sync is `N/A` because this stage changed report/index docs and a host-local LaunchAgent override only. |
| Stale Stage `12` collector | `pgrep -fl 'roehub-stage12|stage12.*collector|stage 12.*collector|soak.*collector'` | `stale_collector|none`; nothing stopped. |
| Strategy run SQL counts | `SELECT count(*) FILTER (...) FROM strategy_runs` | `running_strategy_runs=1`, `active_strategy_runs=1`, `running_paper_testnet=1`, `active_paper_testnet=1`. |
| Selected run SQL | Join `strategy_runs` + `strategy_live_profiles` for `bb5d31ca-...` | state `running`, mode `testnet`, profile `ready`, reason `testnet_ready_recent_auth_and_connection`, connection `8cec780c-...`. |
| Testnet connection SQL | `exchange_connections` for selected owner | Active testnet rows exist for `binance/spot`, `binance/futures`, `bybit/spot`, `bybit/futures`; selected row is `binance spot testnet active`. |
| Account projections | `exchange_account_snapshots` | Selected spot projection refreshed at `2026-06-21 18:05:35+03`; dashboard account age `8s`, `status=fresh`, `ready_for_risk=True`. |
| Strategy producer readiness | `curl http://127.0.0.1:9207/health/ready` | `ready=True`, `enabled=True`, `allow_all=False`, `allowed_modes=paper,testnet`, allowlist counts `1/1`. |
| Strategy producer metrics | `curl http://127.0.0.1:9207/metrics` filtered | `strategy_producer_admin_enabled 1.0`, `allow_all 0.0`, `allowed_mode{paper}=1.0`, `allowed_mode{testnet}=1.0`, allowlist entries `user=1.0`, `strategy=1.0`, `ready 1.0`, iteration errors `0.0`. |
| `/strategies` API active state | `GET /ui/strategies/dashboard?strategy_id=ee15e181-...` | `200`; selected strategy status `live`, run state `running`, mode `testnet`, profile `ready`, producer status `running`, run id `bb5d31ca-...`, compatibility `launchable`, market data `ready`, account `fresh`. |
| Direct strategy API | `GET /strategies/{strategy_id}`, `/live-profile`, `/compatibility-readiness` | Strategy `MA(3,5)` returned `200`; live profile returned `testnet/ready`; compatibility returned `launchable`, market data `ready`. |
| Exchange-execution readiness | `curl http://127.0.0.1:9206/health/ready` | `status=ready`, `status_reason=all_dependencies_ready`, `adapter_mode=testnet`. |
| Exchange-execution metrics | `curl http://127.0.0.1:9206/metrics` filtered | Dependencies ready for config, adapter, rate limit, ledger PITR, Redis, backpressure, DLQ, clock drift, Postgres; `exchange_execution_adapter_disabled 0.0`. |
| Mainnet boundary | SQL on `execution_orders` | `mainnet_orders_recent=0` for the proof window; exchange-execution remains `testnet`. |
| Redis transport | `redis-cli XINFO STREAM/GROUPS execution.requests.v1` | stream length `41`; consumer group `exchange-execution.v1` pending `0`; retry stream length `1`; DLQ length `2` unchanged from prior blocked baseline. |
| Monit | `monit summary` | `OK` for strategy live runner, exchange-execution, exchange-control, Keycloak, OpenBao, market-data workers, and backtest job runner. |
| Prometheus scrape targets | PromQL `up{job=~"node-exporter|strategy-producer|exchange-execution|redis-exporter|postgres-exporter"}` | 5 targets returned `1`: node-exporter, Redis exporter, Postgres exporter, exchange-execution, strategy-producer. |
| Resource baseline queries | PromQL CPU/load/memory | CPU busy query returned one node result `10.6546`; `node_load1=2.5713`; node memory query was unavailable in current metric naming and remains a residual telemetry naming gap, not a readiness blocker because process RSS was available. |
| Process RSS | `ps -o pid,ppid,etime,%cpu,rss,command` | strategy-live-runner pid `18311`, RSS `70272 KiB`; exchange-execution pid `13428`, RSS `97376 KiB`. |

Notes:

- `strategy_producer_polled_runs` remained `0.0` during this short readiness proof. Stage `12.2` owns functional canary evidence that producer cycles poll active runs and create source events/signals over 30-60 minutes.
- Stage `12.1` did not start a canary, burst, or 6h soak.

## Conditional Service-Call Coverage

This gate covers only services touched by the Testnet readiness proof. Surfaces not touched by Stage `12.1` are marked `N/A` instead of implied as tested.

| Caller / callee | Purpose | Timeout / retry behavior | Evidence | Failure behavior |
|---|---|---|---|---|
| Authenticated API -> Postgres | Create/read the selected strategy, live profile, and run; read dashboard state. | Existing API transaction behavior; no custom retry added for this gate. | Strategy, live-profile, compatibility, and dashboard calls returned `200`; SQL showed the selected run `running`. | Would block readiness because no selected active subject would exist. |
| Runtime proof script -> exchange-control | Refresh Testnet account projections for Binance spot and futures connections. | Bounded host-local client timeout was raised from the short default to `20s` after the first account-state call timed out; no unbounded retry loop. | Spot and futures projections became `fresh` with `account_state_read_ok` and USDT free `>= 50`. | Would block readiness because `testnet_ready_recent_auth_and_connection` could not be trusted. |
| `strategy-live-runner` -> Postgres / Redis / market data | Load active runs and expose producer health/metrics for the selected scope. | Existing service loop behavior; Stage `12.1` only verifies readiness and active state, not canary-duration polling. | Health ready, allowlist counts `1/1`, SQL active run count `1`, Redis pending `0`. | Stage `12.2` remains blocked until functional polling/source-event evidence passes. |
| `exchange-execution` -> Redis / Postgres / exchange-control / exchange adapters | Verify execution service is ready and remains Testnet-bound for the next gate. | Existing service readiness checks and backpressure/DLQ guards. | Health `ready`, dependencies ready, Redis stream available, adapter mode `testnet`, mainnet recent orders `0`. | Would block canary start or require exchange-execution repair before any Testnet submit proof. |
| Prometheus / node-exporter / Monit -> runtime services | Confirm telemetry and supervision are reachable for the readiness baseline. | Scrape and Monit status only; no alert changes in this stage. | Core scrape targets returned `up=1`, Monit summary `OK`, process RSS captured for producer and exchange-execution. | Missing target would block or become an explicit telemetry residual risk depending on surface. |
| Browser/Web -> API | Prove the `/strategies` surface can report the active Testnet state. | Existing authenticated request behavior; temporary session was revoked after proof. | Dashboard API returned `producer_status=running`, `run_state=running`, `mode=testnet`, account `fresh`. | Would block browser/API proof for Stage `12.1`. |
| N/A -> notification delivery | Telegram/email/operator notification delivery was not part of Stage `12.1`. | N/A | N/A | Covered by later notification/runbook stages, not by this readiness gate. |

## Runtime Config Change

The producer was enabled by adding non-secret `EnvironmentVariables` to the Mac Studio user LaunchAgent:

| Variable | Value |
|---|---|
| `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED` | `true` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOW_ALL` | `false` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOWED_MODES` | `paper,testnet` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOWED_USER_IDS` | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOWED_STRATEGY_IDS` | `ee15e181-309f-478e-8726-04a299f1292f` |

Recovery file: `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist.stage12-1-backup`.

This is a host-runtime override, not a repository default change. The default repo config remains fail-closed.

## Contract Impact

| Dimension | Impact | Notes |
|---|---:|---|
| Public API contract | `none` | No route, payload, or response schema changed. |
| Port contract | `none` | No port/interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or schema change. |
| Config schema/defaults | `none` | No repo config schema/default changed. |
| Host runtime config | `compatible-change` | LaunchAgent now enables producer only for one paper/testnet allowlisted subject. |
| Request hash / cache key / persistence identity | `none` | No identity semantics changed. |
| Service-call semantics | `none` | Used existing API, exchange-control, DB, Redis, health, and metrics contracts. |
| External side effects / unknown-state semantics | `compatible-change` | One bounded Testnet strategy/profile/run and account projections were created; no mainnet path enabled. |
| Logs / metrics / audit / report semantics | `compatible-change` | Stage report and ledger move from blocked to accepted; producer metrics now expose enabled scoped runtime. |
| Alert/runbook semantics | `none` | No alert or runbook changed. |
| Benchmark / rollout gate impact | `compatible-change` | Stage `12.2` may start; Stage `12.1` no longer blocks on empty allowlist/no running run. |
| Performance risk on verified hot path | `none` | No code changed; resource evidence recorded as baseline only. |
| Browser-visible behavior | `compatible-change` | `/strategies` API dashboard now has an active Testnet selected strategy state. |

## Quality Gates

| Gate | Result | Evidence |
|---|---:|---|
| `python -m tools.docs.generate_docs_index --check` | passed | `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.` |
| Secret scan by inspection | passed | Report records only sanitized commands/results and no raw credentials, cookies, tokens, DSNs, exchange keys, or session values. |
| Cold-head artifact review | passed after artifact update | Cold self-review fallback completed; see receipt below. |

## Cold-Head Review

| Field | Result |
|---|---|
| Cold-head review | completed |
| Mode | cold self-review fallback; subagent tools are present, but the available tool policy only permits spawning when the user explicitly requests delegation/subagents. |
| Verdict | artifact handoff is ready as an `accepted` Stage `12.1` report; docs-index check passed, and scoped publish is the remaining executor handoff action. |
| Blockers fixed | Replaced stale blocked verdict; recorded selected Testnet subject; added producer allowlist, API active-state, SQL running-count, account projection, Redis, Monit, Prometheus, exchange-execution, mainnet-boundary, and RSS evidence; documented host runtime LaunchAgent override and recovery file. |
| Residual risks | `strategy_producer_polled_runs` stayed `0.0` in this short readiness window; Stage `12.2` must prove 30-60m functional polling/source-event behavior. Node memory PromQL naming returned no result, but process RSS proof is available. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-1-readiness-gate.md` | none | Accepted Stage `12.1` Testnet readiness report. | `compatible-change`: report/rollout gate evidence. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Mark `12.1 accepted` and open `12.2`. | `compatible-change`: ledger/handoff. |
| none | `docs/architecture/README.md` | none | Generated docs index consistency after report update. | `none`: generated docs index. |
| `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist.stage12-1-backup` | none | none | Host-local recovery copy before producer runtime override. | `none`: backup only. |
| none | `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist` | none | Enable producer with scoped paper/testnet allowlist for selected Testnet subject. | `compatible-change`: host runtime config only, no repo default change. |

## Decision

Stage `12.1` is accepted.

Stage `12.2` may start with the selected Testnet subject:

- owner `ab094ba2-61d7-4fbf-be8f-cbad9f351572`;
- strategy `ee15e181-309f-478e-8726-04a299f1292f`;
- profile `5103b2db-5211-4f62-9e0e-a23605de9b41`;
- run `bb5d31ca-3f82-4237-bfb1-25cc9d6bf156`;
- connection `8cec780c-c19c-4781-bd22-2af1d592039d`;
- mode `testnet`, exchange `binance`, market `spot`, symbol `BTCUSDT`.

Stage `12.2` must prove actual producer polling/source-event behavior over the functional canary window. Stage `12.3` burst and Stage `12.4` 6h soak remain gated until their own prompts.
