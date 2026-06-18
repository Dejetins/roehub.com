# Stage 09: Real testnet representative orders

Статус: `blocked`

Дата обновления: `2026-06-19`

## Pre-Start

User required before start: nothing for currently usable active UI-created testnet bindings. Do not send secrets in chat; any missing or broken credential binding must be repaired through `/settings` rotation unless an explicit production data-repair approval is given.

Stage `08` проверен в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`: статус `accepted`, `Next stage allowed = yes`, blockers `none`.

## Scope

Stage `09` должен выполнить representative real testnet order coverage для `BTCUSDT`: Binance/Bybit, spot/futures, supported long/short branches и sizing groups из Stage `03` (`fixed_quote`, `fixed_equity_pct`) через existing source event -> risk -> Redis -> `exchange-execution` -> native adapter -> order/status/fill/reconciliation/outbox path.

## Concrete File List

| File | Action | Reason |
|---|---:|---|
| `src/trading/contexts/exchange_control/adapters/outbound/exchange_account_state.py` | modified earlier in unblock path | Bybit V5 futures account-state reads must pass `symbol` or `settleCoin` for linear order/position reads. |
| `tests/unit/contexts/exchange_control/test_exchange_account_state_reader.py` | modified earlier in unblock path | Regression for Bybit futures account-state scoped params and min-notional parsing. |
| `apps/exchange_execution/adapters/native_http.py` | modified | Binance order endpoints reject scientific decimal strings such as `5E+4`; serialize order params as plain decimals. |
| `tests/unit/apps/exchange_execution/test_native_http_binance_endpoints.py` | modified | Regression for plain decimal serialization on Binance and Bybit order params. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/09-real-testnet-representative-orders.md` | modified | Record runtime order evidence, blockers, cleanup and handoff. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modified | Update Stage `09` ledger state and delivery evidence. |

## Contract Impact

| Dimension | Classification | Note |
|---|---|---|
| Public API contract | `none` | No public request/response schema changed in this stage run. |
| Port contract | `none` | Existing source/risk/dispatch/order ports preserved. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration/table/schema change. |
| Config schema | `none` | No env/config default changed. |
| Request hash / cache / identity | `none` | Idempotency and identity behavior unchanged. |
| Service-call semantics | `compatible-change` | Native adapter order params now use plain decimal strings accepted by exchanges; same values, safer wire format. |
| External side effects | `compatible-change` | Real testnet limit orders were submitted and immediately cancelled under existing testnet-only runtime policy. |
| Logs / metrics / audit / report | `compatible-change` | Adds sanitized runtime evidence. No secrets or raw signed payloads recorded. |
| Browser-visible behavior | `none` | UI not changed in this stage run. |

## Runtime Baseline And Fixes

| Surface | Evidence |
|---|---|
| Stage `08` | `accepted`; `Next stage allowed = yes`. |
| Owner | `smoke_e2e_keycloak`, `owner_user_id=ab094ba2-61d7-4fbf-be8f-cbad9f351572`. |
| Runtime mode | `exchange-execution /health/ready`: `ready/all_dependencies_ready`, `adapter_mode=testnet`. |
| Bybit futures account-state blocker | Fixed in commit `5d50b40a`; Bybit V5 linear reads now include `symbol=BTCUSDT` or fallback `settleCoin=USDT`. |
| Binance futures submit blocker | Fixed in commit `884ed284`; native adapter no longer sends scientific notation for decimal `price`/`quantity`. |
| GitHub delivery | CI `27794497558` and deploys passed for `5d50b40a`; CI `27795342251`, Deploy Backend `27795396215`, Publish App Image `27795396224`, Deploy Web `27795396217` passed for `884ed284`. |
| Mac Studio sync | `/Users/daniildegtyarev/Projects/roehub.com` synced to `884ed284310b2b2c850c8cb7617e9e9c44bf7324`. |
| Runtime sync | `/opt/roehub/app` manually refreshed through the same deploy-backend rsync bundle; runtime contains `_decimal_api_value`; launchd services reloaded. |
| Smoke | `scripts/macos/smoke_prod.sh` passed after reload; exchange-execution ready with pending `0`. |

## Active Connection Inventory

| Exchange | Market | Connection | State | Stage `09` use |
|---|---|---|---|---|
| Binance | spot | `8cec780c-c19c-4781-bd22-2af1d592039d` / `binance_testnet_2` | connection active and UI-ready, but `active_credential_version_id` points to a disabled credential version | `blocked`: execution guard returns `exchange_credential_not_active`. |
| Binance | futures | `18e2de29-27e5-489b-94d3-f681e7e12e2c` / `binance_testnet_2` | active credential, account-state fresh, BTCUSDT isolated `1x`, one-way, qty `0` | used for long and short real testnet orders. |
| Bybit | spot | `af4c90fa-0a3b-4baf-a816-3630322bdf1b` / `bybit_testnet` | active credential, account-state fresh | used for long real testnet orders. |
| Bybit | futures | `6e61bf36-2202-437b-810e-ebe7ed48ba59` / `bybit_testnet` | active credential, account-state fresh, BTCUSDT qty `0`, leverage `10`, margin mode not proven isolated `1x` | used for long real testnet orders; short blocked by config guard. |

Safe fingerprint comparison confirmed Binance spot and futures rows point to the same physical key suffix `RcSh`, but the spot binding's active credential version is disabled. I did not perform ciphertext-copy DB repair because the safe production path remains `/settings` rotate/re-add unless explicitly approved.

## Representative Matrix

All orders used limit prices away from the observed market and `cancel_after_submit=true`. `$50` allocation was rounded only as required by exchange precision/min-notional.

| Exchange | Market | Direction | Sizing group | Result | Evidence |
|---|---|---|---|---|---|
| Binance | spot | long | `fixed_quote` | `blocked` | `stage09-20260618T2317Z`: risk accepted, Redis dispatched, execution guard `exchange_credential_not_active`; no submit. |
| Binance | spot | long | `fixed_equity_pct` | `blocked` | Same guard `exchange_credential_not_active`; no submit. |
| Binance | spot | short | `fixed_quote` | `blocked` | `spot_short_not_supported`; no fake spot short. |
| Binance | spot | short | `fixed_equity_pct` | `blocked` | `spot_short_not_supported`; no fake spot short. |
| Binance | futures | long | `fixed_quote` | `passed` | `stage09-20260618T2327Z`: order `15425253388`, submit `new`, status `new`, cancel `canceled`, fills `0`. |
| Binance | futures | long | `fixed_equity_pct` | `passed` | `stage09-20260618T2327Z`: order `15425257665`, submit `new`, status `new`, cancel `canceled`, fills `0`. |
| Binance | futures | short | `fixed_quote` | `passed` | `stage09-20260618T2327Z`: order `15425258496`, submit `new`, status `new`, cancel `canceled`, fills `0`; isolated `1x` pre-state. |
| Binance | futures | short | `fixed_equity_pct` | `passed` | `stage09-20260618T2327Z`: order `15425258939`, submit `new`, status `new`, cancel `canceled`, fills `0`; isolated `1x` pre-state. |
| Bybit | spot | long | `fixed_quote` | `passed` | `stage09-20260618T2317Z`: order `2240500971791018240`, submit/status/cancel recorded, fills `0`. |
| Bybit | spot | long | `fixed_equity_pct` | `passed` | `stage09-20260618T2317Z`: order `2240500992989030656`, submit/status/cancel recorded, fills `0`. |
| Bybit | spot | short | `fixed_quote` | `blocked` | `spot_short_not_supported`; no fake spot short. |
| Bybit | spot | short | `fixed_equity_pct` | `blocked` | `spot_short_not_supported`; no fake spot short. |
| Bybit | futures | long | `fixed_quote` | `passed` | `stage09-20260618T2317Z`: order `1487c6a6-aae4-4dff-ac75-3dd61c058633`, submit/status/cancel recorded, fills `0`. |
| Bybit | futures | long | `fixed_equity_pct` | `passed` | `stage09-20260618T2317Z`: order `b2d4a228-26cf-4d4f-800a-a292e9ae0e66`, submit/status/cancel recorded, fills `0`. |
| Bybit | futures | short | `fixed_quote` | `blocked` | Account-state fresh but BTCUSDT leverage is `10` and margin mode is not proven isolated `1x`; no auto-config, no submit. |
| Bybit | futures | short | `fixed_equity_pct` | `blocked` | Same futures-short config guard; no auto-config, no submit. |

## Redis, Events, Latency, Cleanup

| Evidence | Result |
|---|---|
| Clean Binance futures run tag | `stage09-20260618T2327Z`, background consumer only. |
| Redis state after clean run | `execution.requests.v1` length `33`, pending `0`, DLQ length `2`. |
| Event sequence | For all clean Binance futures rows: `submit_pending -> private_stream_backfill -> submitted -> status_checked -> cancelled`. |
| Binance futures submit latency | `685.9-690.2 ms` recorded in `execution_orders.latency_ms`. |
| Bybit submit latency | `256.3-272.7 ms` recorded in `execution_orders.latency_ms`. |
| Fills | `execution_fills` count `0` and qty `0` for all Stage `09` real testnet rows. |
| Open orders after cleanup | Binance futures `0`, Bybit spot `0`, Bybit futures `0`. |
| Position after cleanup | Binance futures BTCUSDT qty `0`, isolated `1x`, one-way; Bybit futures BTCUSDT qty `0`, leverage `10`, one-way. |

No mainnet submit was attempted. No raw secrets, API keys, ciphertext, signed payloads, cookies or tokens were written to this report.

## Business Impact

Краткое бизнес-резюме: реальный testnet-контур уже умеет безопасно провести
заявку до биржи, проверить статус, отменить ее и записать результат в ledger
для Binance futures и Bybit spot/futures. Это означает, что основной
операционный путь живой торговли не остается только unit/integration-тестом.
Оставшийся разрыв точечный: Binance spot в UI выглядит готовым, но execution
guard не принимает его из-за disabled credential version. Такой guard правильный
для денег и секретов, но пока не позволяет закрыть spot-long acceptance.

Stage `09` now proves the real testnet execution path for the operationally useful buckets that are safe with the current account configuration: Binance futures long/short, Bybit spot long, and Bybit futures long. This reduces release risk for the supervised strategy producer because order intent, risk acceptance, Redis dispatch, native exchange submit, status check, cancellation, and ledger recording have all crossed the real provider boundary.

The remaining business blocker is narrow and visible: Binance spot appears ready in `/settings`, but execution correctly refuses it because its active credential version is disabled. Until that is rotated or explicitly repaired, spot-long production confidence is incomplete even though futures and Bybit long paths are working.

## Conditional Service-Call Coverage

| Call surface | Coverage |
|---|---|
| Source event / intent / risk | Covered by `ops_test` source events with accepted risk audits for all attempted rows. |
| Redis dispatch | Covered by `execution.requests.v1` message ids and post-run pending `0`. |
| Exchange execution process | Covered through launchd `exchange-execution` background consumer and native adapter events. |
| Native exchange submit | Covered for Binance futures and Bybit spot/futures; Binance spot is explicitly blocked before provider submit. |
| Status / cancel | Covered for all submitted real testnet orders. |
| Fill / funding reconciliation | Fill count is `0` for all rows because limit orders were cancelled before execution; futures funding remains `pending` by design when no funding event exists. |
| Browser UI | Explicit N/A for this stage run; no UI behavior was changed. |
| New outbound provider surface | Explicit N/A; the stage reused existing Binance/Bybit native HTTP adapters and changed only wire-safe decimal formatting/account-state parameters. |

## Logging, Redaction, Monitoring

| Surface | Coverage |
|---|---|
| Redaction | Report and command outputs include only connection ids, order ids, statuses, timings, and aggregate account state. No raw secrets, ciphertext, signatures, cookies, tokens, or provider signed payloads are recorded. |
| Logs | No provider payload logging was enabled. Runtime evidence came from DB ledger rows, health endpoints, and sanitized command output. |
| Metrics | `exchange-execution` readiness/Redis metrics were checked; submit latency is recorded in `execution_orders.latency_ms`. |
| Alerts/runbook | N/A for new alert rules; this stage used the existing Mac Studio deploy/smoke runbook and did not add monitoring configuration. |

## Quality Gates

| Gate | Result |
|---|---|
| `uv run ruff check apps/exchange_execution src/trading/contexts/live_execution src/trading/contexts/exchange_control tests` | passed |
| `uv run pyright apps/exchange_execution src/trading/contexts/live_execution src/trading/contexts/exchange_control tests` | passed (`0 errors`) |
| `uv run pytest -q tests/unit/apps/exchange_execution tests/unit/contexts/live_execution tests/unit/contexts/exchange_control` | passed (`132 passed`) |
| `python -m tools.docs.generate_docs_index --check` | passed after report and ledger update |

## Acceptance Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| Representative matrix has pass/block result for every required bucket | `passed` | Matrix above covers every exchange/market/direction/sizing row. |
| At least one real testnet order path per accepted representative bucket | `passed` | Binance futures long/short, Bybit spot long, Bybit futures long all submitted/status-checked/cancelled through Redis/exchange-execution/native adapters. |
| Redis ack-after-durable, pending, retry, DLQ evidence | `passed` | Clean run left pending `0`; existing DLQ length `2` unchanged. |
| Metrics/latency evidence | `passed` | Submit latency recorded in DB and runtime metrics exposed after orders. |
| No mainnet submit and no secret leakage | `passed` | `adapter_mode=testnet`; report contains only sanitized IDs/statuses. |
| Full Stage `09` unblock | `blocked` | Binance spot active credential version is disabled; Bybit futures short not isolated `1x`. |

## Blockers

| Blocker | Severity | Owner / next action | Acceptance impact |
|---|---|---|---|
| Binance spot binding `8cec780c` is active/UI-ready but its active credential version is `disabled`. | critical | Rotate or re-add `binance_testnet_2` through `/settings` for spot, or explicitly approve a production data repair that copies the active sibling futures ciphertext into a new spot credential version. | Blocks Binance spot long real-order proof. |
| Bybit futures BTCUSDT is not proven isolated `1x` (`leverage=10`, margin mode not isolated). | expected config blocker | User/operator changes Bybit testnet position config manually if short proof is required; platform must not auto-configure it. | Blocks Bybit futures short real-order proof only. |
| Spot short branches are unsupported in v1. | expected product blocker | None for v1 unless a margin/borrow product is explicitly added. | Accepted as blocked behavior. |

## Handoff

Stage `09` is materially improved but remains `blocked` for full closure because one supported bucket, Binance spot long, still lacks an execution-usable active credential. The safe next step is `/settings` rotate/re-add for Binance spot; after that, rerun only the Binance spot long `fixed_quote` and `fixed_equity_pct` rows with the same bounded limit/cancel pattern. Do not place spot shorts, do not auto-configure Bybit futures, and do not perform data repair unless explicitly approved.
