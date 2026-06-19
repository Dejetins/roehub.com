# Stage 09: Real testnet representative orders

Статус: `accepted`

Дата обновления: `2026-06-19`

## Pre-Start

User required before start: nothing remains for Stage `09` acceptance. Active UI-created testnet bindings exist for Binance and Bybit spot/futures. Do not send secrets in chat; any future credential repair must go through `/settings` rotation/re-check unless an explicit production data-repair approval is given.

Stage `08` проверен в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`: статус `accepted`, `Next stage allowed = yes`, blockers `none`.

## Scope

Stage `09` proves representative real testnet order coverage for `BTCUSDT`: Binance/Bybit, spot/futures, supported long/short branches and sizing groups from Stage `03` (`fixed_quote`, `fixed_equity_pct`) through existing source event -> risk -> Redis -> `exchange-execution` -> native adapter -> order/status/fill/reconciliation/outbox path.

Spot short remains an accepted unsupported v1 branch (`spot_short_not_supported`) because no margin/borrow spot product exists. It is recorded as product-blocked, not as a Stage `09` failure.

## Concrete File List

| File | Action | Reason |
|---|---:|---|
| `src/trading/contexts/exchange_control/adapters/outbound/exchange_account_state.py` | modified | Bybit V5 futures reads pass required linear scope params; Binance/Bybit futures account-config commands set and read back isolated `1x` without order placement. |
| `src/trading/contexts/exchange_control/application/account_state.py` | modified | Adds explicit account-config command and guards. |
| `src/trading/contexts/exchange_control/adapters/inbound/http/app.py` | modified | Adds internal account-config endpoint for active owned testnet futures bindings. |
| `apps/api/routes/ui_account.py` / `apps/api/dto/ui_account.py` / `apps/api/exchange_control_client.py` | modified | Adds compatible UI/API account-config route and DTOs. |
| `apps/web/dist/js/pages/settings.js` / `apps/web/locales/en.json` / `apps/web/locales/ru.json` | modified | Adds market-level readiness display and `Iso 1x` operator action for testnet futures rows. |
| `apps/exchange_execution/adapters/native_http.py` | modified earlier | Binance order endpoints reject scientific decimal strings; serialize order params as plain decimals. Binance Spot Demo REST listenKey HTTP `410` is now degraded preflight evidence, not a submit blocker. |
| `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/05-safe-testnet-exchange-binding.md` | modified | Prompt artifact now allows explicit platform-managed testnet futures account config. |
| `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/09-real-testnet-representative-orders.md` | modified | Prompt artifact now requires account-config proof before real futures short acceptance. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md` | modified | Architecture plan updated from manual-only isolated `1x` to explicit safe platform-managed testnet account config. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/09-real-testnet-representative-orders.md` | modified | Records final runtime order evidence, cleanup, contract impact, and acceptance. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modified | Updates Stage `09` ledger state and handoff. |

## Contract Impact

| Dimension | Classification | Note |
|---|---|---|
| Public API contract | `compatible-change` | Adds `POST /ui/account/exchange-connections/{connection_id}/account-config`; existing routes and DTO fields remain valid. |
| Internal service contract | `compatible-change` | Adds internal exchange-control account-config command for active owned testnet futures connections. |
| DTO schema | `compatible-change` | Adds account-config request/result DTOs; existing payloads are unchanged. |
| Persisted schema | `none` | No migration/table/schema change. Account-state evidence continues through existing DB/read-model surfaces. |
| Config schema | `none` | No env/config default changed. |
| Request hash / cache / identity | `none` | Idempotency, launch identity, strategy identity, and owner ownership checks are unchanged. |
| Service-call semantics | `compatible-change` | Binance/Bybit testnet futures margin/leverage can now be explicitly configured by operator action; order submit never performs hidden account reconfiguration. |
| External side effects | `compatible-change` | Testnet-only futures account settings were set/read back; real testnet limit orders were submitted and immediately cancelled. No mainnet submit. |
| Logs / metrics / audit / report | `compatible-change` | Adds sanitized account-config/order evidence. No secrets, cookies, signed payloads, ciphertext, or raw provider credential material recorded. |
| Browser-visible behavior | `compatible-change` | `/settings` now shows spot/futures readiness separately and exposes `Iso 1x` for eligible testnet futures bindings. |

## Runtime Baseline And Fixes

| Surface | Evidence |
|---|---|
| Stage `08` | `accepted`; `Next stage allowed = yes`. |
| Owner | `smoke_e2e_keycloak`, `owner_user_id=ab094ba2-61d7-4fbf-be8f-cbad9f351572`. |
| Runtime mode | `exchange-execution /health/ready`: `ready/all_dependencies_ready`, `adapter_mode=testnet`. |
| Bybit V5 account config docs | Implementation follows Bybit V5 `set-margin-mode`, `set-leverage`, and account-info read-back; `position.tradeMode` is not trusted because Bybit marks it deprecated for UTA. |
| Binance USD-M account config docs | Implementation follows Binance USD-M Futures `marginType`, `leverage`, and `positionRisk` read-back; already-isolated response is accepted as idempotent success. |
| Bybit futures account-state blocker | Fixed in commit `5d50b40a`; Bybit V5 linear reads include `symbol=BTCUSDT` or fallback `settleCoin=USDT`. |
| Binance futures submit blocker | Fixed in commit `884ed284`; native adapter no longer sends scientific notation for decimal `price`/`quantity`. |
| Binance spot REST listenKey blocker | Fixed in commit `2ba8b167`; Binance spot private-stream preflight records `binance_spot_rest_user_stream_deprecated` and continues supported Demo Spot REST submit/status/cancel. |
| Testnet account-config controls | Delivered in commit `e093ce2e`; UI/API and exchange-control can configure Binance/Bybit futures testnet rows to isolated `1x` with read-back proof. |
| GitHub delivery | CI `27846306643`, Deploy Backend `27846380580`, Publish App Image `27846380521`, and Deploy Web `27846380520` passed for `e093ce2e`. Previous Stage `09` repair workflows also passed for `5d50b40a`, `884ed284`, and `2ba8b167`. |
| Mac Studio sync | `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `e093ce2e7d56d03f466d9e7f7e8f540de65c27d0`; pre-existing unrelated Stage `04b` files remain unstaged/untouched. |
| Runtime sync | `/opt/roehub/app` contains the new account-config endpoint and Settings UI action. |
| Smoke | `scripts/macos/smoke_prod.sh` passed with expected authenticated-current-user `401`, Redis `PONG`, and services listed. |
| Browser proof | `/settings#api` rendered Bybit and Binance rows with `SPOT Ready`, `FUTURES Ready`, and `Iso 1x`; console errors `0`, failed requests `0`, secret DOM scan `false`. Screenshot: `output/playwright/stage09-settings-account-config-ready.png`. |

## Active Connection Inventory

| Exchange | Market | Connection | State | Stage `09` use |
|---|---|---|---|---|
| Binance | spot | `8cec780c-c19c-4781-bd22-2af1d592039d` / `binance_testnet_2` | connection active and UI-ready; readiness `ready_for_trading`; validation `valid_trade_enabled`; last validation `2026-06-19 19:03:44` | used for long real testnet orders after `/settings` rotation and Binance spot preflight fix. |
| Binance | futures | `18e2de29-27e5-489b-94d3-f681e7e12e2c` / `binance_testnet_2` | active credential, account-state fresh, BTCUSDT isolated `1x`, one-way, qty `0` | used for long and short real testnet orders. |
| Bybit | spot | `af4c90fa-0a3b-4baf-a816-3630322bdf1b` / `bybit_testnet` | active credential, account-state fresh | used for long real testnet orders. |
| Bybit | futures | `6e61bf36-2202-437b-810e-ebe7ed48ba59` / `bybit_testnet` | active credential, account-state fresh, BTCUSDT isolated `1x`, one-way, qty `0` | used for long and short real testnet orders after explicit account-config action. |

Safe fingerprint comparison confirmed Binance spot/futures rows point to the same physical key suffix `RcSh`, and Bybit spot/futures rows point to the same physical key suffix `AuN5`. No ciphertext-copy DB repair was performed.

## Futures Account-Config Proof

Run tag: `stage09-account-config-20260619T201608Z`.

| Exchange | Connection | Before | Command result | After |
|---|---|---|---|---|
| Bybit futures | `6e61bf36-2202-437b-810e-ebe7ed48ba59` | account `unified`, sync `fresh/account_state_read_ok`, open orders `0`, position qty `0`, leverage `10`, margin mode `cross`, one-way | `fresh/account_config_write_ok`, target `isolated/1`, observed `isolated/1`, one-way | sync `fresh`, open orders `0`, qty `0`, leverage `1`, margin mode `isolated`, one-way |
| Binance futures | `18e2de29-27e5-489b-94d3-f681e7e12e2c` | sync `fresh/account_state_read_ok`, open orders `0`, position qty `0`, already isolated `1x`, one-way | `fresh/account_config_write_ok`, target `isolated/1`, observed `isolated/1`, one-way | sync `fresh`, open orders `0`, qty `0`, isolated `1x`, one-way |

The command refuses non-testnet, non-futures, inactive, not-owned, not-trade-ready rows and refuses configuration when open orders or non-zero position quantity exist. It does not place orders.

## Representative Matrix

All submitted orders used bounded testnet limit/cancel flow with `cancel_after_submit=true`. `$50` allocation was rounded only as required by exchange precision/min-notional.

| Exchange | Market | Direction | Sizing group | Result | Evidence |
|---|---|---|---|---|---|
| Binance | spot | long | `fixed_quote` | `passed` | `stage09-binance-spot-runtime-20260619T1921Z`: order `42579978450`, submit `new`, status `new`, cancel `canceled`, fills `0`; degraded private-stream preflight `binance_spot_rest_user_stream_deprecated` recorded. |
| Binance | spot | long | `fixed_equity_pct` | `passed` | `stage09-binance-spot-runtime-20260619T1921Z`: order `42579989113`, submit `new`, status `new`, cancel `canceled`, fills `0`; degraded private-stream preflight `binance_spot_rest_user_stream_deprecated` recorded. |
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
| Bybit | futures | short | `fixed_quote` | `passed` | `stage09-bybit-futures-short-platform-config-20260619T201909Z`: provider order `8f55f433-88c2-466a-af08-dbadc4f078b7`, DB order `c63fc48c-bd83-4f10-b90e-a5346ae07787`, status `cancelled`, fills `0`, latency `243.456 ms`. |
| Bybit | futures | short | `fixed_equity_pct` | `passed` | `stage09-bybit-futures-short-platform-config-20260619T201909Z`: provider order `21d65882-4530-4d70-bf6c-a914bc2b4769`, DB order `7d6aead0-8bbb-4b2e-8e62-dbc6a37204e5`, status `cancelled`, fills `0`, latency `244.426 ms`. |

## Redis, Events, Latency, Cleanup

| Evidence | Result |
|---|---|
| Clean Binance futures run tag | `stage09-20260618T2327Z`, background consumer only. |
| Clean Binance spot repair run tag | `stage09-binance-spot-runtime-20260619T1921Z`, background consumer only. |
| Clean Bybit futures short run tag | `stage09-bybit-futures-short-platform-config-20260619T201909Z`, background consumer processed both Redis messages before the foreground runner polled them. |
| Redis state after final Bybit futures short proof | `execution.requests.v1` length `41`, pending `0`, DLQ length `2` baseline. |
| Event sequence | Submitted rows recorded source event, accepted risk, Redis dispatch, native submit, status/cancel, and final DB status. |
| Fills | `execution_fills` count `0` and qty `0` for all Stage `09` real testnet rows. |
| Open orders after cleanup | Binance futures `0`, Binance spot cancelled, Bybit spot `0`, Bybit futures `0`. |
| Position after cleanup | Binance futures BTCUSDT qty `0`, isolated `1x`, one-way; Bybit futures BTCUSDT qty `0`, isolated `1x`, one-way. |
| Exchange execution readiness after cleanup | `ready/all_dependencies_ready`, `adapter_mode=testnet`. |

No mainnet submit was attempted. No raw secrets, API keys, ciphertext, signed payloads, cookies or tokens were written to this report.

## Business Impact

Краткое бизнес-резюме: реальный testnet-контур теперь доказан для Binance и Bybit, spot и futures, а futures account config больше не требует ручного шага на бирже перед проверкой. Платформа умеет явно и безопасно привести testnet futures connection к isolated `1x`, проверить это через read-back, затем провести заявку до биржи, проверить статус, отменить ее и записать результат в ledger.

Stage `09` now proves the real testnet execution path for the operationally useful matrix: Binance spot long, Binance futures long/short, Bybit spot long, and Bybit futures long/short across both sizing groups. Spot short remains product-blocked by design, not silently faked.

## Conditional Service-Call Coverage

| Call surface | Coverage |
|---|---|
| Source event / intent / risk | Covered by `ops_test` source events with accepted risk audits for all attempted rows. |
| Redis dispatch | Covered by `execution.requests.v1` message ids and post-run pending `0`. |
| Exchange execution process | Covered through launchd `exchange-execution` background consumer and native adapter events. |
| Account config service call | Covered for Binance and Bybit testnet futures; explicit operator action, no hidden mutation during submit. |
| Native exchange submit | Covered for Binance spot/futures and Bybit spot/futures supported rows. |
| Status / cancel | Covered for all submitted real testnet orders. |
| Fill / funding reconciliation | Fill count is `0` for all rows because limit orders were cancelled before execution; futures funding remains `pending` by design when no funding event exists. |
| Browser UI | Covered by authenticated `/settings#api` proof showing separate spot/futures readiness and account-config action. |
| New outbound provider surface | Covered by unit truth-table, API route tests, runtime account-config proof, and post-write account-state read-back for Binance/Bybit. |

## Logging, Redaction, Monitoring

| Surface | Coverage |
|---|---|
| Redaction | Report and command outputs include only connection ids, masked suffixes, order ids, statuses, timings, and aggregate account state. No raw secrets, ciphertext, signatures, cookies, tokens, or provider signed payloads are recorded. |
| Logs | No provider payload logging was enabled. Runtime evidence came from DB ledger rows, health endpoints, UI proof, and sanitized command output. |
| Metrics | `exchange-execution` readiness/Redis metrics were checked; submit latency is recorded in `execution_orders.latency_ms`. |
| Alerts/runbook | N/A for new alert rules; this stage used the existing Mac Studio deploy/smoke runbook and did not add monitoring configuration. |

## Cold-Head Artifact Review

`cold self-review fallback` completed on `2026-06-19`: an independent subagent pass was not used because delegation was not explicitly requested. Review lenses covered stage ledger continuity, prompt artifact continuity, Mac Studio path contract, service-call coverage, validation depth, redaction boundaries, browser/auth scope, and Stage `09` advancement criteria.

Verdict: `Release`. Blockers fixed: stale `blocked` status, stale Bybit manual-only config assumption, missing account-config service-call coverage, and stale handoff language. Residual risks: only future exchange API behavior drift; current runtime proof passed for both Binance and Bybit testnet.

## Quality Gates

| Gate | Result |
|---|---|
| `python -m py_compile` on touched Python modules/tests | passed |
| `python -m json.tool apps/web/locales/en.json` / `ru.json` | passed |
| `node --check apps/web/dist/js/pages/settings.js` | passed |
| Focused account-config/API/UI tests | `87 passed, 3 warnings` |
| Expanded exchange-control/API/web tests | `297 passed, 3 warnings` |
| `uv run ruff check src/trading/contexts/exchange_control src/trading/contexts/identity apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web/test_app_routes.py` | passed |
| `uv run pyright src/trading/contexts/exchange_control src/trading/contexts/identity apps/api tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web/test_app_routes.py` | passed (`0 errors`) |
| Stage `09` earlier exchange-execution gates | `uv run ruff`, `uv run pyright`, and focused exchange execution/control tests passed (`133 passed`) |
| `python -m tools.docs.generate_docs_index --check` | passed after final acceptance update. |
| GitHub Actions | CI/deploy passed for implementation commit `e093ce2e`; final docs-only commit requires docs-index/CI check after push. |
| Mac Studio smoke | passed after implementation deploy; final docs-only sync has no runtime service impact and is covered by the executor final report. |

## Acceptance Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| Representative matrix has pass/block result for every required bucket | `passed` | Matrix above covers every exchange/market/direction/sizing row. |
| At least one real testnet order path per accepted representative bucket | `passed` | Binance spot long, Binance futures long/short, Bybit spot long, and Bybit futures long/short all submitted/status-checked/cancelled through Redis/exchange-execution/native adapters. |
| Futures margin/leverage can be managed on both exchanges | `passed` | Binance and Bybit futures testnet account-config commands returned `fresh/account_config_write_ok` and read back isolated `1x`. |
| Redis ack-after-durable, pending, retry, DLQ evidence | `passed` | Final run left pending `0`; DLQ length stayed at baseline `2`. |
| Metrics/latency evidence | `passed` | Submit latency recorded in DB and runtime metrics exposed after orders. |
| No mainnet submit and no secret leakage | `passed` | `adapter_mode=testnet`; report contains only sanitized IDs/statuses. |
| Full Stage `09` unblock | `passed` | The previous Bybit futures short blocker is resolved by explicit platform-managed account config and real short submit/cancel proof. |

## Blockers

| Blocker | Severity | Owner / next action | Acceptance impact |
|---|---|---|---|
| none | none | Stage `10` may start after final docs-only delivery/sync evidence is recorded. | none |
| Spot short branches are unsupported in v1. | expected product exclusion | None for v1 unless a margin/borrow product is explicitly added. | Not a Stage `09` blocker. |

## Handoff

Stage `09` is accepted after platform-managed testnet futures account config, Binance/Bybit spot/futures readiness proof, real testnet submit/status/cancel coverage, browser Settings proof, CI/deploy, Mac Studio sync, and production smoke.

Stage `10` may proceed. Keep the same constraints: no mainnet submit, no chat-supplied secrets, no fake spot shorts, no hidden margin/leverage mutation inside order submit, and explicit read-back evidence for any future account-config write.
