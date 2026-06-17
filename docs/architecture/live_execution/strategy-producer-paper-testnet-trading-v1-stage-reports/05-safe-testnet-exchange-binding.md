# Stage 05: Safe Testnet Exchange Binding

Статус: `accepted`

## Pre-Start

User required before start: nothing

Stage `04` проверен в ledger: `accepted`; следующий stage разрешен.

## Scope

Stage `05` ограничен безопасной привязкой стратегии к уже добавленным через `/settings` Binance/Bybit testnet-подключениям. Auto-config запрещен: stage не вызывает exchange APIs для изменения leverage, margin mode или position mode.

## Concrete Planned File List Before Editing

| File | Planned action | Reason |
|---|---:|---|
| `src/trading/contexts/strategy/application/ports/exchange_connection_readiness.py` | modify | Передать context запуска в checker без секретов: mode/market/symbol/direction/notional. |
| `src/trading/contexts/strategy/application/use_cases/live_strategy_profiles.py` | modify | Блокировать futures short до fresh account/config readiness; сохранить текущие monitor/paper/live правила. |
| `apps/api/routes/strategies.py` | modify | Передать launch context из backtest launch в profile readiness; сохранить DTO shape. |
| `apps/api/wiring/modules/strategy.py` | modify | Связать exchange-control readiness с account projection/config guard через существующие Postgres projections and local non-secret account-state mapping. |
| `src/trading/contexts/live_execution/domain/account_state.py` | modify | Добавить безопасный `side`/notional requirement context для guard. |
| `src/trading/contexts/live_execution/application/use_cases/account_projection.py` | modify | Расширить fail-closed reason vocabulary для missing balance, min-notional, unsafe futures short и config mismatch. |
| `src/trading/contexts/exchange_control/adapters/outbound/exchange_account_state.py` | modify | Добавить read-only Binance testnet account/config reads alongside existing Bybit reads. |
| `tests/unit/contexts/exchange_control/test_exchange_account_state_reader.py` | create | Cover Binance/Bybit read-only account-state mapping without network or secrets. |
| `tests/unit/contexts/live_execution/test_account_projection_service.py` | modify | Доказать futures short accepted only on isolated `1x`, plus mismatch/missing balance blockers. |
| `tests/unit/apps/api/test_strategies_routes.py` | modify | Доказать launch route передает context and blocks unsafe futures short through readiness. |
| `tests/unit/apps/api/test_strategy_wiring_module.py` | modify | Доказать real wiring checker persists sanitized projection and requires safe futures short guard. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/05-safe-testnet-exchange-binding.md` | create | Stage evidence report. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify | Stage ledger handoff after validation. |
| `docs/architecture/README.md` | modify if generated index requires it | Docs index for new stage report; pre-existing dirty README/ML entries are not Stage 05 scope. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| this report; `tests/unit/contexts/exchange_control/test_exchange_account_state_reader.py` | `src/trading/contexts/exchange_control/adapters/outbound/exchange_account_state.py` | none | Add read-only Binance account/config reader support and Bybit futures config normalization for account-state projections. | compatible-change: no endpoint shape change; exchange-control account-state reads now support Binance in addition to existing Bybit. |
| none | `src/trading/contexts/live_execution/domain/account_state.py`; `src/trading/contexts/live_execution/application/use_cases/account_projection.py`; `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/account_projection_repository.py`; `tests/unit/contexts/live_execution/test_account_projection_service.py` | none | Require balance, min-notional, margin mode and leverage proof for futures shorts; persist the non-secret requirement context in existing guard JSON. | compatible-change: additive requirement fields in existing JSONB; existing schema/table shape unchanged. |
| none | `src/trading/contexts/strategy/application/ports/exchange_connection_readiness.py`; `src/trading/contexts/strategy/application/ports/__init__.py`; `src/trading/contexts/strategy/application/__init__.py`; `src/trading/contexts/strategy/application/use_cases/live_strategy_profiles.py`; `apps/api/routes/strategies.py`; `apps/api/wiring/modules/strategy.py`; `tests/unit/apps/api/test_strategies_routes.py`; `tests/unit/apps/api/test_strategy_wiring_module.py` | none | Pass non-secret launch context into exchange readiness, bind testnet launches to owned testnet connection metadata, and call account projection/config guard for testnet futures short. | compatible-change: stricter fail-closed readiness semantics for unsafe testnet launches; DTO shape unchanged. |
| none | `src/trading/contexts/exchange_control/application/connections.py`; `tests/unit/contexts/exchange_control/test_exchange_control_runtime.py` | none | Convert legacy/non-Transit ciphertext decrypt failures into bounded validation/account-state unavailable errors after runtime proof exposed active Binance futures credentials that exchange-control cannot decrypt. | compatible-change: fail-closed error reason becomes more specific; no secret exposure and no endpoint shape change. |
| none | `src/trading/contexts/exchange_control/adapters/outbound/exchange_validation.py`; `tests/unit/contexts/exchange_control/test_exchange_validation.py` | none | Validate Binance futures testnet credentials against the Binance Futures Testnet API instead of the Spot Testnet account-restrictions endpoint. | compatible-change: fixes false invalid-credentials classification for Binance futures testnet keys; no endpoint shape or secret-handling change. |
| none | this report; `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Record final operator-side Binance Futures Testnet funding/config proof and Stage `05` acceptance. | none: docs/evidence update only. |

Files outside prompt expected paths: `src/trading/contexts/strategy/application/*` and `tests/unit/apps/api/test_strategy_wiring_module.py` are touched because the existing strategy launch readiness port and API wiring are the narrowest place to pass non-secret launch context and enforce exchange binding before starting a run. `src/trading/contexts/exchange_control/application/connections.py` and its runtime test were added after Mac Studio proof showed the active Binance futures testnet credentials fail before provider I/O because the stored ciphertexts are not Transit ciphertexts; the fix keeps the custody failure bounded and secret-safe.

## Contract Impact

Initial classification before implementation:

| Dimension | Classification | Note |
|---|---|---|
| Public API contract | `compatible-change` | Existing launch/profile response DTOs are intended to stay unchanged; readiness may block more unsafe testnet futures short cases. |
| Port contract | `compatible-change` | Exchange readiness checker gains optional non-secret launch context. |
| DTO schema | `none` | No response/request field shape change planned. |
| Persisted schema | `none` | Use existing projection/config guard tables. |
| Config/env schema | `none` | No new env keys planned. |
| Service-call semantics | `compatible-change` | Launch readiness may consult account projection/config guard before accepting futures short. |
| External side effects | `none` | Only read-only exchange account/config calls; no auto-config or order submit. |
| Metrics/audit/report semantics | `compatible-change` | New bounded readiness reasons may appear in existing metrics/report labels. |

## Evidence

### Local gates

| Command | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/exchange_control/test_exchange_account_state_reader.py tests/unit/contexts/live_execution/test_account_projection_service.py tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/api/test_strategy_wiring_module.py` | passed, `23 passed` |
| `uv run pytest -q tests/unit/contexts/exchange_control/test_exchange_control_runtime.py::test_internal_account_state_read_reports_legacy_ciphertext_as_unavailable` | passed |
| `uv run pytest -q tests/unit/contexts/exchange_control/test_exchange_validation.py` | passed, `8 passed` |
| `uv run ruff check src/trading/contexts/exchange_control src/trading/contexts/live_execution apps tests` | passed |
| `uv run pyright src/trading/contexts/exchange_control src/trading/contexts/live_execution apps tests` | passed, `0 errors` |
| `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/contexts/live_execution tests/unit/apps` | passed, initially `419 passed, 3 warnings`; after Binance futures validator repair, `421 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed |

### Fail-closed proof

| Scenario | Evidence | Result |
|---|---|---|
| Futures short with isolated margin and leverage `1x`, USDT balance and min-notional satisfied | `tests/unit/contexts/live_execution/test_account_projection_service.py::test_futures_short_guard_accepts_only_isolated_1x_with_balance_and_notional`; `tests/unit/apps/api/test_strategy_wiring_module.py::test_exchange_control_readiness_checker_accepts_safe_testnet_futures_short` | accepted with `safe_testnet_futures_short_1x_isolated_verified`; sanitized projection and guard result persisted. |
| Missing futures position/config evidence | `tests/unit/contexts/live_execution/test_account_projection_service.py::test_futures_short_guard_blocks_missing_position_config` | blocked with `unsafe_futures_short`. |
| Margin/leverage mismatch, missing balance and min-notional issue | `tests/unit/contexts/live_execution/test_account_projection_service.py::test_futures_short_guard_blocks_margin_balance_and_min_notional_mismatch` | blocked with bounded reasons `min_notional_issue`, `missing_balance`, `margin_mode_mismatch`, `leverage_mismatch`. |
| Launch route futures short mismatch | `tests/unit/apps/api/test_strategies_routes.py::test_launch_from_backtest_variant_blocks_unsafe_testnet_futures_short` | API blocks before run creation with `strategy_launch.readiness_blocked`, reason `unsafe_futures_short`. |
| Launch route verified futures short | `tests/unit/apps/api/test_strategies_routes.py::test_launch_from_backtest_variant_accepts_verified_testnet_futures_short` | API starts run only when checker reports `safe_testnet_futures_short_1x_isolated_verified`. |

### Target-runtime pre-deploy inventory

Mac Studio sanitized SQL showed active testnet keys exist before deploy:

| Exchange | Market | Environment | Status | Capability | Readiness |
|---|---|---|---|---|---|
| Bybit | spot | testnet | active | trading | ready_for_trading |
| Binance | futures | testnet | active | trading | ready_for_trading |

Pre-deploy exchange-control account-state reads:

| Exchange | Market | Result |
|---|---|---|
| Bybit | spot testnet | `fresh/account_state_read_ok`, balances `2`, positions `0`, open orders `0`, filters `1`. |
| Binance | futures testnet | request failed on old runtime before Stage `05` deploy; expected because current deployed reader did not yet support Binance account-state. |

### GitHub delivery and Mac Studio runtime

| Evidence | Result |
|---|---|
| Direct main delivery | `8442802065599983559092b02d60ab6044800c4f` implemented Stage `05`; `5fba0440845d3cbb775fec67bdfb69a7ae6b5bdf` repaired the docs index; `17773dd9c12df80d01819082d3a9f979c59f1eab` added the bounded legacy-ciphertext blocker. |
| GitHub CI | Initial CI `27656780948` failed only on docs-index drift; final CI `27657223558` succeeded on `17773dd9`. |
| Deploy workflows | Publish App Image `27657392776`, Deploy Web `27657392799` and `27657401927`, Deploy Backend `27657392761` succeeded. Earlier forced backend deploy `27656995422` proved the `5fba0440` runtime sync path after the docs-only commit had skipped backend deploy. |
| Mac Studio checkout/runtime | `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `17773dd9`; `/opt/roehub/app` contains `safe_testnet_futures_short_1x_isolated_verified`, Binance account-state reader support, and the legacy-ciphertext guard. |
| Runtime smoke | `/opt/roehub/app/scripts/macos/smoke_prod.sh` exited `0` after final backend deploy. |
| Binance futures validator repair | Commit `4affd97b058d64d8bc72795f241e5af877275549` fixed Binance futures testnet validation; it is included under `origin/main` SHA `d2354e0e51b1ed7dfecc26440d8a81fd47c756ca`. CI `27713063290` succeeded and forced Deploy Backend `27713170598` succeeded. |
| Runtime smoke after repair | `/opt/roehub/app/scripts/macos/smoke_prod.sh` exited `0`; `/opt/roehub/app` contains the Binance futures `/fapi` validation path. |

### Post-deploy exchange-read evidence

Sanitized Mac Studio probe after final deploy:

| Connection | API readiness | Account-state result |
|---|---|---|
| Binance futures testnet `b53cf726` | active, trading, `ready_for_trading`, `trading_policy_ok` | blocked before provider I/O: `exchange_connection_account_state_unavailable`. |
| Binance futures testnet `ed7bf9a0` | active, trading, `ready_for_trading`, `trading_policy_ok` | blocked before provider I/O: `exchange_connection_account_state_unavailable`. |
| Bybit spot testnet `af4c90fa` | active, trading, `ready_for_trading`, `trading_policy_ok` | `fresh/account_state_read_ok`, account mode `unified`, balances `2`, positions `0`, open orders `0`, filters `1`. |

Direct root-cause diagnostic for both active Binance futures testnet rows:

| Connection | Credential decrypt evidence | Result |
|---|---|---|
| `b53cf726` | active credential failed before Binance HTTP; exception type `ValueError`, message `exchange credential ciphertext must be a Transit ciphertext` | exchange-control now reports bounded blocker `exchange_connection_account_state_unavailable`. |
| `ed7bf9a0` | active credential failed before Binance HTTP; exception type `ValueError`, message `exchange credential ciphertext must be a Transit ciphertext` | exchange-control now reports bounded blocker `exchange_connection_account_state_unavailable`. |

Futures-short readiness proof after final deploy:

| Scenario | Runtime result |
|---|---|
| Binance futures short, notional `50` | `eligible=false`, reason `exchange-control internal request failed with status 503 code exchange_connection_account_state_unavailable`. |
| Binance futures short, notional `1` | `eligible=false`, same custody blocker before min-notional/config guard can run. |

Repeat proof after Binance futures validator repair and operator key re-add:

| Evidence | Runtime result |
|---|---|
| Stored connection | New Binance futures testnet connection `0b8c536b`, key last4 `RcSh`, status `active`, credential status `active`, ciphertext scheme `exchange_control_transit_v1`, capability `trading`, readiness `ready_for_trading`, reason `trading_policy_ok`. The older invalid/history row for the same last4 is disabled. |
| Runtime validator probe | `valid_trade_enabled`, reason `trade_permission_detected`, permissions `trade`, market `futures`, environment `testnet`, warnings `none`. |
| Account-state read | `fresh/account_state_read_ok`; balances `8`, positions `1`, open orders `0`, filters `716`. |
| BTCUSDT position/config | position quantity `0`, position mode `one_way`, margin mode `cross`, leverage `20`. |
| USDT balance | free `0E-8`, locked `0`, total `0E-8`. |
| BTCUSDT filters | min notional `50`, min qty `0.0001`, step size `0.0001`, tick size `0.10`. |
| Futures-short readiness, notional `50` | `eligible=false`, reason `insufficient_balance`. |
| Futures-short readiness, notional `1` | `eligible=false`, reason `min_notional_issue`. |
| Latest guard result, notional `50` | `mismatch`, reasons `insufficient_balance`, `margin_mode_mismatch`, `leverage_mismatch`. |
| Latest guard result, notional `1` | `mismatch`, reasons `min_notional_issue`, `insufficient_balance`, `margin_mode_mismatch`, `leverage_mismatch`. |

Repeat proof after operator funding/config update:

| Evidence | Runtime result |
|---|---|
| Runtime boundary | Mac Studio `/opt/roehub/app`, in-process `ExchangeConnectionService` + `HttpExchangeAccountStateReader` using stored Transit credentials; no chat-provided secrets. |
| Binance requests used | Validator uses Binance Futures Testnet `/fapi/v2/account`; account-state reader uses `/fapi/v2/account`, `/fapi/v2/positionRisk`, `/fapi/v1/openOrders`, and `/fapi/v1/exchangeInfo`. |
| Stored connection | Binance futures testnet connection `0b8c536b`, key last4 `RcSh`, status `active`, credential status `active`, ciphertext scheme `exchange_control_transit_v1`. |
| Runtime validator probe | `valid_trade_enabled`, reason `trade_permission_detected`, `effective_capability=trading`, readiness `ready_for_trading`, reason `trading_policy_ok`, IP status `not_restricted_testnet`. |
| Account-state read | `fresh/account_state_read_ok`; balances `8`, positions `1`, open orders `0`, filters `716` for BTCUSDT-only Stage `05` proof. |
| USDT balance | free `4999.95403601`, total `4999.95403601`. |
| BTCUSDT position/config | position quantity `0`, position mode `one_way`, margin mode `isolated`, leverage `1`. |
| BTCUSDT filters | min notional `50`, min qty `0.0001`, step size `0.0001`, tick size `0.10`. |
| Futures-short readiness, notional `50` | `ready_for_risk=true`, status `fresh`, reason `account_projection_fresh`; latest persisted guard remains the `$50` accepted path. |
| Futures-short fail-closed check, notional `1` | `mismatch`, reason `min_notional_issue`, proving below-min-notional submit remains blocked. |

Metrics after final probe included bounded account-state labels:

| Metric | Value |
|---|---|
| `exchange_account_state_read_total{exchange="unknown",reason="exchange_connection_account_state_unavailable",result="rejected"}` | `4` |
| `exchange_account_state_read_total{exchange="bybit",reason="account_state_read_ok",result="fresh"}` | `1` |

## Blockers

Stage `05` is `accepted`.

Resolved blocker: the new Binance futures testnet connection is stored with Transit ciphertext, validates through the Futures Testnet API, and account-state reads are now `fresh/account_state_read_ok`.

Resolved blocker: after operator-side Binance Futures Testnet setup, BTCUSDT is configured as isolated margin with leverage `1x`, USDT free balance is above the required `$50` notional, and the Stage `05` guard reports `ready_for_risk=true`.

No active Stage `05` blocker remains.

## Handoff

Stage `06` may start. The next executor must keep the same safety boundary: no keys in chat/docs, no mainnet, no exchange auto-config, and no real order submit outside later stages that explicitly require testnet order proof.
