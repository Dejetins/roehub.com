---
doc: rl-trading-agent-platform-v1-stage-16-testnet-rl-integration
status: blocked
stage: 16
updated_at: 2026-07-04
---

# Stage 16: Testnet RL Integration

Status: `blocked`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## 1. Result and stage status

Stage `16` delivered the bounded RL-to-testnet bridge code, but the stage is **not accepted**. The real-boundary Mac Studio run created `ml_agent_decision` source events, Redis-dispatched accepted intents, real Binance/Bybit testnet orders and fills, and one unsupported-branch no-intent block. Acceptance is blocked because the final order records ended in `adapter_error` and the latest reconciliation rows are `pending`, not clean matched terminal evidence.

Коротко по-бизнесу: RL теперь технически доходит до существующего testnet execution контура и реально открывает тестовые сделки, но продуктово включать testnet-режим нельзя. Пользователь увидит риск некорректного статуса сделки: fill уже есть, а финальный execution ledger говорит `adapter_error`/`pending`. Это хуже, чем отсутствие сделки, потому что оператору нужен ручной разбор состояния.

Prerequisite check before implementation:

| Required prerequisite | Observed ledger status | Result |
|---|---|---|
| RL Stage `15` Paper RL integration | `accepted` | `passed` |
| Classic producer Stage `09` Real testnet representative orders | `accepted` | `passed` |

Prompt evidence:

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/16-testnet-rl-integration.md` |
| Prompt sha256 | `8effe068af60edab26ee1f3cd2ee7f42c555dcf2022fba1602b1f756e0ce2596` |

Delivery state:

| Surface | Evidence |
|---|---|
| Code commit 1 | `1f4c331a42004c40fd5fd637160a35579bf63e93` (`Add RL testnet execution harness`) |
| Code commit 2 | `47160eaf98c3fbc37b435be2a80a2a5df4f9050b` (`Fix RL testnet futures sizing`) |
| GitHub CI | `28684653311` and `28685057151`, both `success` |
| Deploy workflows | For `47160eaf...`: `Publish App Image` `28685115509`, `Deploy Backend` `28685115505`, `Deploy Web` `28685115489`, all `success` |
| Mac Studio checkout/runtime | `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `47160eaf...`; `/opt/roehub/app` exposed `testnet-once --quantity`; `scripts/macos/smoke_prod.sh` passed |

## 2. File manifest

| Path | Status | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/live_execution/domain/risk_gate.py` | modified | Added stricter fail-closed non-paper `ml_agent_decision` checks for strategy binding/profile/run, position ownership, capital reservation, policy, market data and account context. | `compatible-change` |
| `src/trading/contexts/rl_trading/adapters/outbound/acl/live_execution_producer.py` | modified | Added RL testnet ACL method that creates `ml_agent_decision` source events and accepted intents only for supported branches; added futures `quantity` support. | `compatible-change` |
| `apps/worker/rl_trading_inference/main/main.py` | modified | Added `testnet-once` proof harness, Redis dispatch mode, duplicate replay check, sanitized output and optional `--quantity`. | `compatible-change` operator CLI |
| `tests/unit/contexts/live_execution/test_execution_ingress_service.py` | modified | Covered fail-closed ML testnet risk gate cases. | `none` |
| `tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py` | modified | Covered testnet ACL idempotency, unsupported spot short and futures sizing. | `none` |
| `tests/unit/apps/worker/test_rl_trading_inference.py` | modified | Covered `testnet-once` dispatch, duplicate dispatch and unsupported branch output. | `none` |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/16-testnet-rl-integration.md` | created | Stage report, evidence, blocker and handoff. | `compatible-change` docs |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Records Stage `16` blocked state and next-stage gate. | `compatible-change` docs |
| `docs/architecture/README.md` | modified | Generated docs index after adding the Stage `16` report. | `compatible-change` docs |

Outside expected paths: none.

Runtime artifacts outside git:

| Path | Evidence |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage16_testnet_rl_integration_v1/stage16_testnet_20260703t215200z/feature_fixture_manifest.json` | Sanitized feature fixtures; manifest sha256 `de6df27ff01d5345baa014137069500df44af1f6f841390d8b8b9673bfe36cad`. |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage16_testnet_rl_integration_v1/stage16_testnet_20260703t215200z/stage16_testnet_runtime_summary.json` | Sanitized runtime summary written by the proof harness; summary hash was not collected before SSH access became unavailable. |

## 3. Evidence and quality gates

Local quality gates:

| Gate | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/live_execution/test_execution_ingress_service.py::test_risk_gate_rejects_source_aware_safety_cases tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py tests/unit/apps/worker/test_rl_trading_inference.py::test_testnet_cli_dispatches_intent_and_duplicate_dispatch tests/unit/apps/worker/test_rl_trading_inference.py::test_testnet_cli_blocks_spot_short_without_dispatch` | `26 passed` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | `497 passed, 8 skipped, 3 warnings` |
| `uv run pytest -q tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py tests/unit/apps/worker/test_rl_trading_inference.py` | `55 passed` |
| `uv run pytest -q tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py tests/unit/apps/worker/test_rl_trading_inference.py::test_testnet_cli_dispatches_intent_and_duplicate_dispatch tests/unit/apps/worker/test_rl_trading_inference.py::test_testnet_cli_blocks_spot_short_without_dispatch` | `7 passed` after futures `quantity` fix |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | `0 errors` |
| `uv run ruff check .` | passed |
| `uv run pyright` | `0 errors` |
| `uv run pytest -q -ra` | `1558 passed, 16 skipped, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after blocked report/ledger update |
| Cold artifact review | cold self-review fallback; independent sub-agent was not used because the available delegation tool requires an explicit user request; verdict passed after status, proof-boundary and redaction checks |

Runtime evidence collected on Mac Studio after `main` delivery:

| Case | CLI/action evidence | DB/order evidence | Verdict |
|---|---|---|---|
| `binance_spot_open_long` | `action=open_long`, `outcome=intent_created`, `risk_gate_accepted`, intent `df9ebd15-4023-42ad-a5b5-8fd3754a8802` | one fill recorded (`0.000190000000` BTC), one earlier reconciliation `matched/spot_order_status_and_fills_matched`, but final order `adapter_error/exchange_http_400` and latest reconciliation `pending/adapter_error_reconciliation_pending` | blocked |
| `bybit_spot_open_long` | `action=open_long`, `outcome=intent_created`, `risk_gate_accepted`, intent `d669532b-5e66-4e33-8704-415a8bd7dbb8` | one fill recorded (`0.000096000000` BTC), one earlier reconciliation `matched/spot_order_status_and_fills_matched`, but final order `adapter_error/exchange_ret_code_170213` and latest reconciliation `pending/adapter_error_reconciliation_pending` | blocked |
| `binance_futures_open_short` | `action=open_short`, `outcome=intent_created`, `risk_gate_accepted`, intent `984ef0e6-17b0-45a7-b523-4b33cdc2c77f` | one fill recorded (`0.001000000000` BTC), reconciliation remained `pending/funding_reconciliation_pending`, then final order `adapter_error/exchange_http_400` | blocked |
| `bybit_futures_open_short` | `action=open_short`, `outcome=intent_created`, `risk_gate_accepted`, intent `3ce42f88-c4af-4671-816b-0b44732e4cdc` | one fill recorded (`0.001000000000` BTC), reconciliation remained `pending/funding_reconciliation_pending`, then final order `adapter_error/exchange_ret_code_110001` | blocked |
| `bybit_spot_open_short_block` | `action=open_short`, `outcome=no_intent`, `outcome_reason=testnet_spot_short_not_supported`, `intent_id=null` | no order/fill/reconciliation created for this case | passed unsupported-branch block |

Runtime aggregate:

| Metric | Value |
|---|---|
| `execution_source_events` delta | `+5` |
| `execution_intents` delta | `+4` |
| `execution_orders` delta | `+4` |
| `execution_fills` delta | `+4` |
| `execution_reconciliation_runs` delta | `+8` |
| `exchange_execution_request_observations` delta | `+4` |
| non-testnet order delta | `0` |

## 4. Contract, safety, and delivery state

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route or public API payload changed. |
| Operator CLI | `compatible-change` | Added explicit `testnet-once` and optional `--quantity`; existing commands are unchanged. |
| Domain/port behavior | `compatible-change` | `ml_agent_decision` can now produce testnet intents through the existing execution ingress; unsupported branches fail closed. |
| Persisted schema | `none` | No migration/table change. The runtime attempt wrote existing execution source/intent/order/fill/reconciliation rows. |
| Config/defaults | `none` in git | No committed config default changed. A launchctl env override attempt for `ROEHUB_EXCHANGE_EXECUTION_CANCEL_AFTER_SUBMIT=false` was ineffective because the launchd command sources `roehub.env` later. |
| External side effects | `compatible-change`, not accepted | Real testnet orders/fills occurred on Binance/Bybit spot/futures. Mainnet stayed unchanged (`non_testnet_orders` delta `0`). |
| Secrets/redaction | `passed` | No secrets, passphrases, tokens, ciphertext, raw signed requests, raw provider payloads or checkpoint tensors were written to git docs. Runtime summaries contain sanitized IDs/status/counts. |

Service-call coverage:

| Boundary | Coverage |
|---|---|
| RL/ML worker -> `live_execution` ingress | Covered by `testnet-once` source-event and intent creation. |
| `live_execution` -> Redis dispatch | Covered by four dispatched intents and duplicate-dispatch checks. |
| Redis -> `exchange-execution` worker | Covered by four `exchange_execution_request_observations`. |
| `exchange-execution` -> Binance/Bybit testnet adapters | Covered by real spot/futures order/fill rows; final status is blocked. |
| ML worker -> exchange SDK/secret custody | N/A by design; ML worker does not resolve credentials or call exchange SDKs. |
| Mainnet provider calls | N/A; non-testnet order delta stayed `0`. |

Alerts, monitoring, and runbooks:

| Surface | Coverage |
|---|---|
| Exchange-execution readiness | Mac Studio `/health/ready` reported `ready/all_dependencies_ready`, `adapter_mode=testnet`, Redis pending `0` before the runtime run. |
| Runtime smoke | `scripts/macos/smoke_prod.sh` passed after deployment of `47160eaf...`. |
| New alert/runbook | N/A for accepted rollout because Stage `16` is blocked. Follow-up should use the existing exchange-execution runbook plus a specific market-order cancel-path repair note before acceptance. |
| Operator action | Required: restore Mac Studio SSH/keychain access, inspect/close testnet positions if needed, and rerun clean Stage `16` proof. |

Safety caveats:

- Stage `16` is blocked despite real fills because final order state and latest reconciliation are not clean terminal evidence.
- The attempted `launchctl setenv ROEHUB_EXCHANGE_EXECUTION_CANCEL_AFTER_SUBMIT=false` could not be verified as unset after SSH authentication became unavailable. It did not affect the launched process in this run because the launchd command sources `/Users/daniildegtyarev/.config/roehub/roehub.env` after launchctl environment.
- Testnet spot buys and futures shorts created real testnet fills. They are not mainnet positions, but follow-up cleanup/close should be handled by an operator with Mac Studio access.

## 5. Blockers and next-stage handoff

Blocker: clean real-boundary testnet evidence is missing. Required accepted evidence must show supported RL testnet branches finishing with terminal order/fill/reconciliation state, not latest `adapter_error` plus `pending` reconciliation.

Observed root cause candidate: market orders filled, but the running `exchange-execution` process still followed the configured post-submit cancel/status path and converted filled orders into adapter errors. A bounded `cancel_after_submit=false` canary could not be completed because SSH access to `macstudio` became unavailable after the first runtime run (`Permission denied` / `Too many authentication failures`).

Stage `17` is **not allowed**. Next executor should rerun Stage `16` after restoring Mac Studio SSH/keychain access and either:

- run a bounded clean canary with `ROEHUB_EXCHANGE_EXECUTION_CANCEL_AFTER_SUBMIT=false` applied after sourcing host-local env, then restore the normal launchd service; or
- change the exchange-execution market-order lifecycle so filled market orders do not enter the cancel path.

Do not treat this report as testnet approval. Paper remains accepted from Stage `15`; testnet remains blocked; live/mainnet remain blocked.
