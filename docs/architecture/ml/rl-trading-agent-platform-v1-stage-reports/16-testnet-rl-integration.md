---
doc: rl-trading-agent-platform-v1-stage-16-testnet-rl-integration
status: accepted
stage: 16
updated_at: 2026-07-05
---

# Stage 16: Testnet RL Integration

Status: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## 1. Result and stage status

Stage `16` is accepted after `post_main_production_runtime_proof`: changed revision `f83b3d7daccacb2979fb0fea180fcdabe3d6226d` reached `main`, green CI/GitHub Actions passed, deploy/sync completed, `/opt/roehub/app` matched the changed runtime file hash, and only then the target runtime proof was collected. Pre-main/read-only Mac Studio checks are not counted for acceptance. The RL `ml_agent_decision` path now reaches the existing `live_execution` ingress, Redis dispatch and `exchange-execution` worker, creates real Binance/Bybit spot/futures testnet orders and fills for supported branches, and blocks unsupported spot short without creating an intent.

Бизнес-смысл: RL-контур впервые доказан не только в paper/monitor-only режиме, а на реальном testnet execution пути. Это означает, что продукт может переходить к следующей testnet-нагрузке Stage `17`, но это не разрешение на live/mainnet: деньги/позиции двигались только в Binance/Bybit testnet, а mainnet по-прежнему закрыт до отдельных Stage `19`/`20` gate.

The earlier blocked attempt is kept as root-cause evidence: filled market orders were later sent through the configured cancel path, and exchanges rejected cancelling already-filled market orders. Commit `f83b3d7daccacb2979fb0fea180fcdabe3d6226d` fixes that by skipping cancel when `status.fills` is present and by treating futures fills without immediate funding events as matched order/fill reconciliation.

Prerequisite check:

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
| Stage bridge commits | `1f4c331a42004c40fd5fd637160a35579bf63e93`, `47160eaf98c3fbc37b435be2a80a2a5df4f9050b` |
| Reconciliation repair commit | `f83b3d7daccacb2979fb0fea180fcdabe3d6226d` |
| GitHub CI | `28745831335` `success` for `f83b3d7daccacb2979fb0fea180fcdabe3d6226d` |
| Deploy workflows | `Publish App Image` `28745957622`, `Deploy Backend` `28745957606`, `Deploy Web` `28745957615` and `28745962995`, all `success` |
| Target runtime deploy/sync | `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `f83b3d7daccacb2979fb0fea180fcdabe3d6226d`; `/opt/roehub/app` file hash for `exchange_execution_process.py` matched `9b1c4402687ef1981864e446d4d4dcabc4362e81b8cf9017acabbc3e8a2536ab`; `scripts/macos/smoke_prod.sh` passed before the accepted testnet run |

## 2. File manifest

| Path | Status | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/live_execution/domain/risk_gate.py` | modified | Added fail-closed non-paper `ml_agent_decision` checks for strategy binding/profile/run, position ownership, capital reservation, policy, market data and account context. | `compatible-change` |
| `src/trading/contexts/rl_trading/adapters/outbound/acl/live_execution_producer.py` | modified | Added RL testnet ACL method for supported branches and futures `quantity`. | `compatible-change` |
| `apps/worker/rl_trading_inference/main/main.py` | modified | Added `testnet-once`, Redis dispatch mode, duplicate replay check, sanitized output and optional `--quantity`. | `compatible-change` operator CLI |
| `src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py` | modified | Skips post-submit cancel for already-filled market orders and records matched futures order/fill reconciliation when funding events are not yet present. | `compatible-change` |
| `tests/unit/contexts/live_execution/test_execution_ingress_service.py` | modified | Covered fail-closed ML testnet risk gate cases. | `none` |
| `tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py` | modified | Covered testnet ACL idempotency, unsupported spot short and futures sizing. | `none` |
| `tests/unit/apps/worker/test_rl_trading_inference.py` | modified | Covered `testnet-once` dispatch, duplicate dispatch and unsupported branch output. | `none` |
| `tests/unit/contexts/live_execution/test_exchange_execution_process.py` | modified | Covered skip-cancel-on-filled-market-order and futures-fill-without-funding matched reconciliation. | `none` |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/16-testnet-rl-integration.md` | created/modified | Stage report, root cause, accepted evidence and handoff. | `compatible-change` docs |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Records Stage `16` accepted state and opens Stage `17`. | `compatible-change` docs |
| `docs/architecture/README.md` | modified | Generated docs index after Stage `16` docs updates. | `compatible-change` docs |

Created: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/16-testnet-rl-integration.md`.

Modified: files listed above.

Deleted: none.

Outside expected paths: none.

Runtime artifacts outside git:

| Path | Evidence |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage16_testnet_rl_integration_v1/stage16_testnet_20260703t215200z/stage16_testnet_runtime_summary.json` | Blocked root-cause run: real fills occurred, but final orders became `adapter_error` and latest reconciliation became `pending`; summary sha256 `51671c17f0c90217775e09ffc82c69ecef4c5c056284f60aec536c017bff082f`. |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage16_testnet_rl_integration_v1/stage16_testnet_20260705t154427z/feature_fixture_manifest.json` | Accepted-run sanitized feature fixtures; manifest sha256 `0840b3d04fd9f7aeeb05c8258f588bf22d331a640ba6341baf54b655abb03aaa`. |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage16_testnet_rl_integration_v1/stage16_testnet_20260705t154427z/stage16_testnet_runtime_summary.json` | Accepted runtime summary; summary sha256 `2a35733feea095c73f03d77e8b00e5881e42d5957332a9e86b3af3d5d71127e2`. |

Candidate manifest:

| Field | Value |
|---|---|
| Path | `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json` |
| sha256 | `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c` |

## 3. Evidence and quality gates

Local quality gates:

| Gate | Result |
|---|---|
| Focused regression pytest for market-order cancel and futures reconciliation | `4 passed` |
| `uv run pytest -q tests/unit/contexts/live_execution/test_exchange_execution_process.py` | `12 passed` |
| Prompt/touched focused pytest | `73 passed` |
| `uv run ruff check src/trading/contexts/live_execution src/trading/contexts/rl_trading apps tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py` | passed |
| `uv run pyright src/trading/contexts/live_execution src/trading/contexts/rl_trading apps tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py` | `0 errors` |
| `uv run ruff check .` | passed |
| `uv run pyright` | `0 errors` |
| `uv run pytest -q -ra` | `1560 passed, 16 skipped, 3 warnings` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after report/ledger update and docs index regeneration |

Accepted Mac Studio runtime proof:

| Case | Runtime evidence | Order/fill/reconciliation evidence | Verdict |
|---|---|---|---|
| `binance_spot_open_long` | `outcome=intent_created`, intent `0229f0b9-43bb-4b8a-9b89-8099b215cb03` | `order_status=status_checked`, `status_reason=filled`, `fill_count=1`, latest reconciliation `matched/spot_order_status_and_fills_matched` | accepted |
| `bybit_spot_open_long` | `outcome=intent_created`, intent `a436dd64-bdce-4d8d-bb8d-1c6bc9de232e` | `order_status=status_checked`, `status_reason=filled`, `fill_count=1`, latest reconciliation `matched/spot_order_status_and_fills_matched` | accepted |
| `binance_futures_open_short` | `outcome=intent_created`, intent `0f9695af-a911-4e5e-955c-5b64e4701f12` | `order_status=status_checked`, `status_reason=filled`, `fill_count=1`, latest reconciliation `matched/futures_order_status_and_fills_matched` | accepted |
| `bybit_futures_open_short` | `outcome=intent_created`, intent `bd7b79d3-95ec-4488-96a3-b6832ef05f9e` | `order_status=status_checked`, `status_reason=filled`, `fill_count=1`, latest reconciliation `matched/futures_order_status_and_fills_matched` | accepted |
| `bybit_spot_open_short_block` | `outcome=no_intent`, `outcome_reason=testnet_spot_short_not_supported`, `intent_id=null` | no order/fill/reconciliation created | accepted unsupported-branch block |

Runtime aggregate for accepted run:

| Metric | Value |
|---|---|
| `execution_source_events` delta | `+5` |
| `execution_intents` delta | `+4` |
| `execution_orders` delta | `+4` |
| `execution_fills` delta | `+4` |
| `execution_reconciliation_runs` delta | `+4` |
| `exchange_execution_request_observations` delta | `+4` |
| non-testnet order delta | `0` |
| acceptance status | `accepted=true` |

Latency, nearest-rank p95 over the four supported real testnet cases:

| Segment | p95 |
|---|---|
| `source_to_intent_ms` | `17.837` |
| `intent_to_dispatch_ms` | `23.046` |
| `dispatch_to_submit_pending_ms` | `1473.374` |
| `source_to_submit_pending_ms` | `1512.129` |
| `provider_adapter_latency_ms` | `681.812` |

Artifact review: cold self-review fallback. Independent sub-agent review was not used because the available delegation tool requires an explicit user request. Verdict: passed after checking status transitions, proof boundaries, redaction, runtime hashes and Stage `17` gate wording.

## 4. Contract, safety, and delivery state

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route or public API payload changed. |
| Operator CLI | `compatible-change` | Added explicit `testnet-once` and optional `--quantity`; existing commands are unchanged. |
| Domain/port behavior | `compatible-change` | `ml_agent_decision` can now create bounded testnet intents through existing execution ingress; unsupported branches fail closed. |
| Exchange-execution lifecycle | `compatible-change` | Filled market orders are no longer pushed into cancel; no public schema/API change. |
| Persisted schema | `none` | No migration/table change. Runtime wrote existing source, intent, order, fill, reconciliation and observation rows. |
| Config/defaults | `none` | No committed config default changed; `cancel_after_submit` remains `true`, but filled market orders skip cancel in code. |
| External side effects | `compatible-change` testnet only | Real Binance/Bybit testnet spot/futures fills occurred. Mainnet stayed untouched: non-testnet order delta `0`. |
| Secrets/redaction | `passed` | No secrets, tokens, cookies, passphrases, ciphertext, raw signed requests, raw provider payloads or checkpoint tensors were written to git docs or runtime summaries. |

Service boundary coverage:

| Boundary | Coverage |
|---|---|
| RL/ML worker -> `live_execution` ingress | Covered by five `ml_agent_decision` source events. |
| `live_execution` -> Redis dispatch | Covered by four dispatched intents and duplicate dispatch replay. |
| Redis -> `exchange-execution` worker | Covered by four `exchange_execution_request_observations`. |
| `exchange-execution` -> Binance/Bybit testnet adapters | Covered by real spot/futures orders and fills. |
| ML worker -> exchange SDK/secret custody | N/A by design; ML worker does not resolve credentials or call exchange SDKs. |
| Mainnet provider calls | N/A; non-testnet order delta stayed `0`. |

Alerts, monitoring and runbooks:

| Surface | Coverage |
|---|---|
| Service smoke | `scripts/macos/smoke_prod.sh` passed before the accepted testnet run. |
| Exchange-execution observations | Covered by four `exchange_execution_request_observations` rows in the accepted run. |
| New alert | N/A; no new alert rule was required because this stage repaired lifecycle behavior and did not add a new operational service. |
| New runbook | N/A; the existing exchange-execution runbook remains the operator surface. Stage `17` may add load/soak-specific notes if new operating behavior appears. |

## 5. Blockers and next-stage handoff

No Stage `16` blocker remains. The earlier `adapter_error`/`pending` blocker is resolved by commit `f83b3d7daccacb2979fb0fea180fcdabe3d6226d` and the accepted runtime proof at `/opt/roehub/state/rl_trading/evaluation_runs/stage16_testnet_rl_integration_v1/stage16_testnet_20260705t154427z/stage16_testnet_runtime_summary.json`.

Stage `17` is allowed to start. Handoff notes:

- Treat Stage `16` as testnet approval only, not live/mainnet approval.
- Keep mainnet blocked until Stage `19` readiness review and explicit Stage `20` approval.
- The accepted proof created small real testnet spot buys and futures shorts; operator cleanup may still be needed on Binance/Bybit testnet accounts.
- Stage `17` should start from `strategy_run_id=2ed77f4d-ee5f-4c2c-b86f-6a57c553fa5d`, runtime summary sha256 `2a35733feea095c73f03d77e8b00e5881e42d5957332a9e86b3af3d5d71127e2`, and code commit `f83b3d7daccacb2979fb0fea180fcdabe3d6226d`.
