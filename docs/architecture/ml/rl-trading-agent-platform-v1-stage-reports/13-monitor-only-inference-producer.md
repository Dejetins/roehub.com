---
doc: rl-trading-agent-platform-v1-stage-13-monitor-only-inference-producer
status: blocked
stage: 13
updated_at: 2026-07-03
---

# Stage 13: Monitor-only Inference Producer

Статус: `blocked`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## Pre-Edit Gate

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/13-monitor-only-inference-producer.md` |
| Prompt sha256 | `6a67a8b5dde3f12625190863f6a7e33baaa17ccccad2b8ba34a1871f1ac8d306` |
| Ledger state observed before work | `current_stage=13`; Stage `13` pending/current |
| Prerequisite verdict | accepted Stage `10`, Stage `10A`, Stage `11`, Stage `12`; Stage `13` may proceed |
| `.codex/agents/.context/promt_manager_state.yaml` | read; ignored as stale prompt-generation state because it still points to Stage `02A` and branch/PR delivery, while current `.codex/AGENTS.md`, prompt and ledger require `main`/local-only execution unless explicitly published |
| Browser/auth | `/strategies` browser QA required only after source-event evidence exists; smoke credentials must not be written into docs, traces, screenshots or logs |
| Exchange/provider effects | `N/A`; monitor-only inference must not call exchange SDKs, create order intents, dispatch Redis execution requests, read provider credentials or submit paper/testnet/live/mainnet orders |

## Planned Concrete File List Before Edit

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/monitor_only_inference.py` | Stage `13` monitor-only feature parsing, train/live parity, preloaded supervised policy inference, source-event payload and latency summary domain surface. | `compatible-change` additive internal Python domain surface |
| `src/trading/contexts/rl_trading/adapters/outbound/acl/__init__.py` | Adapter package marker/export for live-execution ACL. | `compatible-change` internal import surface |
| `src/trading/contexts/rl_trading/adapters/outbound/acl/live_execution_producer.py` | ACL that records `ml_agent_decision` source events and marks terminal `no_intent` without creating order intents. | `compatible-change` additive source-event integration |
| `apps/worker/rl_trading_inference/__init__.py` | Worker package marker. | `compatible-change` new optional worker package |
| `apps/worker/rl_trading_inference/main/__init__.py` | Worker main package marker. | `compatible-change` new optional worker entrypoint |
| `apps/worker/rl_trading_inference/main/main.py` | CLI for disabled-by-default status, parity checks and one-shot monitor-only inference canary. | `compatible-change` new optional worker entrypoint |
| `apps/worker/rl_trading_inference/wiring/__init__.py` | Worker wiring package marker. | `compatible-change` internal import surface |
| `apps/worker/rl_trading_inference/wiring/modules/__init__.py` | Worker wiring module exports. | `compatible-change` internal import surface |
| `apps/worker/rl_trading_inference/wiring/modules/rl_trading_inference.py` | Worker config loader, Redis live-window reader, Prometheus metrics, health/readiness and one-shot runtime composition. | `compatible-change` disabled-by-default runtime surface |
| `tests/unit/contexts/rl_trading/domain/test_monitor_only_inference.py` | Focused coverage for feature parity, preloaded model inference, source-event payload and latency summaries. | `none` test-only |
| `tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py` | Prove `ml_agent_decision` writes terminal `no_intent` and creates no `ExecutionIntent`. | `none` test-only |
| `tests/unit/apps/worker/test_rl_trading_inference.py` | Worker/config/metrics/CLI smoke coverage. | `none` test-only |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/__init__.py` | Export Stage `13` domain surface. | `compatible-change` additive internal Python exports |
| `src/trading/contexts/rl_trading/adapters/outbound/__init__.py` | Export Stage `13` ACL adapter. | `compatible-change` additive internal Python exports |
| `configs/dev/rl_trading_ml_runtime.yaml` | Add disabled-by-default monitor-only inference config, metrics port, Redis source and latency budgets. | `compatible-change` additive fail-closed config/default contract |
| `configs/test/rl_trading_ml_runtime.yaml` | Add disabled-by-default monitor-only inference config for tests. | `compatible-change` additive fail-closed config/default contract |
| `configs/prod/rl_trading_ml_runtime.yaml` | Add disabled-by-default monitor-only inference config for Mac Studio production-like runtime. | `compatible-change` additive fail-closed config/default contract |
| `tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py` | Assert Stage `13` inference config stays disabled, monitor-only and host-local. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/13-monitor-only-inference-producer.md` | Stage report, evidence and handoff. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Final Stage `13` status/evidence and next-stage allowance after validation. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Docs index sync if required by generator. | `compatible-change` docs index |

Deleted: none planned.

Outside expected paths: none planned. The new ACL adapter is under `src/trading/contexts/rl_trading`, and the new worker is under the prompt's expected `apps/worker/rl_trading_inference` surface.

## Scope

Входит:

- добавить доменный Stage `13` контракт для Redis schema v1 payload -> feature window -> deterministic feature matrix/hash;
- добавить preloaded supervised policy loader для accepted Stage `08M` manifest;
- добавить ACL producer, который пишет `ml_agent_decision` source event и сразу переводит его в terminal `no_intent`;
- добавить disabled-by-default worker/config/CLI/metrics/readiness surface;
- покрыть feature parity, no-intent source events, config, metrics and CLI focused tests.

Не входит:

- paper/testnet/live/mainnet order intents;
- exchange SDK calls, secret custody, provider credentials, signed requests;
- hot-path ClickHouse scans;
- model training, retraining, promotion or user-owned custom model upload;
- Stage `14+` risk/sizing behavior.

## File Manifest

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/monitor_only_inference.py` | Stage `13` feature window parsing, parity, preloaded supervised policy inference, source-event payload and latency summary. | `compatible-change` additive Python domain surface |
| `src/trading/contexts/rl_trading/adapters/outbound/acl/__init__.py` | ACL package export. | `compatible-change` internal import surface |
| `src/trading/contexts/rl_trading/adapters/outbound/acl/live_execution_producer.py` | Records `ml_agent_decision` source events and marks `no_intent` without creating `ExecutionIntent`. | `compatible-change` additive live-execution integration through existing source-event port |
| `apps/worker/rl_trading_inference/__init__.py` | Worker package marker. | `compatible-change` optional worker package |
| `apps/worker/rl_trading_inference/main/__init__.py` | Worker main package marker. | `compatible-change` optional worker package |
| `apps/worker/rl_trading_inference/main/main.py` | CLI for `status`, `parity`, `canary-once`, and bounded health/metrics server smoke. | `compatible-change` optional worker entrypoint |
| `apps/worker/rl_trading_inference/wiring/__init__.py` | Worker wiring package marker. | `compatible-change` internal import surface |
| `apps/worker/rl_trading_inference/wiring/modules/__init__.py` | Worker wiring exports. | `compatible-change` internal import surface |
| `apps/worker/rl_trading_inference/wiring/modules/rl_trading_inference.py` | Config loader, Redis read-only window reader, Prometheus metrics and health/readiness HTTP server. | `compatible-change` disabled-by-default runtime surface |
| `tests/unit/contexts/rl_trading/domain/test_monitor_only_inference.py` | Feature parity, fail-closed missing fields, preloaded policy, source-event payload and latency tests. | `none` test-only |
| `tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py` | Proves no-intent source-event write and no `ExecutionIntent`. | `none` test-only |
| `tests/unit/apps/worker/test_rl_trading_inference.py` | Worker config, Redis reader, metrics and CLI tests. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/13-monitor-only-inference-producer.md` | Stage report. | `compatible-change` docs/report |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/__init__.py` | Export Stage `13` domain surface. | `compatible-change` additive Python exports |
| `src/trading/contexts/rl_trading/adapters/outbound/__init__.py` | Export Stage `13` ACL adapter. | `compatible-change` additive Python exports |
| `configs/dev/rl_trading_ml_runtime.yaml` | Add disabled-by-default monitor-only inference config, Redis input and latency budget. | `compatible-change` additive fail-closed config defaults |
| `configs/test/rl_trading_ml_runtime.yaml` | Same contract for test profile. | `compatible-change` additive fail-closed config defaults |
| `configs/prod/rl_trading_ml_runtime.yaml` | Same contract for prod/Mac Studio profile. | `compatible-change` additive fail-closed config defaults |
| `tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py` | Assert Stage `13` config remains disabled, monitor-only and source-event fail-closed. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Records blocked Stage `13` state and next-stage gate. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Mechanical docs index sync after adding the Stage `13` report. | `compatible-change` docs index |

Deleted: none.

Outside expected paths: none for owned Stage `13` work.

Unrelated dirty worktree state observed and ignored: existing mainnet prompt/architecture files under `.codex/agents/generated/mainnet-real-money-trading-v1/`, `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, and `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`. These are not Stage `13` changes and were not edited for this report.

## Observed State

| Area | Evidence summary |
|---|---|
| Prerequisites | Ledger showed accepted Stage `10`, `10A`, `11`, and `12`; Stage `13` was current/pending before implementation. |
| Accepted candidate contract | Domain loader only accepts Stage `09` constants for accepted Stage `08M` candidate id `stage08m_a3823cbd01143878_fd7c614b` and manifest sha256 `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c`. |
| Live feed contract | Redis schema v1 parser requires `schema_version=1`; feature construction still fails closed when `volume_quote` or `trades_count` is absent. |
| Source event contract | Existing live-execution `ml_agent_decision` source type and `no_intent` outcome are reused; no persisted schema or order-intent model change was needed. |
| Runtime config | All profiles keep `inference.enabled=false` and `source_events.enabled=false`; readiness reports `degraded_reasons=["inference_disabled"]`. |
| Runtime source availability | Local accepted manifest search under `/opt/roehub/state/rl_trading` and repo returned no manifest; local Redis on `127.0.0.1:6379` refused connection; Mac Studio SSH failed before runtime access. |

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API route/DTO was changed. |
| Live-execution source-event contract | `compatible-change` | Adds a new producer of existing `source_type=ml_agent_decision` events through existing ingress/repository ports. |
| Execution/order intent contract | `none` | Monitor-only path never calls `create_intent`; tests assert `intents_created=0`. |
| Persisted schema | `none` | No migration or table contract changed. |
| Config schema/defaults | `compatible-change` | Adds disabled-by-default `inference.mode=monitor_only`, Redis read source, metrics port and latency budget. |
| Metrics/logs | `compatible-change` | Adds bounded `rl_trading_inference_*` metrics with no user/strategy identifiers in labels. |
| Browser-visible behavior | `none` | Stage `11` reusable outcome journal already renders `ml_agent_decision`; no UI change was made. Browser QA was not run because no runtime source-event evidence exists. |
| Runtime/deploy | `unknown` | Code is local only and not delivered to `main`, CI, deploy or Mac Studio runtime. |

## Quality Gates And Evidence

| Gate | Result |
|---|---|
| `uv run python -m py_compile ...` for new Stage `13` Python files | passed |
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_monitor_only_inference.py tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py tests/unit/apps/worker/test_rl_trading_inference.py tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py` | passed: `13 passed` |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed: `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed: `478 passed, 8 skipped, 3 warnings`; skips are optional `torch` tests in default non-`rl-ml` env |
| `uv run python -m apps.worker.rl_trading_inference.main.main status --config configs/dev/rl_trading_ml_runtime.yaml` | passed; output reports `ready=false`, `mode=monitor_only`, `degraded_reasons=["inference_disabled"]`, `source_event_outcome=no_intent` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after mechanical docs index sync |
| Mac Studio runtime access | blocked: `ssh macstudio` failed with too many authentication failures; `ssh -o IdentitiesOnly=yes -o PreferredAuthentications=publickey macstudio` failed `Permission denied` |
| Local live Redis source | blocked: Python Redis client to `127.0.0.1:6379` returned `Connection refused` |
| Accepted manifest source | blocked locally: no Stage `08M` candidate manifest found under `/opt/roehub/state/rl_trading` or repo checkout |

## Blockers

Stage `13` is `blocked`, not `accepted`.

Acceptance blockers:

- required Mac Studio supervised/runtime evidence could not run because SSH authentication failed before host access;
- required Redis/live feature window proof could not run because no local Redis was reachable and Mac Studio Redis could not be inspected;
- accepted Stage `08M` runtime manifest was not available locally, so a real accepted-manifest canary could not run from this checkout;
- DB source-event evidence and browser `/strategies` journal QA require a runtime source event and therefore were not executed.

These blockers are external runtime/source availability blockers, not local code/test failures.

## Delivery State

`local-only`; no branch, PR, staging, commit, push, CI, deploy, Mac Studio sync, production smoke, provider call or exchange side effect was performed.

No secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, raw signed requests or checkpoint tensors were written to docs/tests/logs.

## Cold Self-Review

Mode: `cold self-review fallback`.

Verdict: `Blocked for acceptance; implementation can be retained`.

Findings:

- `Blocker`: Prompt requires target runtime/e2e proof; unavailable SSH/Redis/manifest means Stage `13` must stay blocked and Stage `14` must not start.
- `Follow-up check`: after runtime access is restored, rerun accepted-manifest canary, Redis/offline parity, DB source-event query, metrics scrape, and `/strategies` browser QA.
- `Residual risk`: code has not been deployed or exercised against Mac Studio production-like runtime, so runtime import/env/Redis/DB differences may still appear.

## Next-Stage Handoff

Stage `14` is not allowed.

To resume Stage `13`, an operator should restore Mac Studio SSH access without sharing secrets in chat, then run the same Stage `13` code against:

- accepted Stage `08M` candidate manifest under `/opt/roehub/state/rl_trading/`;
- a live Redis `md.candles.1m.<instrument_key>` closed-candle window containing `volume_quote` and `trades_count`;
- the matching offline/canonical fixture for identical feature-vector proof;
- live-execution DB source-event storage proving `ml_agent_decision -> no_intent` and `0` intents;
- Prometheus/health readiness scrape;
- `/strategies` signal/outcome journal browser QA.
