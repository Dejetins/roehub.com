---
doc: rl-trading-agent-platform-v1-stage-15-paper-rl-integration
status: accepted
stage: 15
updated_at: 2026-07-03
---

# Stage 15: Paper RL Integration

Status: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## 1. Result and stage status

Stage `15` added a paper-only bridge from accepted RL inference decisions into the existing `live_execution` source-event, risk-gate, intent and paper-accounting path. Business outcome: RL can now prove paper execution behavior without opening testnet, live, mainnet, exchange SDK, Redis dispatch or secret-custody ownership in the ML worker.

What changed:

- `ml_agent_decision` paper decisions can create a source event and no-dispatch intent through `ExecutionIngressService`.
- Paper fills/accounting for RL decisions reuse the existing paper accounting repository and stable `source_event_id` identity.
- `apps/worker/rl_trading_inference` gained an explicit `paper-once` local harness for sanitized Stage `15` proof.
- Hold and unsupported RL paper actions remain `no_intent`; only `open_long`/`open_short` create paper no-dispatch intents/orders.

Business-readable impact / бизнес-эффект:

| Area | Impact |
|---|---|
| User capability | Пользовательский RL-сценарий теперь можно проверять в paper mode с детерминированными paper fills/accounting до открытия любого реального биржевого пути. |
| Operator safety | Paper evidence остается no-dispatch: stage доказывает source events, risk decisions, idempotency и accounting без testnet/mainnet side effects. |
| Rollout control | Stage `16` может стартовать от зафиксированного paper ledger contract, но testnet должен собрать отдельное доказательство и не наследует paper approval. |

Prerequisite check:

| Required prerequisite | Observed ledger status | Result |
|---|---|---|
| RL Stage `14` User risk/sizing policy | `accepted` | `passed` |
| Classic producer Stage `07` Paper full branch coverage | `accepted` | `passed` |

Prompt evidence:

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/15-paper-rl-integration.md` |
| Prompt sha256 | `878378541a8d4324d340c8d236c1ee972ecc800657ac8eddaa1c796b12a60df3` |

The optional `.codex/agents/.context/promt_manager_state.yaml` still contains stale prompt-pack delivery wording from early RL stages. It was read and ignored for execution facts because the live ledger, current prompt and `.codex/AGENTS.md` supersede it.

## 2. File manifest

### Created

| Path | Reason | Contract impact |
|---|---|---|
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/15-paper-rl-integration.md` | Stage `15` report with prompt hash, manifest, evidence and handoff. | `compatible-change` docs/report |

### Modified

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/adapters/outbound/acl/live_execution_producer.py` | Add `record_paper_decision` ACL method from RL decision to existing live_execution ingress and paper accounting. | `compatible-change` additive ACL behavior |
| `apps/worker/rl_trading_inference/main/main.py` | Add explicit `paper-once` local harness for Stage `15` paper ledger/idempotency/parity evidence. | `compatible-change` additive opt-in CLI command |
| `src/trading/contexts/live_execution/domain/risk_gate.py` | Add `ml_agent_decision` paper no-dispatch branch when `paper_no_exchange_submit=True`. | `compatible-change` additive risk semantics |
| `src/trading/contexts/live_execution/application/use_cases/paper_accounting.py` | Add `record_rl_paper_execution` using existing paper order/fill/accounting storage identity. | `compatible-change` additive application behavior |
| `tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py` | Cover RL paper ACL idempotency, hold no-intent and paper accounting bridge. | `none` tests |
| `tests/unit/contexts/live_execution/test_execution_ingress_service.py` | Cover `ml_agent_decision` paper no-dispatch risk branch and replay behavior. | `none` tests |
| `tests/unit/contexts/live_execution/test_paper_accounting_service.py` | Cover RL paper accounting idempotency and simulator/accounting parity. | `none` tests |
| `tests/unit/apps/worker/test_rl_trading_inference.py` | Cover the `paper-once` worker harness. | `none` tests |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `15` accepted and open Stage `16`. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Docs index regeneration after adding the Stage `15` report. | `compatible-change` docs index |

### Deleted

| Path | Reason | Contract impact |
|---|---|---|
| - | No files deleted. | `none` |

### Outside expected paths

| Path | Justification |
|---|---|
| `tests/unit/apps/worker/test_rl_trading_inference.py` | Test coverage for the expected primary worker path `apps/worker/rl_trading_inference`. |

## 3. Evidence and quality gates

Focused checks:

| Command | Result |
|---|---|
| `uv run pytest -q tests/unit/apps/worker/test_rl_trading_inference.py::test_paper_cli_records_intent_order_and_parity tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py tests/unit/contexts/live_execution/test_execution_ingress_service.py::test_ml_agent_decision_paper_no_exchange_submit_uses_no_dispatch_risk_branch tests/unit/contexts/live_execution/test_paper_accounting_service.py::test_rl_paper_execution_records_idempotent_order_fill_and_accounting_parity` | `6 passed` |
| `uv run ruff check apps/worker/rl_trading_inference/main/main.py tests/unit/apps/worker/test_rl_trading_inference.py src/trading/contexts/rl_trading/adapters/outbound/acl/live_execution_producer.py src/trading/contexts/live_execution/domain/risk_gate.py src/trading/contexts/live_execution/application/use_cases/paper_accounting.py tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/contexts/live_execution/test_paper_accounting_service.py` | passed |
| `uv run pyright apps/worker/rl_trading_inference/main/main.py tests/unit/apps/worker/test_rl_trading_inference.py src/trading/contexts/rl_trading/adapters/outbound/acl/live_execution_producer.py src/trading/contexts/live_execution/domain/risk_gate.py src/trading/contexts/live_execution/application/use_cases/paper_accounting.py tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/contexts/live_execution/test_paper_accounting_service.py` | `0 errors, 0 warnings, 0 informations` |

Prompt-level gates:

| Command | Result |
|---|---|
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | `0 errors, 0 warnings, 0 informations` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | `493 passed, 8 skipped, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after regenerating `docs/architecture/README.md` |

The skipped tests are existing optional Torch gates in the default environment where `torch` is not installed.

Real-boundary local evidence, not tests-only:

| Surface | Evidence |
|---|---|
| Worker harness | `uv run python` invoked `apps.worker.rl_trading_inference.main` `paper-once` with temporary JSON files outside the repo. |
| Decision path | action `open_long`, source type `ml_agent_decision`, source event ref `rl:f64b121721179850901c0465da65c72130e75d5844c66fe9c8eafb459c953de9`. |
| Paper execution ledger | `source_events_created=1`, `intents_created=1`, `paper_orders_created=1`, `paper_fills_created=1`, `paper_accounting_created=1`. |
| Idempotency proof | replay in the same harness returned `duplicate_replay=true` with no duplicate source event, intent, order, fill or accounting rows. |
| Simulator/accounting parity | `simulator_parity.status=accepted`, `max_abs_diff=0E-8`, `tolerance=0`, diffs for `equity`, `fee_total`, and `position_quantity` all `0E-8`. |
| Safety boundary | risk status `rejected`, risk reason `paper_no_exchange_submit`; no Redis dispatch, testnet, live, mainnet or exchange SDK path was invoked. |

Proof boundary:

| Boundary label | Stage `15` status |
|---|---|
| `local_in_process_worker_harness` | completed; this is the only runtime-like evidence claimed for this local-only stage. |
| `target_host_readiness_pre_main` | not run and not claimed. |
| `read_only_existing_runtime_smoke` | not run and not claimed. |
| `post_main_production_runtime_proof` | not run and not claimed. This proof requires the changed revision on `main`, green CI/GitHub Actions for that revision, deploy/sync to the target runtime, and then production smoke/runtime verification. |

## 4. Contract, safety, and delivery state

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API route or payload changed. |
| DTO schema | `none` | No browser/API DTO changed. |
| Port/domain contract | `compatible-change` | Additive RL ACL method and source-aware risk-gate behavior for `ml_agent_decision` paper mode. |
| Persisted schema | `none` | Existing execution and paper accounting tables/repositories are reused; no migration. |
| Config/defaults | `none` | No runtime profile or default changed; `paper-once` is explicit opt-in CLI. |
| Request hash / cache key / persistence identity | `compatible-change` | Adds deterministic paper idempotency keys for RL source events/intents; existing identities are unchanged. |
| Service-call auth/timeout/retry/error semantics | `compatible-change` | Paper path uses existing ingress/repository semantics; unknown/replay state is handled by source/intent idempotency lookup. |
| External side effects | `compatible-change` | Adds paper ledger writes only; no exchange submit, provider call, Redis execution dispatch, testnet, live or mainnet side effect. |
| Logs/metrics/traces/audit/redaction | `compatible-change` | Worker harness prints sanitized counts, hashes and status; it does not print secrets, credentials, raw provider payloads or user credentials. |
| Alerts/monitoring/runbooks | `none` | No alert or runbook behavior changed. |
| Browser-visible behavior | `none` | Browser runtime verification is disabled and no UI changed. |
| Performance | `none` for performance claims | No hot-path performance claim is made; harness latencies are diagnostic only. |

Service-call coverage:

| Surface | Coverage |
|---|---|
| Auth model | N/A for this local worker harness; no user credentials, exchange credentials or browser auth are consumed. |
| Timeout/retry/fallback | No external service call was added. Replay behavior is idempotency lookup through existing repositories. |
| Error/status behavior | Additive paper no-dispatch status only: `ml_agent_decision` can return `risk_rejected/paper_no_exchange_submit` in paper mode. |
| Redaction | Harness evidence records counts, hashes and statuses only; no secrets, cookies, tokens, raw provider payloads or signed requests are recorded. |

Operational notes:

- `paper-once` is a local proof command, not a production scheduler enablement.
- `paper_no_exchange_submit` is still represented as a rejected risk decision so existing dispatch machinery cannot publish it.
- Close behavior is fail-closed as `paper_close_position_snapshot_required` until a later stage owns strategy-position close accounting.
- Large artifacts and temporary harness JSON remained outside git.

Delivery state: `local-only`.

No branch, commit, push, CI, deploy, Mac Studio sync, `target_host_readiness_pre_main`, `read_only_existing_runtime_smoke` or `post_main_production_runtime_proof` was performed for Stage `15` in this run. `post_main_production_runtime_proof` remains a future publish/deploy boundary and requires `main`, green CI/GitHub Actions, deploy/sync and production verification.

## 5. Blockers and next-stage handoff

Stage `15` blockers: none for local acceptance.

Next-stage state:

| Stage | Allowed now | Reason |
|---|---|---|
| `16` Testnet RL integration | `yes` for the local staged chain | Stage `15` paper source-event/intent/accounting/idempotency/parity evidence is accepted locally. |
| Mainnet | `no` | Mainnet remains blocked until Stage `19` approval and Stage `20` prompt conditions. |

Handoff for Stage `16`:

- Reuse the `ml_agent_decision -> ExecutionIngressService -> risk gate -> intent` path added here.
- Do not treat `paper_no_exchange_submit` as testnet approval; Stage `16` must collect real testnet readiness/evidence separately.
- Preserve RL strategy-run scoping: `owner_user_id + strategy_run_id + exchange + market_type + symbol`.
- Keep retry behavior idempotency-first: source and intent replay must look up existing records before any retry.

Cold self-review fallback: completed. Verdict `Release after fixes`; fixes applied in this report/ledger include explicit redaction coverage, alert/runbook `N/A`, delivery state, docs index evidence and Stage `16` handoff. Residual risk: no production/main delivery or Mac Studio runtime proof has been claimed for this local-only stage.
