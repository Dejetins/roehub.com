---
doc: rl-trading-agent-platform-v1-stage-18-rl-soak-incident-drills
stage: "18"
status: accepted
mode: monitor_only_technical_soak
updated_at: 2026-07-05
---

# Stage 18: RL soak and incident drills

Status: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

## 1. Result and stage status

Stage `18` is accepted only as `monitor_only_technical_soak`.

The accepted evidence proves bounded monitor-only runtime safety and incident
drill behavior for up to `20` tickers. It does not prove 24h/7d full soak,
model quality, trading edge, paper/testnet/live execution readiness, product
readiness, full trade-readiness or mainnet readiness.

Business-readable impact:

| Area | Result |
|---|---|
| Runtime safety | The Mac Studio main checkout ran the Stage `18` harness at changed revision `83195a411d6995e5533c6f757ba6f68e8d5ead79` and accepted all monitor-only safety checks. |
| Incident response | `kill_switch`, `pause`, `rollback`, `missing_artifact`, `stale_feed`, and `unknown_state` drills all passed in dry-run/fail-closed mode. |
| UI/operator state | Local browser harness opened `/strategies`, verified degraded/safe RL/ML state, `monitor_only` active mode, disabled retraining/rollback controls, `ml_agent_decision -> no_intent`, and `0` console warnings/errors. |
| Next stage | Stage `19` remains blocked because Stage `08N` set `stage19_mainnet_readiness_allowed=false`. |

Prerequisite check:

| Required prerequisite | Observed status | Result |
|---|---|---|
| Stage `08N` candidate quality reclassification | `accepted`, `quality_status=accepted_for_research_only` | passed only for monitor-only technical soak |
| Stage `08N` Stage `18` allowance | `stage18_monitor_only_technical_soak_allowed=true`; `stage18_soak_allowed=false` | bounded mode only |
| Stage `17` runtime/load gate | `accepted` as `infrastructure_only` | passed |
| Current ledger before execution | `current_stage=18` | passed |

Prompt evidence:

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/18-rl-soak-incident-drills.md` |
| Prompt sha256 | `93a1e132968b1d05a499cb77806d4498857e58a8d268ef886f6a8b165ea308f3` |

The optional `.codex/agents/.context/promt_manager_state.yaml` was read and
ignored because it referred to stale Stage `02A` and branch/PR state. The live
ledger, Stage `18` prompt and `.codex/AGENTS.md` superseded it.

## 2. File manifest

### Created

| Path | Reason | Contract impact |
|---|---|---|
| `scripts/rl_trading/stage18_rl_soak_incident_drills.py` | Opt-in Stage `18` CLI that reuses Stage `17` monitor-only load input, validates incident drills and writes a sanitized summary. | `compatible-change` operator CLI/test harness |
| `src/trading/contexts/rl_trading/domain/stage18_soak_incident_drills.py` | Deterministic Stage `18` monitor-only technical soak and incident-drill summarizer. | `compatible-change` internal domain/reporting surface |
| `tests/unit/contexts/rl_trading/domain/test_stage18_soak_incident_drills.py` | Domain coverage for accepted and blocked Stage `18` summaries. | `none` test-only |
| `tests/unit/scripts/rl_trading/test_stage18_rl_soak_incident_drills.py` | CLI coverage for sanitized summary writing. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/18-rl-soak-incident-drills.md` | Stage `18` report, evidence and handoff. | `compatible-change` docs/report |
| `docs/runbooks/rl-trading-operations.md` | RL trading monitor-only operations and incident-drill runbook. | `compatible-change` docs/runbook |

### Modified

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/__init__.py` | Export Stage `18` domain helpers. | `compatible-change` Python package export |
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | Record accepted Stage `18` monitor-only scope and Stage `19` blocker. | `compatible-change` docs/plan |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `18` accepted, set no executable next stage, and record Stage `19` blocker. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Docs index regeneration after adding Stage `18` report/runbook. | `compatible-change` docs index |

### Deleted

| Path | Reason | Contract impact |
|---|---|---|
| - | No files deleted. | `none` |

Runtime/browser artifacts outside git:

| Path | State |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage18_soak_incident_drills_v1/stage18_macstudio_20260705t190918z/stage18_soak_incident_drills_summary.json` | Created on `macstudio`; sanitized Stage `18` summary. |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage17_multi_ticker_runtime_load_v1/stage18_macstudio_20260705t190918z_stage17_input/stage17_multi_ticker_runtime_load_summary.json` | Created by Stage `18` harness as bounded Stage `17` input. |
| `output/playwright/stage18-rl-soak-desktop.png` | Local ignored browser QA screenshot. |

## 3. Evidence and quality gates

Mac Studio runtime evidence:

| Field | Value |
|---|---|
| Host | `macstudio` |
| Checkout | `/Users/daniildegtyarev/Projects/roehub.com` |
| Git revision | `83195a411d6995e5533c6f757ba6f68e8d5ead79` |
| Summary path | `/opt/roehub/state/rl_trading/evaluation_runs/stage18_soak_incident_drills_v1/stage18_macstudio_20260705t190918z/stage18_soak_incident_drills_summary.json` |
| Summary file sha256 | `d50ca6b29e9ec38e7f82fce5eb5e8a79bf98b5f35aa0970e34831b0b74671eb4` |
| Summary hash | `1fd1de0808ead32cf51fd14fb2fa2e3da6199b2763038d41cae5cbf7755ee209` |
| Stage `17` input summary hash | `49192bc4f4e9c34d5e660c6c8f78a37648f1803d12f181e2bf3de3a4a85d16d6` |
| Max observed tickers | `20` |
| Observation count | `26` |
| 24h minimum status | `not_claimed_for_monitor_only_technical_soak` |
| 7d status | `not_claimed_for_monitor_only_technical_soak` |

Runtime acceptance checks:

| Check | Result |
|---|---|
| `stage17_input_accepted` | `true` |
| `stage17_handoff_allows_monitor_only_technical_soak` | `true` |
| `max_tickers_within_stage18_limit` | `true` |
| `monitor_only_source_events_only` | `true` |
| `redis_execution_stream_growth_zero` | `true` |
| `required_incident_drills_passed` | `true` |
| `ui_safe_or_degraded_state_recorded` | `true` |
| `no_exchange_side_effects` | `true` |
| `no_order_state_involved` | `true` |

Incident drills:

| Drill | Result |
|---|---|
| `kill_switch` | `passed` |
| `pause` | `passed` |
| `rollback` | `passed` |
| `missing_artifact` | `passed` |
| `stale_feed` | `passed` |
| `unknown_state` | `passed` |

Browser evidence:

| Surface | Evidence |
|---|---|
| Harness | `local_browser_runtime_harness_no_credentials_no_exchange` |
| URL | `http://127.0.0.1:64218/strategies` |
| Browser title | `Strategies | Roehub` |
| Dashboard requests | `GET /api/ui/strategies/dashboard => 200` |
| Console | `0` errors, `0` warnings |
| RL/ML state | `Degraded / stage18_browser_harness: stage18_monitor_only_technical_soak` |
| Active mode | `monitor_only` |
| Operator controls | `request_retraining` and `request_rollback` visible but disabled |
| Outcome row | `ml_agent_decision: no_intent / monitor_only_no_intent` |
| Slot text | `20 / 20` |
| Screenshot | `output/playwright/stage18-rl-soak-desktop.png` |

Focused local gates:

| Command | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_stage18_soak_incident_drills.py tests/unit/scripts/rl_trading/test_stage18_rl_soak_incident_drills.py` | `4 passed` |
| `uv run ruff check src/trading/contexts/rl_trading/domain/stage18_soak_incident_drills.py scripts/rl_trading/stage18_rl_soak_incident_drills.py tests/unit/contexts/rl_trading/domain/test_stage18_soak_incident_drills.py tests/unit/scripts/rl_trading/test_stage18_rl_soak_incident_drills.py` | passed |
| `uv run pyright src/trading/contexts/rl_trading/domain/stage18_soak_incident_drills.py scripts/rl_trading/stage18_rl_soak_incident_drills.py tests/unit/contexts/rl_trading/domain/test_stage18_soak_incident_drills.py tests/unit/scripts/rl_trading/test_stage18_rl_soak_incident_drills.py` | `0 errors` |

Prompt-level gates before report finalization:

| Command | Result |
|---|---|
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | `503 passed, 8 skipped, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed |
| `git diff --check` | passed |

The skipped tests are existing optional Torch gates in the default local
environment where `torch` is not installed. The warnings are existing `httpx`
deprecation warnings in app tests.

Artifact review:

| Field | Result |
|---|---|
| Review mode | Cold self-review fallback; independent subagent review was not used because current tool policy does not allow spawning subagents unless the user explicitly asks for delegation. |
| Verdict | accepted |
| Fixed blockers | none |
| Follow-up check | Rechecked docs index, `git diff --check`, prompt-level gates and ledger/plan handoff consistency. |
| Residual risks | 24h/7d full soak remains not claimed; `/opt/roehub/app` was not synced/restarted for Stage `18`; Stage `19+` remains blocked by `08N`. |

GitHub delivery for the implementation commit:

| Field | Value |
|---|---|
| Implementation commit | `83195a411d6995e5533c6f757ba6f68e8d5ead79` |
| GitHub CI | `28751386502` success |
| Mac Studio checkout | fast-forwarded to `83195a411d6995e5533c6f757ba6f68e8d5ead79` |
| `/opt/roehub/app` | not synced or restarted for Stage `18` because the accepted proof is main-checkout monitor-only harness evidence, not changed production runtime proof. |

## 4. Contract, safety, and delivery state

Contract impact:

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No public route or API payload changed. |
| Port contract | `none` | No application port/interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration/table/storage schema changed. |
| Config schema/defaults | `none` | Existing config was read; no committed defaults changed. |
| Request hash/cache key/persistence identity | `none` | No identity, cache or persisted hash semantics changed. |
| Operator CLI/test harness | `compatible-change` | Adds opt-in Stage `18` harness; existing commands unchanged. |
| Browser-visible behavior | `none` | No UI code changed; browser evidence verified existing fail-closed RL/ML state. |
| Logs/metrics/report semantics | `compatible-change` | Adds sanitized Stage `18` summary and stage report. |
| External side effects | `none` | No DB writes, Redis dispatch writes, exchange SDK calls, provider calls, paper/testnet/live/mainnet orders or order-state mutations. |
| Docs/runbooks | `compatible-change` | Adds Stage `18` report/runbook and updates plan/ledger/docs index. |

Safety and redaction:

- No secrets, tokens, cookies, passphrases, ciphertext, HMAC, API keys,
  provider payloads, raw signed requests, raw candle payload dumps, model
  checkpoint tensors or datasets were committed.
- Browser QA used a local sanitized harness and did not read
  `ROEHUB_SMOKE_E2E_PASSWORD`.
- Stage `18` summary stores sanitized hashes, counts, mode/state and booleans
  only.
- The ML worker/harness did not call exchange SDKs and did not resolve exchange
  secret custody.

## 5. Blockers and next-stage handoff

Stage `18` does not unlock Stage `19`.

| Next stage | Status | Reason |
|---|---|---|
| `19` Mainnet readiness architecture review | blocked | Stage `08N` recorded `stage19_mainnet_readiness_allowed=false`; Stage `18` was accepted only as technical monitor-only soak. |
| `20` Bounded mainnet canary | blocked | Requires accepted Stage `19`, explicit approval and `stage20_mainnet_canary_allowed=true`. |
| `21` Product rollout | blocked | Requires accepted Stage `20`, product/legal/support/backup gates and `stage21_product_rollout_allowed=true`. |

No executable next prompt is open from this stage. Future work requires either a
new accepted quality reclassification that changes the Stage `08N` downstream
booleans or an explicit new prompt that keeps the current fail-closed
product/mainnet boundary.
