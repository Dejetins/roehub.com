---
doc: rl-trading-agent-platform-v1-stage-02c-action-state-reward-contract
stage: "02C"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-18"
---

# Stage 02C: Action, State And Reward Contract

Статус: `accepted`.

Stage `02C` freezes the Roehub RL action/state/reward contract and strategy-owned close semantics. Stage does not train models, create datasets, add PyTorch/cloud/model hosting, change public API or persistence schemas, submit exchange orders, or change secret custody.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

## Scope

Included:

- verify prerequisite Stage `02B`;
- freeze action ids `0/1/2/3` as `hold`, `open_long`, `open_short`, `close`;
- implement strategy-owned action resolution for `owner_user_id + strategy_run_id + exchange + market_type + symbol`;
- preserve no-pyramiding/no-cross-strategy-close behavior through deterministic domain fixtures;
- freeze external-repo-compatible state extras and action-history encoding;
- freeze training reward v1: `pnl_change / initial_balance - flat-hold inaction penalty`;
- document that live/paper/testnet outcomes are execution-ledger outcomes, not the training reward source of truth;
- record prompt path/hash, file manifest, evidence, contract impact, delivery state and next-stage handoff.

Not included:

- exchange, paper, testnet, live or mainnet side effects;
- live-execution submit/risk-gate behavior changes;
- API, UI, Redis, ClickHouse, Postgres, config or migration changes;
- runtime ML artifacts under `/opt/roehub/state/rl_trading/`;
- user-owned model training or platform-wide retraining from user-specific live outcomes.

## File Manifest

Planned concrete file list narrowed before code edits:

- `src/trading/contexts/rl_trading/domain/action_state_reward_contract.py`
- `src/trading/contexts/rl_trading/domain/__init__.py`
- `tests/unit/contexts/rl_trading/domain/test_action_state_reward_contract.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02c-action-state-reward-contract.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if `python -m tools.docs.generate_docs_index --check` required index regeneration

Procedural deviation record:

| Field | Value |
|---|---|
| `rule` | Prompt requirement to record the narrowed concrete file list in this stage report before editing. |
| `reason` | The list was narrowed before code edits in the execution notes, but this report file was created after the first code patch. |
| `risk` | Traceability order is less strict than the prompt requested; code safety, runtime safety and contract compatibility are not affected. |
| `recovery_path` | This report records the narrowed list, final implemented manifest, contract impact and gate evidence. Future stages should create or update the stage-report stub before the first implementation patch. |

Implemented file manifest:

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/action_state_reward_contract.py`; `tests/unit/contexts/rl_trading/domain/test_action_state_reward_contract.py`; this report | - | - | Freeze Stage `02C` action/state/reward contract and deterministic fixtures. | `compatible-change` new internal/domain contract and docs/report |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export the Stage `02C` contract alongside the accepted Stage `02B` feature contract. | `compatible-change` additive internal exports |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark Stage `02C` accepted, record evidence and open Stage `03`. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Docs index update after adding Stage `02C` report; generator also indexed pre-existing uncommitted Stage `02B` and strategy-producer Stage `09` docs already present in the worktree. | `compatible-change` docs index only |

Outside expected paths: none for authored Stage `02C` code/report/ledger changes. `docs/architecture/README.md` is an allowed docs-index touch; its generated diff also includes pre-existing uncommitted docs that were not authored by this stage.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/02c-action-state-reward-contract.md` |
| Prompt sha256 | `49ece6daa1124cd6ab25c7983a418b799bc2a8bcc483f40883c1e1a3f31b47c7` |
| Ledger state before implementation | Stage `02B` accepted; `current_stage=02C`; Stage `02C` pending |
| Required prerequisite | Stage `02B` accepted |
| Optional compact state | `.codex/agents/.context/promt_manager_state.yaml` refers to this RL plan but is stale: it still says Stage `02A` had not started. Current ledger and Stage `02B` report were used instead. |
| External repo check | Upstream `YuriyKolesnikov/rl-trading-binance/trading_environment.py` was checked for action/reward semantics before implementation. |
| Delivery state | `local-only`; no branch, PR, main delivery, deploy, runtime service, schema, API, UI, exchange, Redis, ClickHouse, Postgres or ML artifact side effect |
| Large artifacts | No datasets, checkpoints, raw provider payloads, logs or runtime ML artifacts were written to git or `/opt/roehub/state/rl_trading/`. |

## Observed State

| Area | Evidence summary |
|---|---|
| Stage prerequisite | Ledger records Stage `02B` as `accepted`; Stage `02C` is the current stage. |
| Feature contract dependency | Stage `02B` accepted feature contract hash `d2e99786b68482d730494c6aeec72a1e9f40ac225729019fac5c82f96f900be9`; Stage `02C` does not modify it. |
| Execution source | `live_execution` already supports `source_type="ml_agent_decision"` and validates sanitized source refs. |
| Strategy ownership anchor | `StrategyRun` already carries `run_id` and `user_id`; Stage `02C` uses the same identity as the RL ownership scope. |
| Risk gate | Existing `ml_agent_decision` risk branch requires account context and `ml_agent_policy_active`; Stage `02C` does not bypass or change it. |
| External repo behavior | Training environment uses actions `0 hold`, `1 open long`, `2 open short`, `3 close`, flat-hold inaction penalty, no mark-to-market reward for open-position hold, and last-step open/close coercion. |

## Action/State/Reward Contract

| Field | Value |
|---|---|
| Contract id | `rl_trading.action_state_reward.roehub_v1` |
| Schema version | `1` |
| Contract hash | `255d765b9474620671167412465fc55a058c0233d5da242a276143fb6816b557` |
| Ownership scope | `owner_user_id + strategy_run_id + exchange + market_type + symbol` |
| Runtime artifact root | `/opt/roehub/state/rl_trading/` |
| Source event type | `ml_agent_decision` |

Action semantics:

| Action id | Name | Roehub intent behavior |
|---:|---|---|
| `0` | `hold` | Always `no_intent`; records decision/source-event/status only. |
| `1` | `open_long` | Creates an open-long intent only when this exact RL strategy scope has no open position. |
| `2` | `open_short` | Creates an open-short intent only when this exact RL strategy scope has no open position; later execution/risk/account gates still decide whether shorting is supported. |
| `3` | `close` | Closes only the position owned by this exact RL strategy scope. It cannot close another strategy's same-ticker position. |

No-pyramiding and ownership behavior:

- repeated same-side open while already open becomes `no_intent` with audit reason `strategy_position_already_open`;
- opposite-side open before close also becomes `no_intent` with audit reason `strategy_position_already_open`;
- close without a strategy-owned position becomes `no_intent` with audit reason `no_strategy_position`;
- a same-user/same-ticker position from another `strategy_run_id`, market, or symbol does not block this strategy scope and cannot be closed by this strategy.

State extras:

| Field | Meaning |
|---|---|
| `position` | `1.0` long, `-1.0` short, `0.0` flat |
| `unrealized` | `(current_price - entry_price) * position / entry_price`; this is an observation feature, not reward for holding. |
| `time_elapsed` | `step_idx / session_len` |
| `time_remaining` | `(session_len - step_idx) / session_len` |
| action history | one-hot encoded last-N action ids, four slots per history item |

Reward v1:

- opening action applies entry fee as negative `pnl_change`;
- closing action realizes trade PnL minus close fee;
- flat hold applies `inaction_penalty_ratio`;
- hold while a position is open keeps reward `0.0` when there is no realized PnL, even if price moves;
- reward formula is `pnl_change / initial_balance - inaction_penalty`;
- last training step blocks new opens by coercing to hold, and forces close of an existing position unless action is already close;
- reward is not rewritten to a Roehub risk score in this stage.

Backtest/live distinction:

- offline training reward is only the training environment reward;
- offline evaluation/backtest must later compute scorecard metrics from realized outcomes and declared fee/slippage/funding policy;
- paper/testnet/live outcome source of truth is execution order/fill/reconciliation ledgers, not training reward;
- user-specific paper/testnet/live outcomes do not enter platform-wide retraining in v1.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No routes or response payloads changed. |
| Port contract | `none` | No existing Python port/protocol signature changed. |
| DTO schema | `none` | No existing DTO or wire payload changed. |
| Persisted schema | `none` | No migration, table or storage schema changed. |
| Config schema/defaults | `none` | No env/YAML/default changed. |
| Request hash / cache key / persistence identity | `none` | Existing runtime identities are unchanged; new RL scope identity is internal/domain-only. |
| Service-call auth/timeout/retry/error semantics | `none` | No service call behavior changed. |
| External side effects / idempotency / unknown-state semantics | `none` | No exchange, paper, testnet, mainnet, provider or durable side effect was added. |
| Logs/metrics/traces/audit/ledger/report/redaction | `compatible-change` | Adds sanitized Stage `02C` report/ledger evidence; no secrets/provider payloads. |
| Alert/runbook semantics | `none` | No alerting or runbook trigger changed. |
| Benchmark / rollout gate | `compatible-change` | Stage `02C` acceptance opens Stage `03`; later execution stages remain gated. |
| Performance hot path | `none` | No runtime path is called by API/worker/execution code in this stage. |
| Browser-visible behavior | `none` | No UI/browser behavior changed; prompt disabled browser runtime verification. |
| Docs/runbooks | `compatible-change` | Stage report, ledger and docs index updated only. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/02c-action-state-reward-contract.md` | passed; `49ece6daa1124cd6ab25c7983a418b799bc2a8bcc483f40883c1e1a3f31b47c7` |
| Focused `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_action_state_reward_contract.py` | passed; `8 passed` |
| Focused `uv run ruff check src/trading/contexts/rl_trading/domain/action_state_reward_contract.py tests/unit/contexts/rl_trading/domain/test_action_state_reward_contract.py` | passed |
| Focused `uv run pyright src/trading/contexts/rl_trading tests/unit/contexts/rl_trading` | passed; `0 errors` |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `334 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after `python -m tools.docs.generate_docs_index` updated `docs/architecture/README.md` |

## Evidence

| Acceptance surface | Evidence |
|---|---|
| Domain fixtures | `test_action_state_reward_contract_hash_and_required_literals_are_stable`, action history and state-extra tests passed. |
| Ownership invariants | Tests prove close is scoped to `owner_user_id + strategy_run_id + exchange + market_type + symbol`, same-ticker different-strategy positions cannot be closed, and non-owned positions do not create a false no-pyramiding block. |
| No-pyramiding | Tests prove repeated same-side and opposite-side open while already open become `no_intent` with `strategy_position_already_open`. |
| Reward compatibility | Tests prove open fee reward, flat-hold penalty, open-position hold reward `0.0`, close realized PnL minus fee, and last-step open/close coercion. |
| Boundary safety | No execution adapter, exchange SDK, secret custody, runtime worker, API route, browser, Redis, ClickHouse, Postgres or migration path was changed. |
| Secrets/artifacts | No secrets, tokens, cookies, ciphertext, raw provider payloads, raw signed requests, checkpoint tensors, datasets or runtime ML artifacts were written. |

Tests are the correct real-boundary evidence for this stage because the changed boundary is a pure internal `rl_trading` domain contract. There is no runtime, API, persistence, browser, exchange, or provider surface to smoke without starting a later stage.

## Cold Self-Review

Mode: `cold self-review fallback`.

Reason: subagent tooling is available in the environment but the tool contract permits spawning subagents only after an explicit user request. Repository Stage `01` used the same fallback for this prompt family.

Checklist result:

- architecture/stage ledger continuity: `Release`;
- traceability: `Release with recorded procedural deviation`;
- validation depth: `Release`;
- service-call, retry, idempotency, redaction, alerts: `N/A` because no runtime/service side effect changed;
- Mac Studio path contract: `Release`; no Mac Studio runtime work or `/opt/roehub/state/rl_trading/` artifact was needed;
- browser auth/tooling: `N/A`; browser runtime verification disabled by prompt and no UI changed.

No Blocker/High findings remain after local follow-up checks.

## Blockers And Residual Risks

| Item | Status | Next action |
|---|---|---|
| Stage `02C` acceptance | No blocker | Stage accepted locally with deterministic domain tests, backend gates, report and ledger update. |
| Procedural stage-report timing | Recorded deviation | Future stages should create/update report stub before implementation patches. |
| Futures metadata gate | Residual fail-closed gate from Stage `02B` | Later stages must source or explicitly approximate funding, mark/index, filters, leverage tiers, fee, slippage and liquidation assumptions before production-grade futures evaluation/activation. |
| RL execution stages | Still blocked later | Stage `15`/`16` still depend on classic producer Stage `07`/`09` after Stage `05` repair. |
| Delivery | `local-only` | No branch, PR, main delivery, CI, Mac Studio deploy or runtime service proof was performed because this stage changed no deployed runtime state and did not request publishing. |

## Next-Stage Handoff

Stage `03` is allowed to start from this local accepted contract.

It must know:

- Stage `02C` contract hash: `255d765b9474620671167412465fc55a058c0233d5da242a276143fb6816b557`.
- Action ids are fixed: `0 hold`, `1 open_long`, `2 open_short`, `3 close`.
- Roehub ownership scope is `owner_user_id + strategy_run_id + exchange + market_type + symbol`.
- No-pyramiding is scoped to this exact RL strategy position only.
- Close cannot affect another strategy run's same-ticker position.
- Training reward v1 remains external-repo-compatible and is not a live outcome source of truth.
- User-specific paper/testnet/live outcomes are monitoring/evaluation data only in v1, not platform-wide retraining input.
- No runtime ML artifacts were created; future artifacts still belong under `/opt/roehub/state/rl_trading/` outside git.
