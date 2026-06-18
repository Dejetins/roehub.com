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

Этот отчет публикует standalone-контракт Stage `02C`: action ids, state extras, action-history encoding, reward v1 и strategy-owned close semantics. Stage `02B` feature/live-feed contract доставлен отдельным sync repair-коммитом, поэтому Stage `02C` больше не заблокирован prerequisite gap.

Stage ledger продвинут на `current_stage=03` после delivery Stage `02B` и повторной проверки Stage `02C` focused gates.

## Scope

Included:

- add the internal `rl_trading` domain package entrypoints required by this contract;
- freeze action ids `0/1/2/3` as `hold`, `open_long`, `open_short`, `close`;
- implement strategy-owned action resolution for `owner_user_id + strategy_run_id + exchange + market_type + symbol`;
- preserve no-pyramiding and no-cross-strategy-close behavior through deterministic domain fixtures;
- freeze external-repo-compatible state extras and action-history encoding;
- freeze training reward v1: `pnl_change / initial_balance - flat-hold inaction penalty`;
- document that live/paper/testnet outcomes are execution-ledger outcomes, not the training reward source of truth.

Not included:

- exchange, paper, testnet, live or mainnet side effects;
- API, UI, Redis, ClickHouse, Postgres, config or migration changes;
- runtime ML artifacts under `/opt/roehub/state/rl_trading/`;
- Stage `03` ML runtime/environment work.

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/__init__.py`; `src/trading/contexts/rl_trading/domain/__init__.py`; `src/trading/contexts/rl_trading/domain/action_state_reward_contract.py`; `tests/unit/contexts/rl_trading/domain/test_action_state_reward_contract.py`; this report | - | - | Add standalone Stage `02C` action/state/reward contract and deterministic fixtures without Stage `02B` files. | `compatible-change` new internal/domain contract and docs/report |
| - | `docs/architecture/README.md` | - | Docs index update after adding this Stage `02C` report. | `compatible-change` docs index only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`; this report | - | Accept Stage `02C` after Stage `02B` delivery and repeated focused verification during repository sync repair. | `compatible-change` docs/ledger only |

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
| Logs/metrics/traces/audit/ledger/report/redaction | `compatible-change` | Adds sanitized Stage `02C` report evidence; no secrets/provider payloads. |
| Benchmark / rollout gate | `compatible-change` | Stage `02C` acceptance opens Stage `03` after Stage `02B` delivery and focused domain verification. |
| Performance hot path | `none` | No runtime path is called by API/worker/execution code in this stage. |
| Browser-visible behavior | `none` | No UI/browser behavior changed. |
| Docs/runbooks | `compatible-change` | Stage report and docs index updated only. |

## Quality Gates

| Gate | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_action_state_reward_contract.py` | passed; `8 passed` |
| `uv run ruff check src/trading/contexts/rl_trading tests/unit/contexts/rl_trading` | passed |
| `uv run pyright src/trading/contexts/rl_trading tests/unit/contexts/rl_trading` | passed; `0 errors` |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |
| 2026-06-18 acceptance repair focused gate: `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_feature_contract.py tests/unit/contexts/rl_trading/domain/test_action_state_reward_contract.py tests/unit/contexts/market_data/adapters/test_redis_streams_live_candle_publisher.py tests/unit/contexts/strategy/adapters/test_redis_strategy_live_candle_stream.py` | passed; `24 passed` |
| 2026-06-18 acceptance repair publish gates: `uv run ruff check .`; `uv run pyright`; `uv run pytest -q -ra`; `python -m tools.docs.generate_docs_index --check` | passed; `1215 passed, 3 warnings`; pyright `0 errors`; docs index up to date |

## Evidence

| Acceptance surface | Evidence |
|---|---|
| Domain fixtures | Tests cover stable hash, action ids, action history and state extras. |
| Ownership invariants | Tests prove close is scoped to `owner_user_id + strategy_run_id + exchange + market_type + symbol`, same-ticker different-strategy positions cannot be closed, and non-owned positions do not create a false no-pyramiding block. |
| No-pyramiding | Tests prove repeated same-side and opposite-side open while already open become `no_intent` with `strategy_position_already_open`. |
| Reward compatibility | Tests prove open fee reward, flat-hold penalty, open-position hold reward `0.0`, close realized PnL minus fee, and last-step open/close coercion. |
| Boundary safety | No execution adapter, exchange SDK, secret custody, runtime worker, API route, browser, Redis, ClickHouse, Postgres or migration path changed. |
| Secrets/artifacts | No secrets, tokens, cookies, ciphertext, raw provider payloads, raw signed requests, checkpoint tensors, datasets or runtime ML artifacts were written. |

Tests are the correct real-boundary evidence for this scoped commit because the changed boundary is a pure internal `rl_trading` domain contract. There is no runtime, API, persistence, browser, exchange, or provider surface to smoke without starting a later stage.

## Next-Stage Handoff

Stage `02C` is accepted in the ledger. Stage `03` can start after confirming this report and the ledger are present on `main`.

The Stage `03` executor should know:

- Stage `02C` contract hash: `255d765b9474620671167412465fc55a058c0233d5da242a276143fb6816b557`.
- Action ids are fixed: `0 hold`, `1 open_long`, `2 open_short`, `3 close`.
- Roehub ownership scope is `owner_user_id + strategy_run_id + exchange + market_type + symbol`.
- No-pyramiding is scoped to this exact RL strategy position only.
- Close cannot affect another strategy run's same-ticker position.
- Training reward v1 remains external-repo-compatible and is not a live outcome source of truth.
- User-specific paper/testnet/live outcomes are monitoring/evaluation data only in v1, not platform-wide retraining input.
- No runtime ML artifacts were created; future artifacts still belong under `/opt/roehub/state/rl_trading/` outside git.
