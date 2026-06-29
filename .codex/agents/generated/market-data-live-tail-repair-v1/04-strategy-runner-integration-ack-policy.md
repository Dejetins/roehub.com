---
prompt_name: 04-strategy-runner-integration-ack-policy
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
scope: "Integrate ClosedCandleTailProvider into StrategyLiveRunner and prove ACK/deferred retry semantics for live gaps."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1.md
      why: "source plan"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
      why: "stage gate"
    - path: docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
      why: "current Strategy runner contract to update"
  task_entrypoints:
    - path: src/trading/contexts/strategy/application/services/live_runner.py
      why: "current checkpoint and gap repair logic"
      inspect_symbols: ["run_once", "_process_candle", "_repair_gap"]
    - path: src/trading/contexts/strategy/application/ports/live_candle_stream.py
      why: "stream read/ack contract"
    - path: src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_streams_live_candle_stream.py
      why: "Redis consumer group ACK/pending behavior"
    - path: apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      why: "runtime wiring"
skill_routing:
  - skill: backend-quality-gates
    use_when: "running focused Python checks"
    timing: "after implementation"
    reason: "Stage 04 changes Strategy runner behavior"
  - skill: root-cause-debugging
    use_when: "gap retry or ACK proof fails"
    timing: "if blocker"
    reason: "this stage directly addresses the Stage 12.4 root cause"
  - skill: publish-ci-deploy
    use_when: "accepted changes need delivery"
    timing: "before final report"
    reason: "direct-main delivery required"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces: ["strategy-runner", "redis-pending-or-backlog", "database", "dedupe"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/04-strategy-runner-integration-ack-policy.md
proof_boundary:
  required_when: "Production runtime proof deferred to Stage 06"
  label: none
  changed_code_production_claim_allowed: false
stage_execution_ledger:
  path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  plan_doc: docs/architecture/market_data/market-data-live-tail-repair-v1.md
  current_stage: "04"
  required_update: true
expected_primary_touches:
  - src/trading/contexts/strategy/application/services/live_runner.py
  - src/trading/contexts/strategy/application/ports
  - src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_streams_live_candle_stream.py
  - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
  - configs/prod/strategy.yaml
  - tests/unit/contexts/strategy
  - tests/integration
  - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/04-strategy-runner-integration-ack-policy.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/market_data/market-data-live-tail-repair-v1.md
  - docs/architecture/README.md
safety_notes:
  - "The runner must not directly import Binance/Bybit REST adapters."
  - "Checkpoint advances only after continuous candles are processed in order."
---

# Task

Implement Stage `04` Strategy runner integration and ACK policy.

## Requirements (Must)

- Verify Stage `03 accepted` in the repair ledger.
- Replace ClickHouse-only `_repair_gap` dependency with `ClosedCandleTailProvider`.
- Keep checkpoint source of truth as `strategy_runs.checkpoint_ts_open`.
- Before code changes, choose and record one concrete policy:
  - pending reclaim with no ACK until checkpoint accepts current candle; or
  - durable repair backlog plus ACK only after hot-cache materialization proof.
- The chosen policy must prove no future candle loss if repair fails once and later succeeds.
- If Redis pending reclaim is chosen, implement the required stream port/adapter behavior and tests.
- If durable backlog is chosen, implement the required persistence and replay behavior and tests.
- Process repaired candles in strict `ts_open` order before the triggering current candle.
- Ensure duplicate `StrategySignal` and duplicate `(strategy_run_id, bar_ts_open)` remain impossible.
- Update `strategy-live-runner-redis-streams-v1.md` with the new repair and ACK semantics.
- Update report and ledger after validation.
- If accepted and files changed, deliver through `publish-ci-deploy`.

## Non-Goals

- Do not change strategy evaluator behavior.
- Do not submit orders or run real exchange execution.
- Do not start 6h soak.

## Acceptance Criteria

- Direct runner test/call proves normal contiguous candle still advances checkpoint.
- Gap test with provider success proves missing candle(s) are processed before the triggering candle and checkpoint reaches triggering candle.
- Failed-repair test proves triggering/future candle is not lost; a later retry after provider recovery processes the complete range.
- Dedupe proof shows duplicate `signal_id` and duplicate `(strategy_run_id, bar_ts_open)` stay `0`.
- Real Redis or integration-level pending/backlog proof matches the chosen ACK policy.
- Ledger marks `04 accepted`.

## Quality Gates

- `uv run ruff check src/trading/contexts/strategy src/trading/contexts/market_data apps/worker/strategy_live_runner tests`
- `uv run pyright src/trading/contexts/strategy src/trading/contexts/market_data apps/worker/strategy_live_runner tests`
- `uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/market_data tests/integration`
- `uv run python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with selected ACK policy, no-loss retry proof, checkpoint evidence, dedupe evidence, file manifest, delivery status, and Stage `05` handoff.
