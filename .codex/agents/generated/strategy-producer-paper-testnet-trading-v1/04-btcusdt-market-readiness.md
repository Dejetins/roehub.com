---
prompt_name: 04-btcusdt-market-readiness
repo: roehub.com
branch: main
scope: "Prove BTCUSDT market-data readiness for Binance/Bybit spot/futures before strategy producer execution."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
      why: "plan"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "stage handoff"
  task_entrypoints:
    - path: src/trading/contexts/market_data
      why: "market readiness and reference data"
    - path: apps/worker/market_data_ws
      why: "live candle publishing"
    - path: apps/api/routes/market_data_reference.py
      why: "market/instrument reference API"
    - path: src/trading/contexts/strategy/application/ports/market_data_readiness.py
      why: "strategy launch readiness dependency"
skill_routing:
  - skill: root-cause-debugging
    use_when: "market stream freshness or reference data is missing unexpectedly"
    timing: if blocker
    reason: "readiness failures need evidence-based localization"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth, stages scoped files, commits, pushes, and opens a draft PR"
  - skill: publish-ci-deploy
    use_when: "accepted changes need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["redis", "clickhouse-or-reference-db", "api", "browser"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/04-btcusdt-market-readiness.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "04"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - src/trading/contexts/market_data
  - src/trading/contexts/strategy
  - apps/api/routes/market_data_reference.py
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/04-btcusdt-market-readiness.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - apps/worker/market_data_ws
  - tests/unit/contexts/market_data
  - tests/unit/apps/api
  - docs/architecture/README.md
safety_notes:
  - "Only BTCUSDT is in scope. Do not silently broaden to other instruments."
---

# Task

Make BTCUSDT market readiness explicit and proven for the strategy producer cycle across Binance/Bybit and spot/futures where current artifacts exist. Missing non-BTCUSDT artifacts are out of scope.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `03` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, publish using `github:yeet`; do not mark the stage `accepted` until the stage report and ledger record main-branch delivery evidence and, for runtime/code stages, Mac Studio host sync/deploy smoke. Use `publish-ci-deploy` only for CI/deploy/host-sync work that `github:yeet` does not cover.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Check/provision only `BTCUSDT` readiness needed by this plan.
- Verify live candle freshness, reference instrument data, precision/min-notional readiness, and strategy launch readiness.
- If a required exchange/market artifact is missing, record it as a blocker for later stages instead of guessing.
- Update UI/API readiness where needed so users see market status on `/strategies`.

## Acceptance Criteria

- Redis stream readiness evidence for `BTCUSDT` closed candles is recorded.
- Reference data/API evidence exists for Binance/Bybit spot/futures if currently available.
- Browser or API shows market readiness/missing/stale reason.
- Stage report records exact keys, calls, and freshness thresholds.

## Quality Gates

- `uv run ruff check src/trading/contexts/market_data src/trading/contexts/strategy apps tests`
- `uv run pyright src/trading/contexts/market_data src/trading/contexts/strategy apps tests`
- `uv run pytest -q tests/unit/contexts/market_data tests/unit/contexts/strategy tests/unit/apps`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with BTCUSDT readiness matrix, real Redis/API/DB evidence, blockers, and handoff.
