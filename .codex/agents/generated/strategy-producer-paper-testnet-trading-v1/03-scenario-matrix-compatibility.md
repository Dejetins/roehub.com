---
prompt_name: 03-scenario-matrix-compatibility
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Create durable scenario matrix coverage for available backtest variants: entry sizing, risk mode, direction, and readiness."
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
    - path: src/trading/contexts/strategy/application
      why: "strategy launch and compatibility use cases"
    - path: src/trading/contexts/backtest/domain/entities/execution_v1.py
      why: "backtest execution/sizing source contracts"
    - path: apps/api/routes/backtests.py
      why: "variant API surface"
    - path: apps/api/routes/ui_backtests.py
      why: "UI variant availability"
skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding matrix DTOs, migrations, or readiness contracts"
    timing: during implementation
    reason: "coverage matrix affects launch contract and persistence"
  - skill: publish-ci-deploy
    use_when: "accepted stage is ready to ship"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces: ["api", "database", "readiness-runtime"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/03-scenario-matrix-compatibility.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "03"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - src/trading/contexts/strategy/application
  - apps/api/routes/backtests.py
  - apps/api/routes/ui_backtests.py
  - alembic/versions
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/03-scenario-matrix-compatibility.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - tests/unit/contexts/strategy
  - tests/unit/apps/api
  - docs/architecture/README.md
safety_notes:
  - "Do not guess scenario combinations. Discover them from actual available variants/contracts."
---

# Task

Implement and prove a durable scenario matrix for the current user-visible backtest variants. The matrix must capture available `entry sizing`, `risk mode`, and `direction` combinations and readiness state for `paper` and `testnet` coverage.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `02` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `publish-ci-deploy` direct-main discipline for scoped publish on `main`; do not create a branch, draft PR, worktree, temporary checkout, local folder, stash, or auxiliary workflow artifact unless the user explicitly requests that exact workflow. Before marking the stage `accepted`, verify `origin/main` contains the changes and record main-branch delivery evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Discover combinations from current backtest/strategy contracts; do not hardcode imagined values.
- Persist or expose coverage state so later paper/testnet stages can prove every required branch.
- Mark unsupported/unready combinations with stable reason codes.
- Treat spot-short as an explicit blocked/unsupported branch unless a margin/borrow product is already implemented and proven; do not let later stages fake a real spot short.
- Keep scope to `BTCUSDT`, `paper`, and `testnet`; mainnet out of scope.
- Update docs and ledger.

## Acceptance Criteria

- Real API calls against top/available variants produce a concrete scenario matrix.
- SQL or durable artifact proves matrix rows and reason codes.
- Compatibility/readiness calls prove launchable/not-launchable behavior for representative rows.
- The matrix distinguishes real-order-capable futures short from blocked spot-short.
- Stage report states exact matrix rows covered and what remains blocked.

## Quality Gates

- `uv run ruff check src/trading/contexts/strategy src/trading/contexts/backtest apps tests`
- `uv run pyright src/trading/contexts/strategy src/trading/contexts/backtest apps tests`
- `uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/backtest tests/unit/apps`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with matrix table, evidence calls, contract impact, blockers, and next-stage handoff.
