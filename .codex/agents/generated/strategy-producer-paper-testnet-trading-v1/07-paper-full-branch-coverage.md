---
prompt_name: 07-paper-full-branch-coverage
repo: roehub.com
branch: main
scope: "Run and prove full paper coverage for all discovered entry sizing, risk mode, and direction scenarios with $50 allocation."
language:
  implementation: python/javascript
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
      why: "plan"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "stage handoff and matrix"
  task_entrypoints:
    - path: src/trading/contexts/live_execution/domain/paper_accounting.py
      why: "paper ledger/accounting domain"
    - path: src/trading/contexts/live_execution/application
      why: "execution source/risk/paper use cases"
    - path: apps/worker/strategy_live_runner
      why: "strategy producer runtime"
    - path: apps/web/dist/js/pages/strategies.js
      why: "paper status UI"
skill_routing:
  - skill: backend-quality-gates
    use_when: "running focused gates for paper/accounting changes"
    timing: during verification
    reason: "paper branch coverage needs deterministic local checks"
  - skill: browser-qa-evidence
    use_when: "verifying paper outcomes in UI"
    timing: during verification
    reason: "user-visible paper state must be observed"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth, stages scoped files, commits, pushes, and opens a draft PR"
  - skill: publish-ci-deploy
    use_when: "accepted changes need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: e2e
  e2e_required: true
  acceptance_surfaces: ["api", "database", "redis", "browser", "metrics"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/07-paper-full-branch-coverage.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "07"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - src/trading/contexts/live_execution
  - apps/worker/strategy_live_runner
  - apps/api/routes/ui_strategies_dashboard.py
  - apps/web/dist/js/pages/strategies.js
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/07-paper-full-branch-coverage.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - tests/unit/contexts/live_execution
  - tests/unit/contexts/strategy
  - tests/unit/apps
  - docs/architecture/README.md
safety_notes:
  - "Paper mode must not submit to exchanges or decrypt exchange credentials."
---

# Task

Run the full scenario matrix from Stage 03 through paper execution with `$50` allocation per strategy. Prove every discovered sizing/risk/direction branch either produces expected paper outcome or is blocked with stable reason.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `06` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, publish using `github:yeet`; do not mark the stage `accepted` until the stage report and ledger record main-branch delivery evidence and, for runtime/code stages, Mac Studio host sync/deploy smoke. Use `publish-ci-deploy` only for CI/deploy/host-sync work that `github:yeet` does not cover.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Use real strategy producer/API flow, not only direct unit calls.
- Persist paper source events, intents, paper orders/fills/accounting and coverage result per matrix row.
- Make fee/funding completeness explicit; do not pretend paper PnL is exact if funding/fees are modeled approximately.
- Cover spot-short as a branch, but keep its real-order capability marked blocked/unsupported unless a margin/borrow product exists.
- Prove no exchange submit, no exchange credential decrypt, and no mainnet path.
- Show UI paper status/outcomes on `/strategies`.

## Acceptance Criteria

- Stage report contains a coverage table for every matrix row.
- DB evidence proves source/intent/paper order/fill/accounting rows for successful rows.
- DB/API evidence proves unsupported spot-short is not later treated as a real spot testnet order.
- Redis/execution stream proof shows no real exchange dispatch for paper-only rows unless explicitly shadowed as non-submit.
- Browser shows paper position/PnL/outcome and blocked reasons.

## Quality Gates

- `uv run ruff check src/trading/contexts/live_execution src/trading/contexts/strategy apps tests`
- `uv run pyright src/trading/contexts/live_execution src/trading/contexts/strategy apps tests`
- `uv run pytest -q tests/unit/contexts/live_execution tests/unit/contexts/strategy tests/unit/apps`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with full matrix evidence, paper accounting facts, UI proof, delivery status, and handoff.
