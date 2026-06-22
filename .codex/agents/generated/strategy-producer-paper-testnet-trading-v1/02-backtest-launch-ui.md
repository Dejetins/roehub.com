---
prompt_name: 02-backtest-launch-ui
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Implement the user launch flow from current /backtests top variants to strategy/profile/run setup for paper/testnet."
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
      why: "previous stage status"
  task_entrypoints:
    - path: apps/api/routes/ui_backtests.py
      why: "backtest UI read model"
    - path: apps/api/routes/strategies.py
      why: "strategy/profile/run commands"
    - path: apps/web/dist/js/pages/backtests.js
      why: "backtest page UI"
    - path: apps/web/dist/js/pages/strategies.js
      why: "target strategy page UI"
skill_routing:
  - skill: browser-qa-evidence
    use_when: "verifying the launch flow in a real browser"
    timing: during verification
    reason: "UI changes require runtime browser proof"
  - skill: contract-impact-analysis
    use_when: "changing API DTOs or browser-visible defaults"
    timing: during implementation
    reason: "launch config is a public contract"
  - skill: publish-ci-deploy
    use_when: "stage is accepted and ready to deliver"
    timing: before ship
    reason: "delivery must include CI/deploy/runtime proof when changes ship"
validation_strategy:
  depth: browser_runtime
  e2e_required: true
  acceptance_surfaces: ["browser", "api", "database"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/02-backtest-launch-ui.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "02"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - apps/api/routes/ui_backtests.py
  - apps/api/routes/strategies.py
  - apps/api/dto/ui_backtests.py
  - apps/web/dist/js/pages/backtests.js
  - apps/web/dist/js/pages/strategies.js
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/02-backtest-launch-ui.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - src/trading/contexts/strategy/application/use_cases
  - tests/unit/apps/api
  - tests/unit/apps/web
  - docs/architecture/README.md
safety_notes:
  - "Default launch must not submit mainnet or live money orders."
  - "Use $50 allocation by default for this plan unless the user changes it later."
---

# Task

Implement the `/backtests` launch UX so a user can take a current UI/top variant and configure a strategy launch for `paper` or `testnet` with `BTCUSDT`, `$50` capital allocation, exchange connection selection, market type, entry sizing, risk mode, and direction.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `01` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `publish-ci-deploy` direct-main discipline for scoped publish on `main`; do not create a branch, draft PR, worktree, temporary checkout, local folder, stash, or auxiliary workflow artifact unless the user explicitly requests that exact workflow. Before marking the stage `accepted`, verify `origin/main` contains the changes and record main-branch delivery evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Do not invent unavailable variants. Use only variants visible through current API/UI and show fail-closed reasons for unsupported variants.
- Default to safe `paper` where appropriate; `testnet` requires selected owned exchange connection and readiness.
- `mainnet` must not be selectable for this plan.
- UI must not ask for API secret again; keys are managed only through `/settings`.
- Add stable API errors/readiness reasons for missing exchange, missing market readiness, unsupported variant, invalid sizing/risk/direction, and insufficient allocation/min notional.
- Update Russian stage report and ledger.

## Acceptance Criteria

- Playwright proves `/backtests` -> launch -> `/strategies` for at least one available variant.
- API/SQL proves created strategy/profile/run config has provenance, `BTCUSDT`, `$50`, mode, market type, sizing/risk/direction, and no secrets.
- Browser shows blocked reason for at least one unsupported or unready launch case.
- Local gates and docs index pass.

## Quality Gates

- `uv run ruff check apps/api apps/web src/trading/contexts/strategy tests`
- `uv run pyright apps/api src/trading/contexts/strategy tests`
- `uv run pytest -q tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/strategy`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report: files changed, API/DB/browser evidence, blocked cases, contract impact, delivery status, next-stage handoff.
