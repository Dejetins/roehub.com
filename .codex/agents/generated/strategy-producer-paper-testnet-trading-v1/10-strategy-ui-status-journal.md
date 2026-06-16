---
prompt_name: 10-strategy-ui-status-journal
repo: roehub.com
branch: main
scope: "Complete /strategies UI status, market context, latest signals, execution outcomes, and manual controls."
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
      why: "stage handoff"
  task_entrypoints:
    - path: apps/api/routes/ui_strategies_dashboard.py
      why: "dashboard read model"
    - path: apps/api/dto/ui_strategies_dashboard.py
      why: "dashboard DTO"
    - path: apps/web/dist/js/pages/strategies.js
      why: "strategy page UI"
    - path: apps/web/templates
      why: "rendered page templates if applicable"
skill_routing:
  - skill: browser-qa-evidence
    use_when: "verifying desktop/mobile UI, console, network, and screenshots"
    timing: during verification
    reason: "UI acceptance must be browser-observed"
  - skill: ui-ux-pro-max
    use_when: "improving layout/interaction clarity without changing product scope"
    timing: during implementation
    reason: "UI needs operational clarity and no overlap"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth, stages scoped files, commits, pushes, and opens a draft PR"
  - skill: publish-ci-deploy
    use_when: "accepted UI/API changes need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: browser_runtime
  e2e_required: true
  acceptance_surfaces: ["browser", "api", "network-console", "dom-secret-scan"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/10-strategy-ui-status-journal.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "10"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - apps/api/routes/ui_strategies_dashboard.py
  - apps/api/dto/ui_strategies_dashboard.py
  - apps/web/dist/js/pages/strategies.js
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/10-strategy-ui-status-journal.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - tests/unit/apps/api
  - tests/unit/apps/web
  - docs/architecture/README.md
safety_notes:
  - "UI must not expose secrets or raw provider payloads."
---

# Task

Complete `/strategies` UI for this cycle: strategy block must show market, exchange, environment, producer status, allocation, readiness, latest signal, source event, intent, order/fill/reconciliation outcome, latency gap, and manual entry/exit controls.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `09` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, publish using `github:yeet`; do not mark the stage `accepted` until the stage report and ledger record main-branch delivery evidence and, for runtime/code stages, Mac Studio host sync/deploy smoke. Use `publish-ci-deploy` only for CI/deploy/host-sync work that `github:yeet` does not cover.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Keep UI work-focused and dense; avoid marketing/landing-page patterns.
- Show `testnet`/`paper` clearly and keep mainnet unavailable.
- Include filters/status for current strategy run and latest events.
- Display blocked/unknown states honestly; do not style mismatch as success.
- Run desktop and mobile browser QA with console/network checks and DOM secret scan.

## Acceptance Criteria

- Playwright proves `/strategies` status/journal for a paper strategy and a testnet strategy or exact blocker if testnet keys are unavailable.
- Network requests return expected 2xx/controlled 4xx; no console errors.
- UI shows market/exchange/environment and execution outcome links.
- DOM/screenshot scan contains no secrets/cookies/tokens/API keys.

## Quality Gates

- `node --check apps/web/dist/js/pages/strategies.js`
- `uv run ruff check apps tests`
- `uv run pyright apps tests`
- `uv run pytest -q tests/unit/apps`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with screenshots/artifact paths, API/browser evidence, UI contract impact, delivery status, and handoff.
