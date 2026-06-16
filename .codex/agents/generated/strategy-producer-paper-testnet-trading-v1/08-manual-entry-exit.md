---
prompt_name: 08-manual-entry-exit
repo: roehub.com
branch: main
scope: "Add manual entry and manual stop/exit controls that use the same execution source-event path."
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
    - path: apps/api/routes/strategies.py
      why: "strategy commands"
    - path: apps/api/routes/ui_execution.py
      why: "execution UI API"
    - path: src/trading/contexts/live_execution/domain/execution_source.py
      why: "manual_request source contract"
    - path: apps/web/dist/js/pages/strategies.js
      why: "manual controls UI"
skill_routing:
  - skill: browser-qa-evidence
    use_when: "verifying manual buttons in UI"
    timing: during verification
    reason: "manual entry/exit is browser-visible"
  - skill: contract-impact-analysis
    use_when: "adding manual action API DTOs"
    timing: during implementation
    reason: "manual commands are public API contracts"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth and scoped staging; branches/PRs are temporary and must be delivered to main and cleaned up before acceptance"
  - skill: publish-ci-deploy
    use_when: "accepted changes need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: browser_runtime
  e2e_required: true
  acceptance_surfaces: ["browser", "api", "database", "redis", "metrics"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/08-manual-entry-exit.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "08"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - apps/api/routes/strategies.py
  - apps/api/routes/ui_execution.py
  - src/trading/contexts/live_execution
  - apps/web/dist/js/pages/strategies.js
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/08-manual-entry-exit.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - tests/unit/apps/api
  - tests/unit/apps/web
  - tests/unit/contexts/live_execution
  - docs/architecture/README.md
safety_notes:
  - "Manual actions must use the same source-event/risk/dispatch path as strategies."
  - "Manual buttons must be paper/testnet only in this plan."
---

# Task

Add separate manual entry and manual stop/exit controls on `/strategies`. They must create `manual_request` source events and then use the same risk, intent, dispatch, order, reconciliation, and outbox path as strategy signals.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `07` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `github:yeet`/`publish-ci-deploy` discipline for scoped publish, but do not leave the stage on a per-stage branch or draft PR. Temporary branches are allowed only when useful; before marking the stage `accepted`, deliver the changes to `main`, push `origin main`, verify main contains the changes, delete any temporary local/remote branch, and record main-branch delivery plus branch-cleanup evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Add idempotent manual entry and manual exit commands.
- Respect profile mode, allowlist, market readiness, account config, capital reservation, ownership lock, and kill switch.
- Show pending/accepted/rejected/unknown outcomes in UI.
- Duplicate clicks must not create duplicate money-moving orders.
- Mainnet remains unavailable.

## Acceptance Criteria

- Playwright clicks manual entry and manual exit in paper mode and records UI outcomes.
- Testnet-safe representative manual action is proven if Stage 05 keys/config allow it; otherwise exact blocker recorded.
- DB proves source event, intent, risk result, order/paper order, outbox rows.
- Redis/metrics show expected dispatch or non-dispatch behavior.

## Quality Gates

- `uv run ruff check apps src/trading/contexts/live_execution tests`
- `uv run pyright apps src/trading/contexts/live_execution tests`
- `uv run pytest -q tests/unit/apps tests/unit/contexts/live_execution`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with UI/API/DB evidence, duplicate/idempotency proof, blocked cases, and handoff.
