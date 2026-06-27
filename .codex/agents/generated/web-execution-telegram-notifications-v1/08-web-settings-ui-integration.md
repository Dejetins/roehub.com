---
prompt_name: "Notifications v1 Stage 08 - Web Settings UI Integration"
repo: "roehub.com"
branch: "main"
scope: "Expose Telegram binding status, scoped notification modes and report schedule in settings UI/API"
language:
  implementation: "python/javascript/html"
  agent_report: "ru"
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
stage_execution_ledger:
  path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
  plan_doc: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
  current_stage: "08"
  required_update: true
validation_strategy:
  depth: "browser_runtime"
  acceptance_surfaces: ["API", "browser", "settings UI"]
proof_boundary:
  label: "read_only_existing_runtime_smoke"
  changed_code_production_claim_allowed: false
user_presence_required: "required only if smoke auth env is missing or user wants manual visual confirmation"
runtime_env_sources:
  report_only_key_presence: true
  optional_keys:
    - "ROEHUB_SMOKE_E2E_PASSWORD"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "browser auth and secret policy"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main/access contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "settings and preference contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage gate"
  task_entrypoints:
    - path: "apps/api/routes/ui_account.py"
      why: "settings API route pattern"
    - path: "apps/api/dto/ui_account.py"
      why: "settings DTO pattern"
    - path: "apps/web/templates/"
      why: "settings UI templates"
    - path: "apps/web/dist/"
      why: "shipped browser assets pattern"
skill_routing:
  - skill: "browser-qa-evidence"
    timing: "during verification"
    reason: "settings UI is browser-visible"
  - skill: "contract-impact-analysis"
    timing: "before implementation"
    reason: "existing account notification DTO must remain compatible"
  - skill: "backend-quality-gates"
    timing: "during verification"
    reason: "API/UI route tests"
expected_primary_touches:
  - "apps/api/routes/"
  - "apps/api/dto/"
  - "apps/web/templates/"
  - "apps/web/dist/"
  - "apps/web/locales/"
  - "tests/unit/apps/api/"
  - "tests/unit/apps/web/"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/08-web-settings-ui-integration.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "apps/web/main/"
  - "docs/architecture/README.md"
quality_gates:
  - "uv run ruff check apps/api apps/web tests/unit/apps"
  - "uv run pyright apps/api apps/web tests/unit/apps"
  - "uv run pytest -q tests/unit/apps/api tests/unit/apps/web"
  - "Browser QA for authenticated settings flow with `smoke_e2e_keycloak` when runtime is available"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `08`: add user-facing settings UI/API for Telegram binding status, scoped notification modes and weekly/monthly report schedule.

User required before start: `required only if smoke auth env is missing or user wants manual visual confirmation`.

## Requirements

- Verify Stage `04` and Stage `06` accepted.
- Keep existing `/ui/account/notifications` response compatible; add new routes/DTOs if needed.
- UI must show Telegram binding status without revealing chat id.
- Controls must cover critical-only, signals, trades, reports and report schedules.
- Browser QA must use `smoke_e2e_keycloak`; password only from host-local env.
- No credentials in screenshots, traces, reports or ledger.

## Acceptance Criteria

- API tests cover read/update of scoped settings and binding status.
- Browser QA proves settings render and update without console/network errors when runtime is available.
- Existing account settings tests remain compatible.

## Final Report

Respond in Russian with: API/UI changes, browser evidence or blocked-auth reason, checks, user/access notes, file manifest and ledger update.
