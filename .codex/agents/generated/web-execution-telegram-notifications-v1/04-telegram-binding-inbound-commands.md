---
prompt_name: "Notifications v1 Stage 04 - Telegram Binding And Inbound Commands"
repo: "roehub.com"
branch: "main"
scope: "Implement Telegram binding code flow, durable updates and stats/settings command handling"
language:
  implementation: "python"
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
  current_stage: "04"
  required_update: true
validation_strategy:
  depth: "integration"
  acceptance_surfaces: ["API", "Telegram update persistence", "command use cases"]
proof_boundary:
  label: "none"
user_presence_required: "required only for real Telegram /start binding smoke; synthetic command tests require nothing"
runtime_env_sources:
  report_only_key_presence: true
  optional_keys:
    - "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN"
    - "TELEGRAM_BOT_TOKEN"
    - "ROEHUB_SMOKE_E2E_PASSWORD"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "browser auth and secret policy"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main/access/user-presence contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "bot command contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage gate"
  task_entrypoints:
    - path: "migrations/postgres/0001_identity_v1.sql"
      why: "confirmed Telegram chat storage"
    - path: "apps/api/routes/ui_account.py"
      why: "account settings API pattern"
    - path: "apps/api/dto/ui_account.py"
      why: "account DTO pattern"
    - path: "src/trading/contexts/strategy/adapters/outbound/acl/identity/confirmed_telegram_chat_binding_resolver.py"
      why: "existing confirmed chat resolver"
skill_routing:
  - skill: "contract-impact-analysis"
    timing: "before implementation"
    reason: "binding and command APIs must be additive"
  - skill: "backend-quality-gates"
    timing: "during verification"
    reason: "API/use-case/inbound worker tests"
expected_primary_touches:
  - "src/trading/contexts/notifications/application/"
  - "src/trading/contexts/notifications/adapters/inbound/telegram/"
  - "src/trading/contexts/notifications/adapters/outbound/acl/identity/"
  - "apps/api/routes/"
  - "apps/api/dto/"
  - "apps/worker/telegram_bot_worker/"
  - "tests/unit/contexts/notifications/"
  - "tests/unit/apps/api/"
  - "tests/unit/apps/"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/04-telegram-binding-inbound-commands.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "apps/api/main/app.py"
  - "configs/dev/"
  - "configs/test/"
  - "configs/prod/"
  - "docs/architecture/README.md"
quality_gates:
  - "uv run ruff check src/trading/contexts/notifications apps/api apps/worker/telegram_bot_worker tests/unit/contexts/notifications tests/unit/apps"
  - "uv run pyright src/trading/contexts/notifications apps/api apps/worker/telegram_bot_worker tests/unit/contexts/notifications tests/unit/apps"
  - "uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/api tests/unit/apps"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `04`: web-generated one-time Telegram binding code, durable Telegram update intake, polling worker shell and command handling for `/start`, `/stats`, `/strategy`, `/exchange`, `/settings`, mode toggles and report toggles.

User required before start: `required only for real Telegram /start binding smoke; synthetic command tests require nothing`.

## Requirements

- Verify Stage `03` accepted.
- Store binding codes hashed with TTL and owner; never accept user id text as proof.
- Store Telegram update ids durably and handle commands idempotently.
- Add synthetic tests for `/stats today|week|month`, `/strategy <id>`, `/exchange <connection>`, `/settings`, `/critical_only`, `/signals_on/off`, `/reports weekly/monthly on/off`.
- Real Telegram binding smoke is optional in this stage; if used, read token from host-local env and require user to send `/start <code>`.
- Do not print token, full chat id, raw update payload, cookies or passwords.

## Acceptance Criteria

- Binding API/use case and command handler tests pass.
- Duplicate `telegram_update_id` is idempotent.
- Unauthorized strategy/exchange command scopes fail closed.
- Stage report records whether real Telegram binding smoke was skipped or completed.

## Final Report

Respond in Russian with: command coverage, binding security, checks, user-presence/access result, file manifest and ledger update.
