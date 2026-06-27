---
prompt_name: "Notifications v1 Stage 01 - Schema, Domain, Ports"
repo: "roehub.com"
branch: "main"
scope: "Create the notifications bounded-context foundation without provider side effects"
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
  current_stage: "01"
  required_update: true
validation_strategy:
  depth: "integration"
  acceptance_surfaces: ["domain", "migration/schema", "repository"]
proof_boundary:
  label: "none"
user_presence_required: "nothing"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "repo contract and main-branch prompt-pack policy"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main-only execution, access and user-presence contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "target architecture"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage status and handoff state"
  task_entrypoints:
    - path: "src/trading/contexts/live_execution/domain/notification.py"
      why: "existing notification fact model to avoid conflicting semantics"
    - path: "alembic/versions/20260603_0030_execution_notifications_producers_v1.py"
      why: "existing execution outbox schema"
    - path: "migrations/postgres/0006_identity_account_settings_v1.sql"
      why: "existing account notification preference schema"
    - path: "tests/unit/apps/migrations"
      why: "migration test pattern"
skill_routing:
  - skill: "contract-impact-analysis"
    timing: "before implementation"
    reason: "schema, DTO and config compatibility must stay additive"
  - skill: "backend-quality-gates"
    timing: "during verification"
    reason: "focused ruff, pyright and pytest gates"
expected_primary_touches:
  - "src/trading/contexts/notifications/"
  - "alembic/versions/"
  - "tests/unit/contexts/notifications/"
  - "tests/unit/apps/migrations/"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/01-notifications-schema-domain-ports.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "src/trading/contexts/__init__.py"
  - "docs/architecture/README.md"
quality_gates:
  - "uv run ruff check src/trading/contexts/notifications tests/unit/contexts/notifications tests/unit/apps/migrations"
  - "uv run pyright src/trading/contexts/notifications tests/unit/contexts/notifications"
  - "uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/migrations"
  - "Real-boundary schema evidence: apply/inspect the notifications migration through the repository migration test harness or a disposable local test database; record table/index/constraint presence in the stage report"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `01`: add the `notifications` bounded-context schema, domain objects, repository ports and test scaffolding. Do not send Telegram messages and do not wire runtime workers yet.

User required before start: `nothing`.

## Requirements

- Work only in `/Users/daniildegtyarev/Projects/roehub.com` on `main`.
- Do not create branches, worktrees, stashes, temporary checkouts, or local coordination folders.
- Add additive persisted schema only; do not change `execution_notification_outbox` semantics.
- Model `NotificationEvent`, `NotificationRoute`, `NotificationDelivery`, `NotificationDeliveryAttempt`, `TelegramUpdate`, and `NotificationReportRun` enough for later stages.
- Include status enums for `pending`, `claimed`, `sent`, `failed`, `retry`, `dead_letter`, `suppressed`, `unknown`.
- Include redaction-oriented constraints for provider payload hashes and avoid raw secret storage.
- Add migration tests and focused domain/repository tests.
- Update the stage ledger and create the Stage `01` report before final response.

## Acceptance Criteria

- Migration/schema tests prove additive tables/indexes/constraints.
- Real-boundary schema evidence proves the migration creates expected tables/indexes/constraints through the repository migration harness or disposable local test database.
- Domain tests cover dedupe keys, status validation, route separation and secret-like field rejection.
- Stage report records file manifest, contract impact and validation evidence.
- Docs index check passes if Markdown docs changed.

## Final Report

Respond in Russian with: scope, changed files, checks run, contract impact, user-presence/access notes, ledger update, residual risks.
