---
prompt_name: "Notifications v1 Stage 07 - Admin Notifications And Runbooks"
repo: "roehub.com"
branch: "main"
scope: "Implement admin critical/alert/report routing, alert metrics and operational runbooks"
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
  current_stage: "07"
  required_update: true
validation_strategy:
  depth: "integration"
  acceptance_surfaces: ["admin route", "synthetic admin events", "metrics/alerts", "runbook"]
proof_boundary:
  label: "none"
user_presence_required: "required only to choose/confirm real admin Telegram recipient; synthetic admin drill requires nothing"
runtime_env_sources:
  report_only_key_presence: true
  optional_keys:
    - "ROEHUB_NOTIFICATIONS_ADMIN_TELEGRAM_CHAT_ID"
    - "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "repo contract"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main/access/user-presence contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "admin notification contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage gate"
  task_entrypoints:
    - path: "apps/api/monitoring.py"
      why: "existing metric registration pattern"
    - path: "infra/macos/prometheus/"
      why: "existing Prometheus/alert assets"
    - path: "docs/runbooks/"
      why: "runbook shape"
skill_routing:
  - skill: "contract-impact-analysis"
    timing: "before implementation"
    reason: "admin route and alert semantics are operational contracts"
  - skill: "backend-quality-gates"
    timing: "during verification"
    reason: "admin event and alert tests"
expected_primary_touches:
  - "src/trading/contexts/notifications/application/"
  - "src/trading/contexts/notifications/adapters/"
  - "apps/api/monitoring.py"
  - "infra/macos/prometheus/"
  - "docs/runbooks/notifications-admin-alerts.md"
  - "tests/unit/contexts/notifications/"
  - "tests/unit/infra/"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/07-admin-notifications-runbooks.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "configs/prod/"
  - "docs/architecture/README.md"
quality_gates:
  - "uv run ruff check src/trading/contexts/notifications tests/unit/contexts/notifications tests/unit/infra"
  - "uv run pyright src/trading/contexts/notifications tests/unit/contexts/notifications tests/unit/infra"
  - "uv run pytest -q tests/unit/contexts/notifications tests/unit/infra"
  - "Real-boundary admin drill evidence: execute synthetic admin critical/alert/report through repository adapters and fake/log provider, proving admin-only route/delivery rows and metrics/alert assets"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `07`: admin notification categories, route separation, synthetic critical/alert/report drills, alert metrics and runbook coverage.

User required before start: `required only to choose/confirm real admin Telegram recipient; synthetic admin drill requires nothing`.

## Requirements

- Verify Stage `03` accepted; Stage `06` is preferred before admin reports.
- Admin route must be separate from user routes.
- Cover `admin_critical`, `admin_alert`, and `admin_report` with synthetic events.
- Alert on critical unknown delivery, dispatcher stuck/pending age, worker down, high retry/429 rate and missed report schedule.
- Runbook must describe diagnosis, redaction, replay policy and escalation owner.
- Do not print admin chat id; report only key presence/redacted hash where needed.

## Acceptance Criteria

- Synthetic admin drills create admin-only event/route/delivery/attempt rows.
- Real-boundary admin drill proves admin-only event, route, delivery and attempt rows through repository adapters or disposable local database.
- Metrics/alert tests pass.
- Runbook includes no secrets and links back to stage evidence.

## Final Report

Respond in Russian with: admin drill result, alert/runbook changes, checks, user/access notes, file manifest and ledger update.
