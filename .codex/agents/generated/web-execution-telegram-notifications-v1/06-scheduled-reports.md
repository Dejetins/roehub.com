---
prompt_name: "Notifications v1 Stage 06 - Scheduled Weekly And Monthly Reports"
repo: "roehub.com"
branch: "main"
scope: "Implement idempotent weekly/monthly report scheduler and report deliveries"
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
  current_stage: "06"
  required_update: true
validation_strategy:
  depth: "integration"
  acceptance_surfaces: ["scheduler", "report run table", "fake/log delivery"]
proof_boundary:
  label: "none"
user_presence_required: "nothing"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "repo contract"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main/access contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "scheduled report contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage gate"
  task_entrypoints:
    - path: "src/trading/contexts/notifications/application/"
      why: "Stage 01-05 notifications context"
    - path: "apps/worker/notification_dispatcher"
      why: "delivery path from Stage 03"
skill_routing:
  - skill: "backend-quality-gates"
    timing: "during verification"
    reason: "scheduler and report-run tests"
  - skill: "contract-impact-analysis"
    timing: "before implementation"
    reason: "schedule settings and report run persistence are durable contracts"
expected_primary_touches:
  - "src/trading/contexts/notifications/application/"
  - "apps/worker/notification_report_scheduler/"
  - "tests/unit/contexts/notifications/"
  - "tests/unit/apps/"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/06-scheduled-reports.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "configs/dev/"
  - "configs/test/"
  - "configs/prod/"
  - "docs/architecture/README.md"
quality_gates:
  - "uv run ruff check src/trading/contexts/notifications apps/worker/notification_report_scheduler tests/unit/contexts/notifications tests/unit/apps"
  - "uv run pyright src/trading/contexts/notifications apps/worker/notification_report_scheduler tests/unit/contexts/notifications tests/unit/apps"
  - "uv run pytest -q tests/unit/contexts/notifications tests/unit/apps"
  - "Real-boundary scheduler evidence: run a report scheduler smoke against seeded route/stats fixtures and fake/log delivery, proving report_run dedupe and delivery candidate creation"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `06`: idempotent weekly/monthly portfolio report runs per active route and delivery through fake/log provider.

User required before start: `nothing`.

## Requirements

- Verify Stage `05` accepted.
- Use stable dedupe key: user, report type, period start/end, scope.
- Respect per-user timezone when available; otherwise make platform default explicit.
- Create report runs with `complete`, `partial`, or `unavailable` stats quality.
- Add missed schedule metric/alert hook but do not require real Telegram provider.
- Update ledger coverage for weekly and monthly `portfolio_report`.

## Acceptance Criteria

- Scheduler tests prove no duplicate report runs for same period.
- Real-boundary scheduler smoke proves weekly/monthly report runs create fake/log delivery candidates through the application wiring or repository integration fixture.
- Weekly/monthly report rendering includes period id and quality status.
- Fake/log delivery path creates delivery rows and attempts.

## Final Report

Respond in Russian with: scheduler behavior, idempotency proof, checks, file manifest, user/access notes and ledger update.
