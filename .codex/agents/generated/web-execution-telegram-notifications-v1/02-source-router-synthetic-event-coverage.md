---
prompt_name: "Notifications v1 Stage 02 - Source Router Synthetic Event Coverage"
repo: "roehub.com"
branch: "main"
scope: "Map source facts into generic notification events and prove every planned notification type synthetically"
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
  current_stage: "02"
  required_update: true
validation_strategy:
  depth: "integration"
  acceptance_surfaces: ["domain", "database synthetic rows", "route decision"]
proof_boundary:
  label: "none"
user_presence_required: "nothing"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "repo contract"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main-only and access contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "category and routing contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "synthetic coverage matrix"
  task_entrypoints:
    - path: "src/trading/contexts/strategy/application/ports/telegram_notifier.py"
      why: "existing strategy event names"
    - path: "src/trading/contexts/live_execution/domain/notification.py"
      why: "existing execution notification event types"
    - path: "apps/api/routes/ui_execution.py"
      why: "existing execution notification API shape"
    - path: "alembic/versions/20260531_0018_strategy_signals_v1.py"
      why: "strategy signal source table"
skill_routing:
  - skill: "contract-impact-analysis"
    timing: "before implementation"
    reason: "source mappings must not change producer contracts"
  - skill: "backend-quality-gates"
    timing: "during verification"
    reason: "focused unit/integration gates"
expected_primary_touches:
  - "src/trading/contexts/notifications/application/"
  - "src/trading/contexts/notifications/adapters/"
  - "tests/unit/contexts/notifications/"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/02-source-router-synthetic-event-coverage.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "tests/integration/"
  - "docs/architecture/README.md"
quality_gates:
  - "uv run ruff check src/trading/contexts/notifications tests/unit/contexts/notifications"
  - "uv run pyright src/trading/contexts/notifications tests/unit/contexts/notifications"
  - "uv run pytest -q tests/unit/contexts/notifications"
  - "Real-boundary synthetic flow evidence: run an application-level smoke that writes/reads notification event, route decision, delivery candidate and attempt rows through the repository adapter or disposable local test database"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `02`: build source adapters/routers that convert synthetic Strategy, Live Execution, report and admin facts into generic `NotificationEvent` and route decisions. Use fake/log delivery only.

User required before start: `nothing`.

## Requirements

- Verify Stage `01` is accepted or explicitly continue only if this prompt is repairing Stage `01`.
- Cover every row in the synthetic notification matrix: strategy failure, strategy signal, trade fill, execution rejected, terminal, unknown, kill switch, weekly/monthly report, day/week/month stats response, strategy stats, exchange stats, admin critical, admin alert and admin report.
- Prove user/admin route separation and preference-mode decisions.
- Do not call Telegram or require Telegram token.
- Preserve current Strategy and Live Execution contracts; source contexts are read/adapted, not rewritten.
- Update ledger matrix with exact synthetic evidence.

## Acceptance Criteria

- Tests prove every notification type reaches event + route decision + fake/log delivery candidate.
- Real-boundary synthetic flow evidence proves at least one user event and one admin event round-trip through repository adapters or a disposable local test database.
- Redaction tests prove no raw provider token/chat id/secret-like label enters event payloads.
- Stage report records type-by-type evidence and blockers if any type remains uncovered.

## Final Report

Respond in Russian with: synthetic coverage table, changed files, checks, user-presence/access status, ledger updates and residual risks.
