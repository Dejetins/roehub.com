---
prompt_name: "Notifications v1 Stage 05 - Stats Query Service Day Week Month"
repo: "roehub.com"
branch: "main"
scope: "Implement portfolio, strategy and exchange stats snapshots for bot commands and reports"
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
  current_stage: "05"
  required_update: true
validation_strategy:
  depth: "integration"
  acceptance_surfaces: ["query service", "database fixtures", "command integration"]
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
      why: "stats quality contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage gate"
  task_entrypoints:
    - path: "alembic/versions/20260531_0018_strategy_signals_v1.py"
      why: "strategy signal ledger"
    - path: "alembic/versions/20260531_0022_capital_reservation_paper_accounting_v1.py"
      why: "paper accounting/fills"
    - path: "alembic/versions/20260531_0027_testnet_order_adapters_v1.py"
      why: "execution orders"
    - path: "alembic/versions/20260602_0029_execution_reconciliation_pitr_v1.py"
      why: "fills/funding/reconciliation"
    - path: "alembic/versions/20260531_0020_exchange_account_projection_config_guard_v1.py"
      why: "exchange snapshots"
skill_routing:
  - skill: "contract-impact-analysis"
    timing: "before implementation"
    reason: "stats DTO/query shape is user-visible through bot/API"
  - skill: "backend-quality-gates"
    timing: "during verification"
    reason: "query fixture tests"
expected_primary_touches:
  - "src/trading/contexts/notifications/application/"
  - "src/trading/contexts/notifications/adapters/outbound/acl/"
  - "tests/unit/contexts/notifications/"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/05-stats-query-service-day-week-month.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "apps/api/dto/"
  - "apps/api/routes/"
  - "docs/architecture/README.md"
quality_gates:
  - "uv run ruff check src/trading/contexts/notifications tests/unit/contexts/notifications"
  - "uv run pyright src/trading/contexts/notifications tests/unit/contexts/notifications"
  - "uv run pytest -q tests/unit/contexts/notifications"
  - "Real-boundary stats evidence: execute the stats query service against seeded repository/database fixtures for day, week and month periods and record quality_status, missing_sources and owner filter results"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `05`: `NotificationStatsQueryService` for day/week/month portfolio stats and strategy/exchange scoped stats. This powers bot command responses and scheduled reports.

User required before start: `nothing`.

## Requirements

- Verify Stage `04` accepted, or implement stats isolated from bot integration if the ledger explicitly allows it.
- Read existing ledgers through ACL/query ports; do not make notifications own trading truth.
- Return `quality_status`: `complete`, `partial`, `unavailable`.
- Include `missing_sources`, freshness and timezone/period metadata.
- Do not infer testnet/mainnet PnL from incomplete fills/order rows.
- Test owner filters for strategy and exchange scopes.

## Acceptance Criteria

- Fixture tests cover day/week/month periods.
- Real-boundary query evidence proves the stats service reads seeded source ledgers through repository/database fixtures, not only pure unit objects.
- Strategy/exchange filters cannot leak another owner’s rows.
- Partial/unavailable behavior is explicit and tested.
- Command integration tests can render stats responses without fake metrics.

## Final Report

Respond in Russian with: stats fields, quality behavior, checks, file manifest, user/access notes and ledger update.
