---
prompt_name: "Notifications v1 Stage 10 - Strategy Telegram Migration"
repo: "roehub.com"
branch: "main"
scope: "Migrate Strategy direct Telegram notifications to the notifications context with rollback"
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
  current_stage: "10"
  required_update: true
validation_strategy:
  depth: "integration"
  acceptance_surfaces: ["strategy runtime path", "notifications router", "rollback flag"]
proof_boundary:
  label: "post_main_production_runtime_proof"
  changed_code_production_claim_allowed: true
user_presence_required: "nothing if Stage 09 accepted"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "repo contract"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main/access contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "migration contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage gate"
  task_entrypoints:
    - path: "src/trading/contexts/strategy/application/services/live_runner.py"
      why: "current direct Telegram publish point"
    - path: "src/trading/contexts/strategy/application/ports/telegram_notifier.py"
      why: "existing Strategy notifier port"
    - path: "apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py"
      why: "runtime wiring and metrics"
    - path: "docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md"
      why: "old/current strategy Telegram documentation"
skill_routing:
  - skill: "contract-impact-analysis"
    timing: "before implementation"
    reason: "runtime behavior and config migration must be compatible"
  - skill: "backend-quality-gates"
    timing: "during verification"
    reason: "strategy runtime and notifications integration tests"
  - skill: "publish-ci-deploy"
    timing: "before production runtime proof"
    reason: "changed-code production proof requires main/CI/deploy"
expected_primary_touches:
  - "src/trading/contexts/strategy/application/services/live_runner.py"
  - "apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py"
  - "src/trading/contexts/notifications/"
  - "configs/dev/strategy.yaml"
  - "configs/test/strategy.yaml"
  - "configs/prod/strategy.yaml"
  - "tests/unit/contexts/strategy/"
  - "tests/unit/contexts/notifications/"
  - "docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/10-strategy-telegram-migration.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/README.md"
quality_gates:
  - "uv run ruff check src/trading/contexts/strategy src/trading/contexts/notifications apps/worker/strategy_live_runner tests/unit/contexts/strategy tests/unit/contexts/notifications"
  - "uv run pyright src/trading/contexts/strategy src/trading/contexts/notifications apps/worker/strategy_live_runner tests/unit/contexts/strategy tests/unit/contexts/notifications"
  - "uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/notifications tests/unit/apps"
  - "Real-boundary strategy runtime evidence: run a controlled strategy failure/signal use-case smoke through strategy_live_runner wiring or focused integration harness and prove notification event/delivery creation with rollback flag behavior"
  - "Post-main production proof if deployed: CI green, deploy/sync from main, Mac Studio smoke and notification runtime proof using `post_main_production_runtime_proof`"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `10`: migrate Strategy failure notifications from direct Strategy Telegram delivery to the `notifications` context while preserving a rollback flag and parity evidence.

User required before start: `nothing if Stage 09 accepted`.

## Requirements

- Verify Stage `09` accepted or record blocker.
- Keep direct Strategy Telegram adapter as rollback only until final closure.
- Route strategy run failures and, where already supported, signal/trade event facts into `notifications`.
- Preserve existing metrics until replacement metrics are proven; document metric compatibility.
- Do not broaden Strategy runtime side effects.
- Update old Strategy Telegram docs to reflect migration/fallback state.

## Acceptance Criteria

- Strategy failure path creates notification event/delivery through notifications.
- Real-boundary strategy runtime smoke proves a controlled failure/signal path creates notification event/delivery through notifications with no real Telegram send in local verification.
- Rollback flag restores old behavior in tests.
- No real Telegram send is required in local tests.
- Production proof is post-main only if deploy path is executed.

## Final Report

Respond in Russian with: migration behavior, rollback, checks, proof boundary, user/access notes, file manifest and ledger update.
