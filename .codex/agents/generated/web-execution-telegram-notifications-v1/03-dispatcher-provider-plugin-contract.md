---
prompt_name: "Notifications v1 Stage 03 - Dispatcher And Provider Plugin Contract"
repo: "roehub.com"
branch: "main"
scope: "Implement delivery dispatcher, provider adapter interface, fake/log provider and Telegram adapter behind safe config"
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
  current_stage: "03"
  required_update: true
validation_strategy:
  depth: "integration"
  acceptance_surfaces: ["dispatcher queue", "provider fake/log adapter", "metrics"]
proof_boundary:
  label: "none"
user_presence_required: "nothing for fake/log provider"
runtime_env_sources:
  report_only_key_presence: true
  optional_keys:
    - "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN"
    - "TELEGRAM_BOT_TOKEN"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "repo contract"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main-only and secret policy"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "provider and unknown-state contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage status"
  task_entrypoints:
    - path: "src/trading/contexts/strategy/adapters/outbound/messaging/telegram/telegram_bot_api_notifier.py"
      why: "existing Telegram HTTP adapter behavior"
    - path: "apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py"
      why: "existing metrics/config wiring pattern"
    - path: "configs/dev/strategy.yaml"
      why: "existing log_only strategy Telegram config pattern"
skill_routing:
  - skill: "contract-impact-analysis"
    timing: "before implementation"
    reason: "provider/config/runtime side effects must be additive"
  - skill: "backend-quality-gates"
    timing: "during verification"
    reason: "dispatcher, adapter and metrics tests"
expected_primary_touches:
  - "src/trading/contexts/notifications/application/"
  - "src/trading/contexts/notifications/adapters/outbound/providers/"
  - "apps/worker/notification_dispatcher/"
  - "configs/dev/"
  - "configs/test/"
  - "configs/prod/"
  - "tests/unit/contexts/notifications/"
  - "tests/unit/apps/"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/03-dispatcher-provider-plugin-contract.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "apps/api/monitoring.py"
  - "infra/macos/prometheus/"
  - "docs/architecture/README.md"
quality_gates:
  - "uv run ruff check src/trading/contexts/notifications apps/worker/notification_dispatcher tests/unit/contexts/notifications tests/unit/apps"
  - "uv run pyright src/trading/contexts/notifications apps/worker/notification_dispatcher tests/unit/contexts/notifications tests/unit/apps"
  - "uv run pytest -q tests/unit/contexts/notifications tests/unit/apps"
  - "Real-boundary dispatcher evidence: run a fake/log provider backlog-drain smoke through the dispatcher composition root against a disposable local database or repository integration fixture; record pending->claimed->sent/unknown/dead_letter transitions"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `03`: delivery dispatcher, provider adapter port, fake/log provider and Telegram adapter behind disabled/log-only defaults. This stage must not send real Telegram messages by default.

User required before start: `nothing for fake/log provider`.

## Requirements

- Verify Stage `02` accepted.
- Implement claim/lease/attempt/backoff/dead-letter/unknown-state handling.
- Treat Telegram timeout/ambiguous 5xx as `unknown`; do not blind retry trade/critical messages.
- Keep `telegram_bot_api` behind config; default dev/test to fake/log or disabled.
- Add metrics for events, route decisions, deliveries, latency, pending age and unknown count.
- Report only presence of Telegram env keys, never raw values.
- Update stage report and ledger.

## Acceptance Criteria

- Dispatcher tests prove lease, retry, expired lease reclaim, dead-letter and unknown behavior.
- Real-boundary dispatcher smoke proves the dispatcher drains fake/log deliveries through the composition root or repository integration fixture.
- Provider tests prove fake/log deterministic delivery and Telegram adapter request redaction without real send.
- Config tests prove safe defaults.
- Metrics tests prove expected counters/gauges are registered or exposed through existing pattern.

## Final Report

Respond in Russian with: dispatcher/provider behavior, unknown-state proof, config defaults, checks, user/access notes, file manifest and ledger update.
