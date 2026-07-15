---
prompt_name: 11-notification-and-telegram-provider-plugins
repo: roehub.com
scope: "Implement NotificationProvider/v1 for notifications and Telegram with installation-wide and per-organization provider instances in the greenfield product."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "11", prerequisites: ["05", "08", "10"], previous_stage_gate: "Stages 05, 08 and 10 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: provider/secrets/evidence rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: provider package/instance decision}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: org and OpenBao evidence}
    - {path: docs/architecture/notifications/web-execution-telegram-notifications-v1.md, why: accepted notification semantics}
  task_entrypoints:
    - {path: src/trading/contexts/notifications/, why: current provider port and delivery model}
    - {path: apps/worker/notification_dispatcher/, why: provider composition}
    - {path: apps/worker/telegram_bot_worker/, why: bot runtime}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before provider contract changes, reason: delivery identity, retries and settings}
  - {skill: backend-quality-gates, timing: verification, reason: provider/worker gates}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/notifications/, apps/worker/notification_dispatcher/, apps/worker/notification_report_scheduler/, apps/worker/telegram_bot_worker/, apps/scheduler/, apps/api/, apps/web/, migrations/, configs/, tests/, docs/architecture/notifications/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve accepted notification history and unrelated routes
file_manifest:
  expected_primary_touches: [src/trading/contexts/notifications/, apps/worker/notification_dispatcher/, apps/worker/telegram_bot_worker/, migrations/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/11-notification-and-telegram-provider-plugins.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [apps/worker/notification_report_scheduler/, apps/scheduler/, apps/api/, apps/web/, configs/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: integration, acceptance_surfaces: [provider contract, two organizations and two bot instances, local provider end-to-end smoke, delivery idempotency, durable update cursor, no critical fallback]}
proof_boundary: {label: N/A, exclusions: [broad real Telegram send, production recipients]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Make the existing notification port the first stable plugin-style contract and support a shared bot, an organization-owned bot and a custom provider client without global raw-token environment configuration.

# Requirements

- Configure installation-wide and per-organization provider instances from fresh v1 settings and new OpenBao refs. Do not read or import current env values, Telegram tokens, recipients or delivery state.
- Define `NotificationProvider/v1` descriptor, config schema, secret refs, channel/template capabilities, health and bounded error codes.
- Separate installed provider package from per-installation/per-organization provider account.
- Add `TelegramBotProvider/v1`, durable update cursor, command registry and recipient resolver.
- Provide a dedicated `roehubctl providers add` / `roehubctl telegram connect` application command contract; implementation may be completed by Stage `18` but API/use case must exist.
- Preserve `sent/retry/unknown/dead_letter/suppressed` semantics and explicit replay.
- `NotificationProvider/v1` owns the delivery call. Default budgets are 3 seconds to connect and 10 seconds overall; cancellation and worker shutdown must leave a durable retry/unknown result rather than an in-memory limbo.
- Retry only bounded, classified transport/provider failures with capped exponential backoff, jitter and `Retry-After`. Use `delivery_id` as the idempotency identity. A timeout after possible provider acceptance maps to `unknown`, not ordinary retry.
- Provider outage degrades only the affected instance and exposes redacted sent/retry/unknown/dead-letter metrics, an alert for critical/trading `unknown`, and a linked operator runbook.
- Never auto-fallback critical or trading messages from an organization bot to the shared bot.

# Validation

Run focused gates and a local end-to-end provider smoke using two organizations and two isolated bot/provider instances against a controlled Telegram-compatible HTTP stub. Prove idempotent delivery, duplicate update handling, per-org secret resolution, cancellation, bounded backoff, `Retry-After`, timeout-before-acceptance, timeout-after-possible-acceptance → `unknown`, health/degraded states and forbidden cross-org routing. Real Telegram is optional and requires a separate bounded canary approval.

# Stop rules

Block on global token leakage, in-memory-only bindings/cursors, automatic critical fallback, blind retry from `unknown`, cross-org delivery or reports containing chat IDs/provider payloads. Update ledger after evidence.
