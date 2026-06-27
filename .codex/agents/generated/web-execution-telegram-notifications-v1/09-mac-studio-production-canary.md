---
prompt_name: "Notifications v1 Stage 09 - Mac Studio Production Canary"
repo: "roehub.com"
branch: "main"
scope: "Prove production topology, fake/log synthetic matrix and one bounded real Telegram canary"
language:
  implementation: "ops/python"
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
  current_stage: "09"
  required_update: true
validation_strategy:
  depth: "ci_deploy"
  acceptance_surfaces: ["CI", "deploy", "Mac Studio runtime", "Telegram canary"]
proof_boundary:
  label: "post_main_production_runtime_proof"
  changed_code_production_claim_allowed: true
user_presence_required: "required for real Telegram canary message confirmation and canary recipient approval"
runtime_env_sources:
  report_only_key_presence: true
  required_for_real_canary:
    - "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN"
    - "ROEHUB_NOTIFICATIONS_ADMIN_TELEGRAM_CHAT_ID or persisted admin route"
    - "ROEHUB_SMOKE_E2E_PASSWORD for browser/auth flows"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "Mac Studio, deploy, browser auth and secret policy"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main/access/user-presence contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "canary contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage gate"
  task_entrypoints:
    - path: "scripts/macos/"
      why: "Mac Studio service management pattern"
    - path: "infra/macos/"
      why: "launchd/Prometheus/Monit assets"
    - path: "docs/runbooks/"
      why: "runbook references"
skill_routing:
  - skill: "publish-ci-deploy"
    timing: "before production runtime proof"
    reason: "changed-code production proof requires main, CI, deploy/sync and smoke"
  - skill: "browser-qa-evidence"
    timing: "during verification when UI/auth is involved"
    reason: "settings/canary verification may be browser-visible"
  - skill: "backend-quality-gates"
    timing: "before deploy"
    reason: "focused local gates"
expected_primary_touches:
  - "infra/macos/"
  - "scripts/macos/"
  - "configs/prod/"
  - "docs/runbooks/"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/09-mac-studio-production-canary.md"
  - "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
possible_secondary_touches:
  - "apps/worker/notification_dispatcher/"
  - "apps/worker/telegram_bot_worker/"
  - "apps/worker/notification_report_scheduler/"
  - "docs/architecture/README.md"
quality_gates:
  - "Focused local gates for touched code/config before deploy"
  - "GitHub CI/deploy proof for main revision"
  - "Mac Studio smoke after deploy/sync"
  - "Fake/log full synthetic notification matrix in production runtime"
  - "One bounded real Telegram canary if user/token/admin recipient are available"
  - "uv run python -m tools.docs.generate_docs_index --check"
---

# Task

Implement Stage `09`: production-safe canary on Mac Studio. Prove worker topology and metrics with fake/log provider first, then one bounded real Telegram canary only after user/token/recipient readiness is confirmed.

User required before start: `required for real Telegram canary message confirmation and canary recipient approval`.

## Requirements

- Verify Stage `08` accepted.
- Use `publish-ci-deploy` discipline for changed-code production proof.
- Use `post_main_production_runtime_proof`; do not present pre-main host checks as changed-code proof.
- Check env key presence only; never print raw values.
- Real Telegram canary must be one test/smoke user route and one admin route at most.
- Revoke/disable temporary canary routes if created only for proof.
- Record fake/log matrix and real canary separately.

## Acceptance Criteria

- Main revision, CI/deploy/sync and Mac Studio smoke evidence recorded.
- Notification workers are healthy and observable.
- Full synthetic matrix passes with fake/log provider in runtime.
- Real Telegram canary is either accepted with user-confirmed receipt or explicitly blocked on missing user/token/recipient.

## Final Report

Respond in Russian with: proof boundary, CI/deploy/runtime evidence, canary result, user-presence/access notes, cleanup, file manifest and ledger update.
