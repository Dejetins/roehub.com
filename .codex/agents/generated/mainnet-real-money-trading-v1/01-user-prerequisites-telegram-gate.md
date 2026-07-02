---
prompt_name: mainnet-real-money-trading-v1-01-user-prerequisites-telegram-gate
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Gate all mainnet execution on user prerequisites and Telegram host readiness."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo safety and proof-boundary contract"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "mainnet plan requirements"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage source of truth"
  task_entrypoints:
    - path: docs/runbooks/notifications-admin-alerts.md
      why: "notification readiness and replay policy"
      inspect_symbols: ["Stage 09 Production Canary", "real Telegram readiness"]
    - path: docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md
      why: "notification platform state"
      inspect_symbols: ["ledger_status", "Telegram", "canary"]
    - path: infra/macos/prometheus/rules/notifications-admin.rules.yml
      why: "notification alerts"
      inspect_symbols: ["NotificationsWorkerDown", "NotificationsCriticalUnknownDelivery"]
  conditional_bundles:
    notification_runtime:
      read_when: "runtime command or worker readiness needs exact entrypoint"
      paths:
        - configs/prod/notifications.yaml
        - scripts/notifications/stage09_production_canary.py
  consult_if_needed:
    - path: docs/runbooks/strategy-live-worker.md
      read_when: "strategy user-alert linkage is unclear"
style_references:
  - .codex/agents/stage_execution_ledger_template.md
hard_requirements:
  telegram_user_confirmation_required: true
  no_mainnet_order_submit: true
  stage_ledger_update_required: true
task_toggles:
  allow_runtime_read_only_checks: true
  allow_real_telegram_readiness: true
skill_routing:
  - skill: root-cause-debugging
    use_when: "Telegram readiness fails unexpectedly after user says it is solved"
    timing: if blocker
    reason: "localize host/provider connectivity without changing scope"
  - skill: publish-ci-deploy
    use_when: "docs or readiness code changes are accepted"
    timing: before ship
    reason: "direct-main delivery and runtime proof"
target_envs: ["macstudio"]
required_literals:
  - "User required before start: user must state Telegram blocker is solved"
  - "Telegram setup/VLESS is out of scope"
non_goals:
  - "Do not set up VLESS/VPN."
  - "Do not send mainnet orders."
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  single_allowed_branch: null
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
  approval_required_for_branch_or_worktree: true
change_ownership:
  parallel_main_expected: true
  owned_change_scope:
    - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/01-user-prerequisites-telegram-gate.md
    - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
    - docs/runbooks/notifications-admin-alerts.md
    - docs/architecture/README.md
  foreign_changes_policy: "ignore and preserve unrelated changes"
  mixed_file_policy: "stage only owned hunks; block mixed file if unsafe"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format:
  language: ru
  sections: ["status", "user_required", "telegram_evidence", "blockers", "next_stage"]
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["user confirmation", "Mac Studio Telegram readiness", "notification runtime"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/01-user-prerequisites-telegram-gate.md
proof_boundary:
  required_when: "Telegram runtime readiness is checked"
  label: read_only_existing_runtime_smoke
  changed_code_production_claim_allowed: false
  blocked_or_deferred_reason: "If code changes are required, accept only after post-main proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["Telegram token", "chat id", "cookies", "credentials"]
remote_command_quoting:
  applies_when: "SSH command includes env, JSON or multiline payload"
  required_pattern: "quoted heredoc or stdin"
  forbidden_pattern: "nested inline JSON or shell payloads"
  temporary_files_allowed_only_when_task_requires_durable_artifact: true
stage_execution_ledger:
  path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
  current_stage: "01"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
prompt_pack_execution:
  mode: goal_driven
  plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
  prompt_pack_dir: .codex/agents/generated/mainnet-real-money-trading-v1/
  stage_ledger: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
  goal_mode_optional: true
  goal_artifact_required: false
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    docs_runbooks: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/01-user-prerequisites-telegram-gate.md", "docs/runbooks/notifications-admin-alerts.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches:
  - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/01-user-prerequisites-telegram-gate.md
  - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
possible_secondary_touches:
  - docs/runbooks/notifications-admin-alerts.md
  - docs/architecture/README.md
safety_notes:
  - "This stage is a hard gate. If user confirmation is absent, mark blocked immediately."
---

# Task

Gate all later mainnet work on explicit user confirmation and Telegram host readiness.

Done means:

- Stage `00` is accepted;
- user has literally confirmed that the Telegram host blocker is solved;
- Mac Studio can prove Telegram readiness without exposing tokens/chat ids;
- ledger records Stage `01 accepted` or `blocked`;
- no mainnet trading stages proceed if this stage is blocked.

## Context / Current State

User stated that Telegram delivery is required for real-money scope, but host connectivity to `api.telegram.org` currently needs an out-of-scope VLESS/VPN fix. This is a blocker until the user says it is solved.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `00` is `accepted`. If not, update Stage `01` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: do not infer Telegram readiness from chat history; user confirmation and runtime evidence must be recorded in `stage_ledger` and the Stage `01` report.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before any implementation or runtime action, read the stage ledger and confirm Stage `00 accepted`; if not, write Stage `01 blocked`, update the ledger, and stop.
- Record `User required before start: user must state Telegram blocker is solved`.
- If that exact user-level confirmation is absent in the current thread/context, stop and mark Stage `01 blocked`.
- Do not configure VLESS/VPN.
- Do not print Telegram tokens, chat ids, cookies or provider payloads.
- Prove runtime readiness via existing notification readiness/canary tools or record a precise blocker.

## Requirements (Should)

- Prefer readiness checks over sending a real message unless the existing runbook requires/permits a canary.
- Update notification runbook only if it lacks mainnet trading alert gating.

## Requirements (Nice-to-have)

- Add a compact mainnet trading alert checklist to the runbook if missing.

# Context acquisition protocol

Read the three always-read files first. Do not load notification implementation files unless the runbook does not name the readiness command.

Reading budget: target `<= 8 files` and `<= 45k tokens`. Expand only for failing Telegram readiness, missing command entrypoint, or docs mismatch.

Stop when user prerequisite status, runtime readiness method, touched files, and next-stage gate are unambiguous.

# Reading manifest

Use `notification_runtime` only if you need exact script/config paths. Do not inspect secrets.

# Work plan (agent should follow)

1. Verify ledger Stage `00 accepted`.
2. Check for explicit user confirmation that Telegram blocker is solved.
3. If absent, write a blocked Stage `01` report and ledger update; stop.
4. If present, run safe Telegram readiness evidence on Mac Studio using runbook-approved commands.
5. If readiness fails, record blocked with root-cause boundary and do not continue.
6. If readiness passes, update report/ledger and open Stage `02`.
7. Run docs gates and deliver scoped changes if needed.

# Acceptance criteria (Definition of Done)

- Stage `01 accepted` only with user confirmation and runtime Telegram readiness proof.
- `01 blocked` if user confirmation or runtime proof is missing.
- No mainnet order, no adapter enablement, no secret leakage.

# Implementation constraints

- VLESS/VPN setup is out of scope.
- Treat Telegram readiness as money-safety prerequisite, not optional observability.

# Files to indicate (expected touched areas)

- Stage report.
- Stage ledger.
- Notification runbook only if missing mainnet alert gate wording.
- Docs index if Markdown changes.
- Final file manifest must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No exchange keys.
- No order submit.
- No provider token changes.

# Quality gates (must run and pass)

- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`

# Final output: report format (strict)

Report in Russian:

- Stage status: accepted or blocked.
- Exact user prerequisite state.
- Telegram readiness command/evidence, sanitized.
- Files changed and foreign changes excluded.
- Whether Stage `02` is allowed.
