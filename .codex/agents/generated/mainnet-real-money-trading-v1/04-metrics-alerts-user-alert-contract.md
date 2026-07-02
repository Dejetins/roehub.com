---
prompt_name: mainnet-real-money-trading-v1-04-metrics-alerts-user-alert-contract
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Add mainnet Prometheus metrics, alerts, dashboard reference, and user-alert readiness contract."
language: {implementation: python, agent_report: ru}
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "metrics and alerts plan"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage source of truth"
  task_entrypoints:
    - path: infra/macos/prometheus/rules/live-execution-stage17.rules.yml
      why: "existing live-execution alert pattern"
      inspect_symbols: ["LiveExecution", "latency", "unknown"]
    - path: infra/macos/prometheus/rules/strategy-producer.rules.yml
      why: "producer alert pattern"
      inspect_symbols: ["StrategyProducer", "notification"]
    - path: docs/runbooks/prod-dashboard-metrics-reference-ru.md
      why: "metrics journal to update"
      inspect_symbols: ["Dashboard structure", "PromQL"]
    - path: docs/runbooks/notifications-admin-alerts.md
      why: "user-alert delivery and replay policy"
      inspect_symbols: ["Telegram", "Unknown Delivery", "Canary"]
  conditional_bundles:
    metrics_code:
      read_when: "new metrics are missing in code"
      paths: ["apps/api/monitoring.py", "apps/exchange_execution", "apps/worker/strategy_live_runner"]
  consult_if_needed:
    - path: docs/runbooks/exchange-execution.md
      read_when: "exchange-execution runbook alert actions need sync"
style_references: [".codex/agents/stage_execution_ledger_template.md"]
hard_requirements: {stage_03_must_be_accepted: true, telegram_gate_must_be_accepted: true, no_order_submit: true}
task_toggles: {allow_prometheus_rules: true, allow_notification_runtime_readiness: true}
skill_routing:
  - skill: backend-performance-evidence
    use_when: "defining latency baseline or thresholds"
    timing: during verification
    reason: "requires comparable latency evidence and no vague speed claims"
  - skill: publish-ci-deploy
    use_when: "accepted metrics/runbook/code changes need delivery"
    timing: before ship
    reason: "main delivery and post-main proof"
target_envs: ["local", "macstudio", "prometheus"]
required_literals: ["mainnet_execution_latency_seconds", "mainnet_user_alert_delivery_total", "No order submit"]
non_goals: ["Do not set up VLESS/VPN.", "Do not place orders."]
branch_policy: {default_branch: main, separate_branch_allowed: false, single_allowed_branch: null, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false, approval_required_for_branch_or_worktree: true}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: ["Stage 04 metrics/alerts/runbook/report/ledger only"]
  foreign_changes_policy: "preserve unrelated changes"
  mixed_file_policy: "stage only owned hunks"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format: {language: ru, sections: ["status", "metrics", "alerts", "user_alerts", "files", "next_stage"]}
quality_gates:
  - cmd: "promtool check rules infra/macos/prometheus/rules/*.yml"
    expect: "passes if promtool is available; otherwise record unavailable"
  - cmd: "uv run ruff check apps src tests"
    expect: "passes if code touched"
  - cmd: "uv run pyright apps src tests"
    expect: "passes if code touched"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["Prometheus rules", "metrics endpoint", "notification readiness", "runbooks"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/04-metrics-alerts-user-alert-contract.md
proof_boundary:
  required_when: "runtime metrics or notification readiness are verified"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Requires origin/main, green CI, deploy/sync, then Prometheus/runtime proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["raw secrets", "Telegram tokens", "chat ids", "cookies"]
remote_command_quoting: {applies_when: "SSH uses PromQL/JSON/SQL payloads", required_pattern: "quoted heredoc/stdin", forbidden_pattern: "nested inline payload quoting", temporary_files_allowed_only_when_task_requires_durable_artifact: true}
stage_execution_ledger: {path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, current_stage: "04", required_update: true, template: .codex/agents/stage_execution_ledger_template.md}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, prompt_pack_dir: .codex/agents/generated/mainnet-real-money-trading-v1/, stage_ledger: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    code: ["apps/**", "src/**"]
    config_infra_migrations: ["infra/macos/prometheus/rules/**", "configs/prod/**"]
    docs_runbooks: ["docs/runbooks/prod-dashboard-metrics-reference-ru.md", "docs/runbooks/notifications-admin-alerts.md", "docs/runbooks/exchange-execution.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/04-metrics-alerts-user-alert-contract.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches: ["infra/macos/prometheus/rules", "docs/runbooks/prod-dashboard-metrics-reference-ru.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/04-metrics-alerts-user-alert-contract.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
possible_secondary_touches: ["apps/api/monitoring.py", "apps/exchange_execution", "apps/worker/strategy_live_runner", "docs/runbooks/notifications-admin-alerts.md", "docs/runbooks/exchange-execution.md", "docs/architecture/README.md"]
safety_notes: ["Metrics must not include high-cardinality order/user identifiers."]
---

# Task

Add or harden mainnet Prometheus metrics, alert rules, dashboard/runbook references, and user-alert readiness contract before mainnet submit is possible.

Done means mainnet latency/slippage/reconciliation/unknown/exposure/user-alert metrics are defined, alerts have severity/owner/actions, and Telegram/user-alert readiness remains a hard gate.

## Context / Current State

Existing Stage 17 alerts are testnet-oriented and often say to keep mainnet disabled. Real-money trading needs separate mainnet metrics and runbook actions.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `03` is `accepted` and Stage `01` remains `accepted`. If not, update Stage `04` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: mainnet alert readiness requires Prometheus/runbook evidence plus user-alert delivery readiness; outbox-only proof is not enough for closure.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before any implementation or runtime action, read the stage ledger and confirm Stage `03 accepted` and Stage `01 accepted`; if not, write Stage `04 blocked`, update the ledger, and stop.
- Record `User required before start: Telegram gate accepted; no new user action unless notification route approval is missing`.
- Add/update metrics and alert docs for `mainnet_*` series from the plan.
- Update `docs/runbooks/prod-dashboard-metrics-reference-ru.md`.
- Prove metrics/rules through Prometheus/runtime evidence after delivery if code/config changed.
- Do not place orders.

## Requirements (Should)

- Reuse existing metric naming and bounded label style.
- Keep alerts actionable, with severity and owner.

## Requirements (Nice-to-have)

- Include copy-paste PromQL for p95/p99 latency panels.

# Context acquisition protocol

Read plan/ledger and existing Prometheus/runbook files first. Do not inspect code unless metrics do not exist or rules cannot be proven.

Reading budget: target `<= 10 files`, expand only for missing metric definitions, promtool failures, or runtime proof blockers.

Stop when metric names, alert rules, runbook updates, and proof commands are clear.

# Reading manifest

Read `metrics_code` only if code changes are required.

# Work plan (agent should follow)

1. Verify previous stage gates.
2. Map required mainnet metrics to existing or planned emitters.
3. Implement missing metrics/rules/runbook entries.
4. Validate rules and docs.
5. Deliver through main if changed.
6. Collect post-main Prometheus/runtime proof.
7. Update stage report and ledger.

# Acceptance criteria (Definition of Done)

- Mainnet metrics and alerts are queryable or explicitly blocked with missing emitter reason.
- User-alert readiness is not bypassed.
- Ledger opens Stage `05` only after evidence.

# Implementation constraints

- No high-cardinality labels.
- No secrets/provider payloads in metrics or reports.

# Files to indicate (expected touched areas)

Final report must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No VLESS/VPN setup.
- No order canary.

# Quality gates (must run and pass)

Run promtool if available, focused Python gates if code changed, docs index, diff check, and post-main Prometheus proof when runtime changed.

# Final output: report format (strict)

Russian report: status, metrics added/proven, alerts/runbooks, user-alert readiness, file manifest, delivery/proof, next stage.
