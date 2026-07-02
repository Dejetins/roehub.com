---
prompt_name: mainnet-real-money-trading-v1-10-closure-cleanup-go-no-go
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Close the mainnet real-money trading v1 cycle with cleanup, evidence, and go/no-go record."
language: {implementation: python, agent_report: ru}
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo finalization rules"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "definition of done"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage source of truth"
  task_entrypoints:
    - path: docs/runbooks/exchange-execution.md
      why: "final runtime and rollback checks"
      inspect_symbols: ["Rollback", "Alerts", "Reconciliation"]
    - path: docs/runbooks/strategy-live-worker.md
      why: "producer scope cleanup"
      inspect_symbols: ["Producer Controls", "Acceptance Evidence"]
    - path: docs/runbooks/prod-dashboard-metrics-reference-ru.md
      why: "metrics dashboard closure"
      inspect_symbols: ["mainnet", "latency", "alerts"]
  conditional_bundles:
    browser:
      read_when: "final /strategies or /settings proof needs route context"
      paths: ["apps/api/wiring/modules/ui_strategies_dashboard.py", "apps/web"]
  consult_if_needed:
    - path: docs/runbooks/notifications-admin-alerts.md
      read_when: "alert delivery state is not clean"
style_references: [".codex/agents/stage_execution_ledger_template.md"]
hard_requirements: {stage_09_must_be_accepted: true, no_residual_exposure: true, ledger_completion_required: true}
task_toggles: {allow_cleanup: true, allow_real_order_submit: false}
skill_routing:
  - skill: browser-qa-evidence
    use_when: "final public UI proof is collected"
    timing: during verification
    reason: "prove user-visible final state"
  - skill: pre-ship-gate
    use_when: "before final delivery/readiness report"
    timing: before ship
    reason: "release-readiness and docs drift"
  - skill: publish-ci-deploy
    use_when: "final scoped docs/report changes need delivery"
    timing: before ship
    reason: "direct-main delivery"
target_envs: ["local", "macstudio", "roehub.com"]
required_literals: ["No residual open orders", "No unknown state", "mainnet remains scoped"]
non_goals: ["Do not run new orders.", "Do not broaden mainnet enablement."]
branch_policy: {default_branch: main, separate_branch_allowed: false, single_allowed_branch: null, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false, approval_required_for_branch_or_worktree: true}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: ["Stage 10 closure docs/report/ledger only unless cleanup bug fix is required"]
  foreign_changes_policy: "preserve unrelated"
  mixed_file_policy: "stage only owned hunks"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format: {language: ru, sections: ["status", "final_evidence", "residual_risk", "files", "go_no_go"]}
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"
validation_strategy:
  depth: ci_deploy
  e2e_required: true
  acceptance_surfaces: ["DB", "Redis", "Prometheus", "Monit", "browser/API", "docs", "CI/deploy if changed"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/10-closure-cleanup-go-no-go.md
proof_boundary:
  required_when: "final runtime proof is collected"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Requires origin/main, green CI, deploy/sync before final runtime proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["raw secrets", "tokens", "cookies", "raw provider payloads"]
remote_command_quoting: {applies_when: "SSH uses SQL/JSON", required_pattern: "quoted heredoc/stdin", forbidden_pattern: "nested inline SQL/JSON", temporary_files_allowed_only_when_task_requires_durable_artifact: true}
stage_execution_ledger: {path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, current_stage: "10", required_update: true, template: .codex/agents/stage_execution_ledger_template.md}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, prompt_pack_dir: .codex/agents/generated/mainnet-real-money-trading-v1/, stage_ledger: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    docs_runbooks: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/10-closure-cleanup-go-no-go.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1.md", "docs/runbooks/**", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/10-closure-cleanup-go-no-go.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
possible_secondary_touches: ["docs/architecture/live_execution/mainnet-real-money-trading-v1.md", "docs/runbooks", "docs/architecture/README.md"]
safety_notes: ["Closure must not broaden mainnet access. It records go/no-go and residual risks."]
---

# Task

Close `Mainnet Real-Money Trading v1` after accepted strategy-driven canaries.

## Context / Current State

This is not a new trading stage. It verifies cleanup, evidence completeness, and final go/no-go state.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `09` is `accepted`. If not, update Stage `10` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: do not place new orders; closure verifies cleanup, go/no-go evidence, and that mainnet remains scoped.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before work, read the ledger and confirm Stage `09 accepted`; if not, mark Stage `10 blocked`.
- Record `User required before start: none unless residual open position/unknown state needs operator action`.
- Do not place new orders.
- Prove no unexpected open orders/positions.
- Prove no unexplained Redis retry/DLQ/pending or reconciliation/unknown debt.
- Prove metrics, alerts, and user notifications are available.
- Collect final browser/API proof.
- Update final report, ledger status, docs index.

## Requirements (Should)

- Keep mainnet enabled only for accepted scoped surfaces; broad rollout remains separate.
- Summarize exact remaining expansion blockers.

## Requirements (Nice-to-have)

- Include a concise final go/no-go table by exchange/market.

# Context acquisition protocol

Read plan/ledger/runbooks first. Do not inspect implementation unless closure proof fails.

Reading budget: target `<= 10 files`; expand only for residual-state blockers or docs drift.

# Reading manifest

Use `browser` only if final UI proof needs route-level context.

# Work plan (agent should follow)

1. Verify Stage `09 accepted`.
2. Snapshot DB/Redis/exchange open orders/positions/metrics/alerts.
3. Collect browser/API proof.
4. Verify docs/runbooks/metrics index.
5. Write final closure report.
6. Update ledger to `completed` only if all gates pass.
7. Deliver scoped docs changes through main if needed.

# Acceptance criteria (Definition of Done)

- Ledger `ledger_status=completed`, `current_stage=none` only if clean.
- No residual unsafe state.
- Final report contains go/no-go and residual risks.

# Implementation constraints

- No new order.
- No deleting ledger rows as cleanup.
- Preserve immutable evidence.

# Files to indicate (expected touched areas)

Final report must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No broad mainnet rollout.
- No new markets/symbols.

# Quality gates (must run and pass)

Docs index, diff check, final runtime/browser/DB/Redis/Prometheus/Monit evidence.

# Final output: report format (strict)

Russian report: final status, evidence matrix, residual risk, go/no-go, file manifest, delivery status.
