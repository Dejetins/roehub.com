---
prompt_name: mainnet-real-money-trading-v1-05-mainnet-adapter-enablement-no-submit
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Make exchange-execution mainnet-capable behind closed submit gates and prove no-submit behavior."
language: {implementation: python, agent_report: ru}
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo safety"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "mainnet adapter policy"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage gate"
  task_entrypoints:
    - path: apps/exchange_execution
      why: "adapter mode and runtime boundary"
      inspect_symbols: ["adapter_mode", "mainnet_hard_block", "run_once"]
    - path: configs/prod/exchange_execution.yaml
      why: "production runtime config"
      inspect_symbols: ["adapter_mode", "ledger", "rate_limit"]
    - path: docs/runbooks/exchange-execution.md
      why: "runtime runbook must match behavior"
      inspect_symbols: ["Adapter mode", "mainnet"]
  conditional_bundles:
    live_execution_dispatch:
      read_when: "no-submit proof needs intent/Redis integration"
      paths: ["src/trading/contexts/live_execution", "tests/unit/contexts/live_execution"]
    infra:
      read_when: "launchd/Monit/prometheus assets change"
      paths: ["infra/macos/launchd", "infra/scripts/monit", "infra/macos/prometheus"]
  consult_if_needed:
    - path: docs/runbooks/exchange-secret-management.md
      read_when: "credential resolver scope is unclear"
style_references: [".codex/agents/stage_execution_ledger_template.md"]
hard_requirements: {stage_04_must_be_accepted: true, mainnet_submit_gate_closed: true, no_real_order_submit: true}
task_toggles: {allow_runtime_config_changes: true, allow_mainnet_capable_mode: true, allow_mainnet_orders: false}
skill_routing:
  - skill: contract-impact-analysis
    use_when: "config/runtime mode semantics change"
    timing: during investigation
    reason: "classify config compatibility"
  - skill: publish-ci-deploy
    use_when: "accepted runtime changes need delivery"
    timing: before ship
    reason: "CI/deploy/post-main proof"
target_envs: ["local", "macstudio"]
required_literals: ["submit gate closed", "No real mainnet order submit", "post_main_production_runtime_proof"]
non_goals: ["Do not set futures leverage/margin.", "Do not submit/cancel orders."]
branch_policy: {default_branch: main, separate_branch_allowed: false, single_allowed_branch: null, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false, approval_required_for_branch_or_worktree: true}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: ["Stage 05 exchange-execution mode/config/report/ledger only"]
  foreign_changes_policy: "preserve unrelated"
  mixed_file_policy: "stage only owned hunks"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format: {language: ru, sections: ["status", "config", "no_submit_evidence", "files", "next_stage"]}
quality_gates:
  - cmd: "uv run ruff check apps/exchange_execution src/trading/contexts/live_execution tests"
    expect: "passes"
  - cmd: "uv run pyright apps/exchange_execution src/trading/contexts/live_execution tests"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["exchange-execution health", "Redis", "DB", "Prometheus", "no-submit mainnet guard"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/05-mainnet-adapter-enablement-no-submit.md
proof_boundary:
  required_when: "changed runtime behavior is verified"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Requires origin/main, green CI, deploy/sync, then runtime proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["raw secrets", "tokens", "credentials", "cookies"]
remote_command_quoting: {applies_when: "SSH uses SQL/JSON", required_pattern: "quoted heredoc/stdin", forbidden_pattern: "nested inline SQL/JSON", temporary_files_allowed_only_when_task_requires_durable_artifact: true}
stage_execution_ledger: {path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, current_stage: "05", required_update: true, template: .codex/agents/stage_execution_ledger_template.md}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, prompt_pack_dir: .codex/agents/generated/mainnet-real-money-trading-v1/, stage_ledger: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    code: ["apps/exchange_execution/**", "src/trading/contexts/live_execution/**"]
    config_infra_migrations: ["configs/prod/exchange_execution.yaml", "infra/macos/**"]
    docs_runbooks: ["docs/runbooks/exchange-execution.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/05-mainnet-adapter-enablement-no-submit.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches: ["apps/exchange_execution", "configs/prod/exchange_execution.yaml", "docs/runbooks/exchange-execution.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/05-mainnet-adapter-enablement-no-submit.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
possible_secondary_touches: ["src/trading/contexts/live_execution", "infra/macos", "docs/architecture/README.md"]
safety_notes: ["Mainnet-capable is not mainnet-enabled. Submit gate must stay closed."]
---

# Task

Make `exchange-execution` mainnet-capable behind fail-closed submit gates and prove no real mainnet order can pass yet.

## Context / Current State

`exchange-execution` currently supports disabled/testnet adapter behavior and mainnet hard-block. This stage prepares explicit gated mainnet capability without opening submit.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stages `02`, `03`, and `04` are `accepted`. If not, update Stage `05` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: mainnet-capable adapter mode must remain no-submit until a later explicit canary gate opens a scoped money-moving window.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before any implementation or runtime action, read the stage ledger and confirm Stage `04 accepted`; if not, write Stage `05 blocked`, update the ledger, and stop.
- Record `User required before start: nothing beyond accepted prior gates`.
- Add mainnet-capable config/mode only if submit gate remains closed by default.
- Prove mainnet intents are observed/rejected/blocked before provider submit.
- Prove health/readiness/metrics expose gated state.
- No real order submit, cancel, or status endpoint for mainnet.

## Requirements (Should)

- Keep testnet behavior backward-compatible.
- Update runbook with exact rollback.

## Requirements (Nice-to-have)

- Add explicit readiness dependency `mainnet_submit_gate`.

# Context acquisition protocol

Read plan/ledger and exchange-execution entrypoints first. Expand to dispatch/infra only for required integration.

Reading budget: target `<= 12 files`; expand for config ambiguity, failing tests, or runtime proof blockers.

# Reading manifest

Use conditional bundles only when implementation requires them.

# Work plan (agent should follow)

1. Verify prior gates.
2. Implement gated mainnet-capable mode.
3. Prove no-submit with controlled mainnet-shaped intent.
4. Run local gates.
5. Deliver to main and collect post-main production proof.
6. Update report/ledger.

# Acceptance criteria (Definition of Done)

- Mainnet submit remains closed until later canary stages.
- Runtime proof shows blocked decision before provider call.
- Ledger opens Stage `06` only after post-main proof.

# Implementation constraints

- No provider side effect.
- No raw provider payloads in evidence.

# Files to indicate (expected touched areas)

Final report must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No futures config mutation.
- No canary order.

# Quality gates (must run and pass)

Run focused ruff, pyright, pytest, docs index, runtime health/Redis/DB/metrics no-submit proof after delivery.

# Final output: report format (strict)

Russian report with status, config, no-submit evidence, rollback, file manifest, delivery, next stage.
