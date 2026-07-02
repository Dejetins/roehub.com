---
prompt_name: mainnet-real-money-trading-v1-08-strategy-producer-live-mode-enablement
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Enable scoped strategy producer live mode after ops canaries, without broad mainnet fan-out."
language: {implementation: python, agent_report: ru}
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "live producer policy"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage source of truth"
  task_entrypoints:
    - path: apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      why: "producer mode allowlist and metrics"
      inspect_symbols: ["_PRODUCER_ALLOWED_MODES", "live", "allowlist"]
    - path: docs/runbooks/strategy-live-worker.md
      why: "producer runtime runbook"
      inspect_symbols: ["Allowed modes", "kill switch", "alerts"]
    - path: src/trading/contexts/strategy
      why: "strategy run/profile live mode behavior"
      inspect_symbols: ["LiveStrategyProfile", "StrategyRun"]
  conditional_bundles:
    ui_strategy:
      read_when: "live mode status is browser-visible"
      paths: ["apps/api/wiring/modules/ui_strategies_dashboard.py", "apps/web"]
    metrics:
      read_when: "producer metrics/rules change"
      paths: ["infra/macos/prometheus/rules/strategy-producer.rules.yml", "apps/worker/strategy_live_runner"]
  consult_if_needed:
    - path: docs/runbooks/exchange-execution.md
      read_when: "execution kill-switch interaction is unclear"
style_references: [".codex/agents/stage_execution_ledger_template.md"]
hard_requirements: {stage_07_must_be_accepted: true, scoped_allowlist_required: true, no_broad_mainnet_fanout: true}
task_toggles: {allow_live_mode_code: true, allow_real_strategy_order: false}
skill_routing:
  - skill: browser-qa-evidence
    use_when: "live/mainnet status changes in /strategies"
    timing: during verification
    reason: "prove user-visible controls and warnings"
  - skill: publish-ci-deploy
    use_when: "accepted runtime changes need delivery"
    timing: before ship
    reason: "post-main producer proof"
target_envs: ["local", "macstudio", "roehub.com"]
required_literals: ["live mode scoped allowlist", "No broad mainnet fan-out", "kill switch"]
non_goals: ["Do not run strategy-driven real order yet.", "Do not allow all users/strategies."]
branch_policy: {default_branch: main, separate_branch_allowed: false, single_allowed_branch: null, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false, approval_required_for_branch_or_worktree: true}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: ["Stage 08 producer live-mode files only"]
  foreign_changes_policy: "preserve unrelated"
  mixed_file_policy: "stage only owned hunks"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format: {language: ru, sections: ["status", "producer_controls", "evidence", "files", "next_stage"]}
quality_gates:
  - cmd: "uv run ruff check apps/worker/strategy_live_runner src/trading/contexts/strategy src/trading/contexts/live_execution apps tests"
    expect: "passes"
  - cmd: "uv run pyright apps/worker/strategy_live_runner src/trading/contexts/strategy src/trading/contexts/live_execution apps tests"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/live_execution tests/unit/apps"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["producer health", "DB", "Redis", "Prometheus", "browser/API", "no real order"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/08-strategy-producer-live-mode-enablement.md
proof_boundary:
  required_when: "producer runtime changes are verified"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Requires origin/main, green CI, deploy/sync, then runtime proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["raw secrets", "tokens", "cookies"]
remote_command_quoting: {applies_when: "SSH uses SQL/JSON", required_pattern: "quoted heredoc/stdin", forbidden_pattern: "nested inline SQL/JSON", temporary_files_allowed_only_when_task_requires_durable_artifact: true}
stage_execution_ledger: {path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, current_stage: "08", required_update: true, template: .codex/agents/stage_execution_ledger_template.md}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, prompt_pack_dir: .codex/agents/generated/mainnet-real-money-trading-v1/, stage_ledger: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    code: ["apps/worker/strategy_live_runner/**", "src/trading/contexts/strategy/**", "src/trading/contexts/live_execution/**", "apps/api/**", "apps/web/**"]
    config_infra_migrations: ["configs/prod/**", "infra/macos/**"]
    docs_runbooks: ["docs/runbooks/strategy-live-worker.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/08-strategy-producer-live-mode-enablement.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches: ["apps/worker/strategy_live_runner", "docs/runbooks/strategy-live-worker.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/08-strategy-producer-live-mode-enablement.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
possible_secondary_touches: ["src/trading/contexts/strategy", "src/trading/contexts/live_execution", "apps/api", "apps/web", "configs/prod", "infra/macos", "docs/architecture/README.md"]
safety_notes: ["Live mode must be scoped to explicit user/strategy/exchange/market allowlists and kill switches."]
---

# Task

Enable strategy producer `live` mode only for explicit scoped mainnet canary subjects.

## Context / Current State

The producer is currently accepted for `paper,testnet` only. This stage opens live-mode plumbing but must not run a strategy-driven real order yet.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `07` is `accepted`. If not, update Stage `08` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: this stage opens scoped live-mode plumbing only; broad mainnet fan-out and strategy-driven real orders remain blocked until Stage `09`.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before implementation, read the stage ledger and confirm Stage `07 accepted`; if not, mark Stage `08 blocked`.
- Record `User required before start: nothing beyond accepted Stage 07; no broad live enablement`.
- Add scoped `live` mode support with allowlists and kill switch.
- Prove default blocks live mode when not scoped.
- Prove scoped live mode can produce safe source-event/risk behavior without real order submit in this stage.
- Update runbook and metrics.

## Requirements (Should)

- Keep `paper,testnet` behavior unchanged.
- Add clear UI/readiness reasons.

## Requirements (Nice-to-have)

- Add a dry-run live source event proof row.

# Context acquisition protocol

Read plan/ledger and producer entrypoint first. Expand only for UI/metrics/runtime proof.

Reading budget: target `<= 12 files`; expand only for producer mode conflicts or failing runtime proof.

# Reading manifest

Read `ui_strategy` only for browser-visible changes; read `metrics` only for rules/emitters.

# Work plan (agent should follow)

1. Verify previous stage accepted.
2. Implement scoped live mode and fail-closed defaults.
3. Prove disabled/missing-allowlist/kill-switch blocks.
4. Prove scoped source-event path without real order submit.
5. Deliver and collect post-main runtime/browser/metrics proof.
6. Update report/ledger.

# Acceptance criteria (Definition of Done)

- No broad live fan-out.
- No real order submit in this stage.
- Stage `09` opens only with scoped automatic strategy window pending.

# Implementation constraints

- Do not expose raw connection ids as metrics labels if high-cardinality.
- Preserve foreign changes.

# Files to indicate (expected touched areas)

Final report must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No strategy-driven mainnet order.
- No Telegram/VPN setup.

# Quality gates (must run and pass)

Focused producer/API/UI gates, docs index, post-main launchd/Monit/Prometheus/browser proof.

# Final output: report format (strict)

Russian report: producer controls, block/allow evidence, no-order proof, file manifest, delivery, next stage.
