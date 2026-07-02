---
prompt_name: mainnet-real-money-trading-v1-03-risk-caps-kill-switch-policy
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Implement mainnet risk caps, capital manifest, and kill-switch policy before submit."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo workflow"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "risk policy source"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage gate and budget blocker"
  task_entrypoints:
    - path: src/trading/contexts/live_execution
      why: "risk gate and execution intent domain"
      inspect_symbols: ["risk", "capital", "ExecutionIntent"]
    - path: apps/api/dto/ui_execution.py
      why: "UI/API execution readiness DTOs"
      inspect_symbols: ["ExecutionReadinessResponse", "kill_switch"]
    - path: apps/web/templates/pages/strategies.html
      why: "strategy UI live readiness"
      inspect_symbols: ["live", "risk", "status"]
  conditional_bundles:
    migrations:
      read_when: "new durable caps/audit tables are needed"
      paths:
        - alembic/versions
        - tests/unit/apps
    metrics:
      read_when: "risk/cap metrics are added"
      paths:
        - apps/api/monitoring.py
        - infra/macos/prometheus/rules/live-execution-stage17.rules.yml
  consult_if_needed:
    - path: docs/runbooks/exchange-execution.md
      read_when: "kill-switch runbook semantics are unclear"
style_references:
  - .codex/agents/stage_execution_ledger_template.md
hard_requirements:
  capital_manifest_required: true
  no_order_submit: true
  kill_switch_fail_closed: true
task_toggles:
  allow_schema_changes: true
  allow_browser_changes: true
skill_routing:
  - skill: contract-impact-analysis
    use_when: "API/DTO/persistence/config changes are introduced"
    timing: during investigation
    reason: "classify compatibility and rollout"
  - skill: browser-qa-evidence
    use_when: "strategy UI readiness or warning changes are visible"
    timing: during verification
    reason: "prove user-visible blocked/ready states"
  - skill: publish-ci-deploy
    use_when: "accepted code changes need main delivery"
    timing: before ship
    reason: "CI/deploy and post-main proof"
target_envs: ["local", "macstudio", "roehub.com"]
required_literals:
  - "per-order canary cap 15 USDT"
  - "global total cap 60 USDT until manifest resolves ambiguity"
  - "market orders only"
non_goals:
  - "Do not submit orders."
  - "Do not enable live producer mode."
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
  owned_change_scope: ["Stage 03 risk/caps/kill-switch files only"]
  foreign_changes_policy: "preserve unrelated changes"
  mixed_file_policy: "stage only owned hunks"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format:
  language: ru
  sections: ["status", "capital_manifest", "risk_policy", "evidence", "next_stage"]
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/live_execution apps tests"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/live_execution apps tests"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
validation_strategy:
  depth: e2e
  e2e_required: true
  acceptance_surfaces: ["API", "DB", "browser", "metrics"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/03-risk-caps-kill-switch-policy.md
proof_boundary:
  required_when: "runtime/code changes are verified"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Requires main, green CI, deploy/sync before runtime proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["raw secrets", "tokens", "credentials", "cookies"]
remote_command_quoting:
  applies_when: "SSH commands contain SQL/JSON"
  required_pattern: "quoted heredoc/stdin"
  forbidden_pattern: "nested inline SQL/JSON"
  temporary_files_allowed_only_when_task_requires_durable_artifact: true
stage_execution_ledger:
  path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
  current_stage: "03"
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
    code: ["src/trading/contexts/live_execution/**", "apps/api/**", "apps/web/**"]
    config_infra_migrations: ["alembic/versions/**", "configs/prod/**"]
    docs_runbooks: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/03-risk-caps-kill-switch-policy.md", "docs/runbooks/exchange-execution.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches:
  - src/trading/contexts/live_execution
  - apps/api
  - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/03-risk-caps-kill-switch-policy.md
  - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
possible_secondary_touches:
  - apps/web
  - alembic/versions
  - infra/macos/prometheus/rules
  - docs/runbooks/exchange-execution.md
  - docs/architecture/README.md
safety_notes:
  - "This stage must prove reject paths and kill switch before any mainnet submit can exist."
---

# Task

Implement or harden mainnet risk caps, capital allocation manifest, and kill-switch policy.

Done means:

- Stage `02 accepted`;
- budget ambiguity is resolved by a durable capital allocation manifest or Stage `03 blocked`;
- risk gate blocks orders above `15 USDT` canary cap and above accepted global/per-market caps;
- kill switches exist for global, user, strategy, exchange, and market scopes;
- UI/API show blocked/ready reasons;
- no order submit occurs.

## Context / Current State

User stated `15 USDT` max notional for first canary, `20 USDT` per market, and `60 USDT` total. Because four market surfaces exist, Stage `03` must record an explicit allocation manifest before any later submit stage.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `02` is `accepted`. If not, update Stage `03` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: capital ambiguity (`20 USDT` per market vs `60 USDT` total) must be resolved in a capital allocation manifest before any later money-moving stage can run.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before any implementation or runtime action, read the stage ledger and confirm Stage `02 accepted`; if not, write Stage `03 blocked`, update the ledger, and stop.
- Record `User required before start: explicit capital allocation manifest resolving 20-per-market vs 60-total`.
- If unresolved, mark Stage `03 blocked`.
- Add durable caps/audit if missing.
- Fail closed by default.
- Add/verify browser-visible blocked/ready states.
- Add/verify metrics for cap rejects and kill-switch state.

## Requirements (Should)

- Prefer additive schema/API changes.
- Keep cap labels low-cardinality.

## Requirements (Nice-to-have)

- Provide an operator SQL/API snippet for reading current caps without secrets.

# Context acquisition protocol

Read plan/ledger, then risk/intent entrypoints. Do not inspect exchange adapters unless risk integration requires it.

Reading budget: target `<= 12 files` and `<= 60k tokens`. Expand only for schema conflicts, UI blockers, or missing risk gate integration.

Stop when capital manifest, risk/cap model, touched files, and validation commands are clear.

# Reading manifest

Use `migrations` only if persistence changes are needed. Use `metrics` only if metrics/rules are touched.

# Work plan (agent should follow)

1. Verify previous stages accepted and capital manifest availability.
2. If user allocation is unresolved, write blocked report and ledger update.
3. Implement risk/cap/kill-switch additions if needed.
4. Prove reject paths via API/DB and UI.
5. Prove no dispatch/order side effects.
6. Run gates, deliver to main, collect post-main proof.
7. Update report and ledger.

# Acceptance criteria (Definition of Done)

- Mainnet risk defaults are closed.
- Any missing cap, stale account state, open kill switch, unsupported order model, or over-limit notional rejects before dispatch.
- Stage `04` opens only after runtime evidence.

# Implementation constraints

- Do not call exchange submit endpoints.
- Do not store raw sensitive exchange data in cap/audit rows.

# Files to indicate (expected touched areas)

List exact touched files and explain any outside expected path.

Final file manifest must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No adapter enablement.
- No futures account config mutation.
- No canary orders.

# Quality gates (must run and pass)

Run focused ruff, pyright, unit tests, docs index, browser/API/DB/metrics runtime proof.

# Final output: report format (strict)

Russian report with status, manifest, caps, kill-switch evidence, no-submit proof, file manifest, delivery status, next stage.
