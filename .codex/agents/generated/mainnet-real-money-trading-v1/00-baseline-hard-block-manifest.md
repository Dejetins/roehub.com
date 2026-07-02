---
prompt_name: mainnet-real-money-trading-v1-00-baseline-hard-block-manifest
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Create Stage 00 baseline report and verify mainnet hard blocks before any real-money work."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo workflow, proof-boundary and scoped staging rules"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "source plan for this prompt pack"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage source of truth"
  task_entrypoints:
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "paper/testnet closure facts"
      inspect_symbols: ["ledger_status", "Stage 14", "mainnet"]
    - path: docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md
      why: "universal gateway foundation and mainnet blockers"
      inspect_symbols: ["Stage 17", "mainnet", "exchange-execution"]
    - path: docs/runbooks/exchange-execution.md
      why: "current exchange-execution runtime contract"
      inspect_symbols: ["Mainnet submit", "Adapter mode", "Stage 17 Alert Actions"]
  conditional_bundles:
    runtime_inventory:
      read_when: "read-only Mac Studio readiness facts are needed"
      paths:
        - docs/runbooks/strategy-live-worker.md
        - infra/macos/prometheus/rules/live-execution-stage17.rules.yml
  consult_if_needed:
    - path: docs/architecture/README.md
      read_when: "docs index drift or linked docs are ambiguous"
style_references:
  - .codex/agents/stage_execution_ledger_template.md
hard_requirements:
  no_mainnet_order_submit: true
  user_required_before_start_must_be_recorded: true
  stage_ledger_update_required: true
task_toggles:
  allow_runtime_read_only_checks: true
  allow_code_changes: false
skill_routing:
  - skill: architecture-review
    use_when: "checking current-state facts and hard-block completeness"
    timing: during investigation
    reason: "keeps baseline evidence distinct from inference"
  - skill: publish-ci-deploy
    use_when: "only after docs-only Stage 00 is accepted and scoped docs changed"
    timing: before ship
    reason: "direct-main delivery if this stage changes docs"
target_envs: ["local", "macstudio"]
required_literals:
  - "User required before start: nothing"
  - "No mainnet order submit"
non_goals:
  - "Do not enable adapters, change runtime config, or place orders."
  - "Do not ask for or print secrets."
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
    - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/00-baseline-hard-block-manifest.md
    - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
    - docs/architecture/README.md
  foreign_changes_policy: "ignore and preserve unrelated changes from other chats"
  mixed_file_policy: "stage only owned hunks; block that file if safe hunk separation is impossible"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git add :/", "git add -- .", "git add *", "git restore --staged .", "git reset .", "git commit -a", "git commit --all", "git commit -am"]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format:
  language: ru
  sections: ["status", "user_required", "evidence", "files", "next_stage"]
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after any docs index regeneration"
  - cmd: "git diff --check"
    expect: "passes"
validation_strategy:
  depth: target_runtime
  e2e_required: false
  acceptance_surfaces: ["docs", "read-only runtime inventory if used"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/00-baseline-hard-block-manifest.md
proof_boundary:
  required_when: "Mac Studio read-only checks are used"
  label: target_host_readiness_pre_main
  changed_code_production_claim_allowed: false
  blocked_or_deferred_reason: "No changed-code runtime proof is required for docs-only baseline."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["raw secrets", "tokens", "credentials", "cookies"]
remote_command_quoting:
  applies_when: "SSH commands contain SQL, JSON, multiline payloads, apostrophes, backticks, or dollar signs"
  required_pattern: "quoted heredoc or stdin"
  forbidden_pattern: "nested inline payload quoting"
  temporary_files_allowed_only_when_task_requires_durable_artifact: true
stage_execution_ledger:
  path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
  current_stage: "00"
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
    docs_runbooks: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/00-baseline-hard-block-manifest.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md", "docs/architecture/README.md"]
    prompt_artifacts: []
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/00-baseline-hard-block-manifest.md"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches:
  - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/00-baseline-hard-block-manifest.md
  - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "This stage must not mutate runtime, keys, balances, or exchange state."
---

# Task

Create Stage `00` baseline and hard-block manifest for `Mainnet Real-Money Trading v1`.

Done means:

- current paper/testnet and universal gateway foundation are summarized from existing ledgers;
- current mainnet blockers are listed with evidence and no hidden assumptions;
- `User required before start: nothing` is recorded for Stage `00`;
- the stage ledger is updated with `00 accepted` or `00 blocked`;
- no mainnet order submit, runtime config change, exchange config change, or secret access occurs.

## Context / Current State

- The paper/testnet strategy producer cycle is closed; mainnet real money remained out of scope.
- The universal order gateway Stage `17` accepted testnet production-readiness proof but kept mainnet blocked.
- This new plan is separate and must stop later stages until Telegram/user prerequisite gates pass.
- Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: no previous stage; before any implementation or runtime action, read `stage_ledger` and confirm `current_stage=00` with Stage `00` still `pending` or `in_progress`. If the ledger points elsewhere, update Stage `00` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: do not infer readiness from chat history; use `stage_ledger`, this prompt, and the plan only.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Read the plan and ledger first.
- Verify the current stage is `00`.
- Record `User required before start: nothing`.
- Separate observed facts from inference.
- Create `00-baseline-hard-block-manifest.md`.
- Update the ledger before final report.
- If you use Mac Studio, label evidence `target_host_readiness_pre_main`.
- Do not claim changed-code runtime proof.

## Requirements (Should)

- Prefer existing accepted ledgers over broad source reading.
- Keep the report compact and Russian.

## Requirements (Nice-to-have)

- Include a short table of reusable foundation components.

# Context acquisition protocol

Read in this order: `.codex/AGENTS.md`, this plan, the stage ledger, then only task entrypoints needed for foundation facts. Do not eager-load every linked document.

Reading budget: target `<= 8 files` and `<= 50k tokens` before writing. Expand only for blockers, conflicting status, or missing hard-block evidence.

Stop when current stage, foundation facts, hard blockers, touched files, and acceptance criteria are clear.

# Reading manifest

Always read the three `always_read` paths. Read `runtime_inventory` only if you need read-only runtime proof. Use `docs/architecture/README.md` only for docs index ambiguity.

# Work plan (agent should follow)

1. Confirm ledger `current_stage=00`.
2. Confirm `User required before start: nothing`.
3. Gather foundation and blocker facts from accepted ledgers/runbooks.
4. Write the Stage `00` report with facts, blockers, non-goals, file manifest, and next handoff.
5. Update ledger: status, evidence, next stage permission.
6. Run docs index check and `git diff --check`.
7. If accepted and docs changed, use scoped direct-main delivery through `publish-ci-deploy`; otherwise record blocker.

# Acceptance criteria (Definition of Done)

- Stage report exists and is secret-safe.
- Ledger marks `00 accepted` only if all required docs/evidence were written and checks pass.
- Next stage allowed only for Stage `01`.
- No runtime mutation or exchange side effect occurred.

# Implementation constraints

- Docs-only unless a read-only runtime inventory is explicitly needed.
- No secrets, no provider calls that require trade keys, no orders.
- Preserve foreign changes.

# Files to indicate (expected touched areas)

- `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/00-baseline-hard-block-manifest.md`
- `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `docs/architecture/README.md` if regenerated.
- Final file manifest must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- Do not implement mainnet features.
- Do not resolve Telegram/VPN.
- Do not ask user for API keys.

# Quality gates (must run and pass)

- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`

# Final output: report format (strict)

Report in Russian:

- Stage status.
- User required before start.
- Evidence collected.
- Files created/modified/deleted/outside expected paths.
- Whether next stage is allowed.
- Commit/push/deploy status if delivery occurred.
