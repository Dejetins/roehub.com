---
prompt_name: <prompt_name>
repo: <repo>
branch: <branch>
scope: "<one-sentence task scope>"

language:
  implementation: <python|...>
  agent_report: <ru|en|...>

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and delivery rules"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state snapshot"
    - path: <latest_executor_report_or_equivalent_if_available>
      why: "latest verified outcomes and unresolved items"

  task_entrypoints:
    - path: <entrypoint_path_1>
      why: "<why this is a canonical entrypoint>"
      inspect_symbols:
        - <symbol_1>
        - <symbol_2>
    - path: <entrypoint_path_2>
      why: "<why this is a canonical entrypoint>"
      inspect_symbols:
        - <symbol_1>
        - <symbol_2>

  conditional_bundles:
    <bundle_name_1>:
      read_when: "<condition for reading this bundle>"
      paths:
        - <path_a>
        - <path_b>

    <bundle_name_2>:
      read_when: "<condition for reading this bundle>"
      paths:
        - <path_a>
        - <path_b>

  consult_if_needed:
    - path: <path>
      read_when: "<blocker / ambiguity / conflict condition>"
    - path: <path>
      read_when: "<blocker / ambiguity / conflict condition>"

style_references:
  - <style_reference_path_1>
  - <style_reference_path_2>

hard_requirements:
  <hard_requirement_key_1>: true
  <hard_requirement_key_2>: true
  <hard_requirement_key_3>: true

task_toggles:
  <toggle_key_1>: true
  <toggle_key_2>: true
  <toggle_key_3>: true

skill_routing:
  - skill: <skill_name>
    use_when: "<condition that triggers this skill>"
    timing: "<before implementation|during investigation|during verification|before ship|if blocker>"
    reason: "<why this skill owns that boundary>"

target_envs:
  - <env_1>
  - <env_2>

required_literals:
  - "<literal_1>"
  - "<literal_2>"

required_keywords:
  - "<domain_keyword_that_bounds_scope_or_evidence>"
  - "<route_symbol_or_contract_name>"

non_goals:
  - "<explicit non-goal 1>"
  - "<explicit non-goal 2>"

branch_policy:
  default_branch: main
  separate_branch_allowed: <true|false>
  single_allowed_branch: <branch_name_or_null>
  stage_specific_branches_forbidden: true
  worktree_allowed: <true|false>
  stash_allowed: <true|false>
  approval_required_for_branch_or_worktree: true

change_ownership:
  parallel_main_expected: true
  owned_change_scope:
    - "<files_or_hunks_this_prompt_is_allowed_to_change>"
  foreign_changes_policy: "ignore and preserve unrelated changes from other chats"
  mixed_file_policy: "stage only owned hunks; block that file if safe hunk separation is impossible"
  forbidden_git_commands:
    - "git add ."
    - "git add -A"
    - "git add --all"
    - "git add :/"
    - "git add -- ."
    - "git add *"
    - "git restore --staged ."
    - "git restore --staged :/"
    - "git restore --staged *"
    - "git reset HEAD ."
    - "git reset ."
    - "git commit -a"
    - "git commit --all"
    - "git commit -am"
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"

final_report_format:
  language: <ru|en|...>
  sections:
    - "<section_1>"
    - "<section_2>"
    - "<section_3>"
    - "<section_4>"
    - "<section_5>"

quality_gates:
  - cmd: "<command_1>"
    expect: "<expected_result_1>"
  - cmd: "<command_2>"
    expect: "<expected_result_2>"

validation_strategy:
  depth: <tests_only|integration|e2e|browser_runtime|target_runtime|benchmark|ci_deploy>
  e2e_required: <true|false>
  acceptance_surfaces:
    - "<api|database|browser|target-host|adapter|benchmark|ci-deploy|none>"
  tests_only_allowed_reason: "<required if depth is tests_only>"
  evidence_target: "<stage_ledger_path_or_report_path>"

proof_boundary:
  required_when: "<local development runtime, clean local installation, cross-platform release candidate, or an explicitly authorized installation is in scope>"
  label: <N/A|local_source_development_proof|local_installation_runtime_proof|cross_platform_release_candidate_proof|authorized_installation_runtime_proof>
  changed_code_production_claim_allowed: <true|false>
  blocked_or_deferred_reason: "<required when the requested proof boundary is not reached or the target installation is not explicitly authorized>"

runtime_env_sources:
  roehub_env_file_order:
    - "$ROEHUB_ENV_FILE"
    - "/Users/daniildegtyarev/.config/roehub/roehub.env"
    - "/etc/roehub/roehub.env"
  report_only_key_presence: true
  forbidden_in_reports:
    - "raw secrets"
    - "tokens"
    - "credentials"
    - "cookies"

remote_command_quoting:
  applies_when: "SSH commands contain SQL, JSON, multiline payloads, apostrophes, backticks, or dollar signs"
  required_pattern: "quoted heredoc or stdin, such as <<'SQL', <<'JSON', --queries-file /dev/stdin, query=@-"
  forbidden_pattern: "nested inline --query \"... symbol='...'\" or equivalent fragile quoting"
  temporary_files_allowed_only_when_task_requires_durable_artifact: true

stage_execution_ledger:
  path: <stage_ledger_path_if_plan_or_prompt_pack>
  plan_doc: <architecture_or_implementation_plan_path>
  current_stage: <stage_id_or_name>
  required_update: <true|false>
  template: .codex/agents/legacy/stage_execution_ledger_template.md

prompt_pack_execution:
  mode: <manual_sequential|goal_driven>
  plan_doc: <architecture_or_implementation_plan_path>
  prompt_pack_dir: .codex/agents/generated/<pack_folder>/
  stage_ledger: <stage_ledger_path_if_plan_or_prompt_pack>
  goal_mode_optional: true
  goal_artifact_required: false

file_manifest:
  required_for_stage_prompts: <true|false>
  expected_groups:
    code:
      - "<src_or_app_path>"
    config_infra_migrations:
      - "<config_or_migration_path>"
    docs_runbooks:
      - "<docs_or_runbook_path>"
    prompt_artifacts:
      - "<prompt_or_template_path>"
    ledger_and_evidence:
      - "<ledger_or_evidence_path>"
  final_report_required_fields:
    - created
    - modified
    - deleted
    - outside_expected_paths
    - outside_expected_paths_justification
    - foreign_changes_excluded
    - mixed_files

expected_primary_touches:
  - "<path_directly_likely_to_change_1>"
  - "<path_directly_likely_to_change_2>"

possible_secondary_touches:
  - "<path_maybe_needed_for_exports_or_docs_or_tests_1>"
  - "<path_maybe_needed_for_exports_or_docs_or_tests_2>"

safety_notes:
  - "<safety_note_1>"
  - "<safety_note_2>"
---

# Task

<Describe the task in implementation-ready form.
State the goal, the intended change, the main invariants, and the exact boundaries.
Make it clear what must be delivered and what must remain unchanged.>

Done means:

- <done_condition_1>
- <done_condition_2>
- <done_condition_3>

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - <completed_item_1>
  - <completed_item_2>
  - <completed_item_3>
- open_items:
  - <open_item_1>
  - <open_item_2>
  - <open_item_3>
- contract_changes:
  - <contract_change_note_1>
  - <contract_change_note_2>
  - <contract_change_note_3>
- touched_paths:
  - <touched_path_note_1>
  - <touched_path_note_2>
  - <touched_path_note_3>
- risks:
  - <risk_1>
  - <risk_2>
  - <risk_3>
- next_focus:
  - <next_focus_1>
  - <next_focus_2>
  - <next_focus_3>

Additional context:

- <stable_context_note_1>
- <stable_context_note_2>

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only the scoped change described in this prompt.
- Preserve all explicitly protected contracts and invariants.
- Add or update targeted tests where needed.
- Update related exports / nearby docs when required.
- Keep the implementation deterministic and reviewable.
- For non-trivial implementation, run local gates plus the nearest meaningful real-boundary or end-to-end validation surface. Tests-only acceptance requires an explicit safe reason.
- If this prompt implements a plan stage, read the stage execution ledger before implementation and update it after validation and before the final report.
- If this prompt is part of a prompt pack, follow `prompt_pack_execution`: use the linked `plan_doc`, `prompt_pack_dir`, and `stage_ledger`; do not require or create `GOAL.md` unless the user explicitly asks for it.
- If a previous required stage is not accepted, stop unless this prompt explicitly repairs, supersedes, or unblocks that stage.
- Follow the front-matter `branch_policy`: work from `main` by default, do not create branches/worktrees/stashes unless explicitly allowed there, and never create per-stage branches.
- Follow the front-matter `change_ownership`: a dirty `main` checkout is expected, but only owned files/hunks for this prompt may be staged, committed, pushed, or reported as delivered.
- Do not let unrelated dirty files from other chats block the task. Preserve them, exclude them from staging, and call them out only when they affect this prompt's files or gates.
- Never use broad staging/unstaging/commit commands such as `git add .`, `git add -A`, `git add --all`, `git add :/`, `git add -- .`, `git add *`, `git restore --staged .`, `git restore --staged :/`, `git restore --staged *`, `git reset HEAD .`, `git reset .`, `git commit -a`, `git commit --all`, `git commit -am`, or `git commit .`; stage or unstage explicit owned paths/hunks only.
- If runtime or release evidence is in scope, use the front-matter `proof_boundary` label exactly and do not claim a stronger boundary than the evidence supports.
- If SSH commands include SQL, JSON, multiline payloads, apostrophes, backticks, or dollar signs, use quoted heredoc/stdin per `remote_command_quoting`; do not use nested inline quoting or temporary files created only to dodge quoting.
- If this is a stage prompt, maintain the front-matter `file_manifest` contract and include created/modified/deleted/outside-expected files in the final report.

- <task-specific must requirement 1>
- <task-specific must requirement 2>
- <task-specific must requirement 3>

## Requirements (Should)

- <task-specific should requirement 1>
- <task-specific should requirement 2>
- <task-specific should requirement 3>

## Requirements (Nice-to-have)

- <task-specific nice-to-have 1>
- <task-specific nice-to-have 2>

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. only the conditional bundle(s) required by touched contracts or failing checks
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once all of the following are true:

- changed contracts are identified,
- touched files are bounded,
- acceptance criteria are implementable without ambiguity,
- no unresolved public API or persistence-contract ambiguity remains.

Expand context only for:

- blockers,
- failing quality gates,
- unclear contracts,
- benchmark threshold conflicts,
- architecture conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules,
  - compact prior state,
  - latest verified prior result
- `task_entrypoints`:
  - canonical code/doc entrypoints for this task
- `conditional_bundles`:
  - read only when the stated condition applies
- `consult_if_needed`:
  - read only for blockers, ambiguity, or conflict resolution

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `<skill_name>`: use <timing> when <condition>; owns <boundary>. Do not preload if the condition does not apply.

1. <step_1>
2. <step_2>
3. <step_3>
4. <step_4>
5. <step_5>

# Acceptance criteria (Definition of Done)

- <acceptance_criterion_1>
- <acceptance_criterion_2>
- <acceptance_criterion_3>
- <acceptance_criterion_4>

# Implementation constraints

## Determinism & ordering

- Keep ordering deterministic and reviewable.
- Preserve stable identity / hash / key semantics unless the prompt explicitly allows change.
- Avoid hidden ordering changes from iteration order, import side effects, or async scheduling.

## API / contracts

- Do not change public or persisted contracts unless explicitly allowed.
- If a contract change is required, make it explicit, additive where possible, and test-covered.
- Keep internal-only metadata out of public request/persistence semantics unless explicitly requested.

## Documentation

- Update only directly relevant docs.
- Keep docs aligned with the delivered change.
- Do not turn local doc updates into repository-wide cleanup.
- For plan-stage work, update the stage execution ledger with status, concise results, evidence, blockers, touched contracts, and next-stage notes. Do not write secrets, tokens, cookies, passphrases, ciphertext, raw provider errors, or credentials into the ledger.

## Tests

- Add/update deterministic tests for the changed behavior.
- Prefer targeted tests over broad unrelated test churn.
- If config or DTO surfaces change, add parsing / compatibility coverage.

## Validation depth

- Treat lint/type/unit tests as local gates, not as sufficient acceptance for non-trivial stages.
- Validate through the nearest changed boundary: API/use-case, persistence/migration, browser runtime, target runtime, external adapter, benchmark/profile, CI/deploy, or production-safe smoke.
- If tests-only is sufficient, state why no contract, persistence, browser-visible, runtime, ops, performance, integration, or delivery surface is affected.
- Record validation evidence in the stage ledger or final report.

## Branch and workspace policy

- Use the front-matter `branch_policy` as the source of truth.
- Default to `main` unless the user explicitly approved one branch for the whole prompt pack.
- Do not create stage-specific branches, branch-specific worktrees, temporary checkouts, local folders, stashes, or auxiliary workflow files unless the exact artifact is explicitly allowed in `branch_policy`.

## Staged plan execution policy

- The execution source of truth is the linked `plan_doc`, `prompt_pack_dir`, and `stage_ledger`.
- `stage_ledger` determines the current stage, whether the next stage is allowed, and whether the overall plan is blocked or complete.
- `manual_sequential` means run only this stage and report the next allowed prompt.
- `goal_driven` means continue only while the ledger explicitly allows the next stage.
- Do not create or require `GOAL.md`; Codex Goal mode is an execution mode over the existing artifacts.

## Parallel main and scoped staging policy

- Multiple chats may share this `main` checkout; unrelated dirty files are expected and are not a blocker.
- Own only the files/hunks required by this prompt. Treat every other visible change as foreign.
- Before staging, compare the actual diff with `expected_primary_touches`, `possible_secondary_touches`, `file_manifest`, and the task scope.
- Stage only owned paths or owned hunks. For mixed files, use patch staging or report that exact file as blocked if safe separation is impossible.
- Before commit or push, inspect `git diff --cached --name-status`, verify that only owned paths are staged, then include `ROEHUB_SCOPED_STAGING_REVIEWED=1` in the commit/push command.
- Final report must list owned files included and foreign changes excluded when publish/delivery happened.

## Proof boundary and remote command safety

- Record `proof_boundary.label` in the stage ledger or final report whenever runtime, release-candidate, or authorized-installation evidence is used.
- `local_source_development_proof` covers source checks and the local development runtime only.
- `local_installation_runtime_proof` requires an isolated self-hosted installation with clean state or explicitly controlled test data.
- `cross_platform_release_candidate_proof` requires the verified source revision, green required CI, supported architecture evidence, release metadata and the required runtime/browser/API/service checks.
- `authorized_installation_runtime_proof` applies only to a concrete installation that the user explicitly placed in scope; never infer that authority from repository access.
- For any approved remote command with SQL/JSON/multiline payloads, use quoted heredoc/stdin and keep local shell, remote shell, and payload parsing separate.

# Files to indicate (expected touched areas)

Primary touches:

- `<primary_touch_1>`
- `<primary_touch_2>`
- `<primary_touch_3>`

Possible secondary touches:

- `<secondary_touch_1>`
- `<secondary_touch_2>`
- `<secondary_touch_3>`

Final report file manifest:

- created:
  - `<path or none>`
- modified:
  - `<path or none>`
- deleted:
  - `<path or none>`
- outside_expected_paths:
  - `<path or none>` — `<justification>`
- foreign_changes_excluded:
  - `<path or none>` — `<why it was not part of this prompt>`
- mixed_files:
  - `<path or none>` — `<owned hunks staged | blocked because safe separation was impossible>`

# Non-goals

- <non_goal_1>
- <non_goal_2>
- <non_goal_3>

# Quality gates (must run and pass)

- `<quality_gate_1>`
- `<quality_gate_2>`
- `<quality_gate_3>`

# Final output: report format (strict)

Your final message MUST be in <ru|en|...> and follow exactly:

1) **<section_1>**

2) **<section_2>**

3) **<section_3>**

4) **<section_4>**

5) **<section_5>**

It MUST also include the file manifest fields required by `file_manifest.final_report_required_fields` when this is a stage prompt.
