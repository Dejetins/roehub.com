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

non_goals:
  - "<explicit non-goal 1>"
  - "<explicit non-goal 2>"

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

stage_execution_ledger:
  path: <stage_ledger_path_if_plan_or_prompt_pack>
  plan_doc: <architecture_or_implementation_plan_path>
  current_stage: <stage_id_or_name>
  required_update: <true|false>
  template: .codex/agents/stage_execution_ledger_template.md

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
- If this prompt implements a plan stage, read the stage execution ledger before implementation and update it after validation and before the final report.
- If a previous required stage is not accepted, stop unless this prompt explicitly repairs, supersedes, or unblocks that stage.

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

# Files to indicate (expected touched areas)

Primary touches:

- `<primary_touch_1>`
- `<primary_touch_2>`
- `<primary_touch_3>`

Possible secondary touches:

- `<secondary_touch_1>`
- `<secondary_touch_2>`
- `<secondary_touch_3>`

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
