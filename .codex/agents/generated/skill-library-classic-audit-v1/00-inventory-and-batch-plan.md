---
prompt_name: 00-inventory-and-batch-plan
repo: roehub.com
branch: main
scope: "Inventory every local skill/plugin SKILL.md and create a complete subagent audit batch plan."
model_preferences:
  primary_agent_model: gpt-5.5
  reasoning_effort: xhigh
language:
  implementation: docs
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and prompt-pack execution policy"
    - path: docs/architecture/agents/skill-library-classic-audit-v1.md
      why: "plan_doc and audit scope"
    - path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md
      why: "stage_ledger and current stage"
  task_entrypoints:
    - path: /Users/daniildegtyarev/.codex/skills
      why: "local skill library root"
    - path: /Users/daniildegtyarev/.codex/plugins/cache
      why: "plugin skill cache root"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  full_skill_inventory: true
  configured_roots_required: true
  canonical_path_deduplication: true
  source_skill_files_read_only: true
task_toggles:
  allow_source_skill_edits: false
  allow_branch_creation: false
  allow_worktree: false
  allow_stash: false
skill_routing:
  - skill: staged-plan-runner
    use_when: "executing this prompt from plan_doc/prompt_pack_dir/stage_ledger"
    timing: "before stage actions"
    reason: "owns goal_driven stage gating"
  - skill: prompt-manager
    use_when: "checking generated audit prompt readiness"
    timing: "before final report"
    reason: "owns prompt artifact quality"
target_envs:
  - local checkout
  - local Codex home
required_literals:
  - "Before solving or editing, emit one short commentary update stating the exact reasoning/root-cause check you are doing."
  - "Do thought experiments before making changes."
  - "Use them to test competing explanations, edge cases, failure modes, and whether the apparent fix would actually solve the root cause."
  - "Do not stop at the first plausible explanation."
  - "Do not stop after a superficial answer. Validate the result with evidence, edge cases, and tests before finalizing."
  - "all scripts and saved audit artifacts stay local to the current machine"
non_goals:
  - "Do not audit the skill bodies deeply in Stage 00; only inventory and batching."
  - "Do not edit source skills/plugins."
  - "Do not create branches, worktrees or stashes."
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  single_allowed_branch: null
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
change_ownership:
  parallel_main_expected: true
  owned_change_scope:
    - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md"
    - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md"
    - "docs/architecture/README.md if docs index changes"
  foreign_changes_policy: "ignore and preserve unrelated changes from other chats"
  mixed_file_policy: "stage only owned hunks; block that file if safe hunk separation is impossible"
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index"
    expect: "updates docs/architecture/README.md if needed"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"
validation_strategy:
  depth: architecture_review
  e2e_required: true
  acceptance_surfaces:
    - "complete local SKILL.md inventory"
    - "batch plan covering every discovered skill"
    - "stage_ledger handoff"
  evidence_target: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md
proof_boundary:
  required_when: "Mac Studio, runtime, deploy or production smoke is not in scope"
  label: none
  changed_code_production_claim_allowed: false
  blocked_or_deferred_reason: "N/A"
stage_execution_ledger:
  path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md
  plan_doc: docs/architecture/agents/skill-library-classic-audit-v1.md
  current_stage: "00"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
prompt_pack_execution:
  mode: goal_driven
  plan_doc: docs/architecture/agents/skill-library-classic-audit-v1.md
  prompt_pack_dir: .codex/agents/generated/skill-library-classic-audit-v1/
  stage_ledger: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md
  goal_mode_optional: true
  goal_artifact_required: false
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    code: []
    docs_runbooks:
      - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md"
      - "docs/architecture/README.md"
    prompt_artifacts: []
    ledger_and_evidence:
      - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md"
  final_report_required_fields:
    - created
    - modified
    - deleted
    - outside_expected_paths
    - outside_expected_paths_justification
    - foreign_changes_excluded
    - mixed_files
expected_primary_touches:
  - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md"
  - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/README.md"
safety_notes:
  - "Source skills/plugins are read-only."
  - "Do not persist raw secrets, env dumps or large copied skill bodies."
---

# Task

Run Stage `00`: inventory every local `SKILL.md` and create a complete batch plan
for classic subagent audit.

Before doing anything else, emit this short commentary update requirement and
follow it:

```text
Before solving or editing, emit one short commentary update stating the exact reasoning/root-cause check you are doing.

Do thought experiments before making changes.
Use them to test competing explanations, edge cases, failure modes, and whether the apparent fix would actually solve the root cause.
Do not stop at the first plausible explanation.

Do not stop after a superficial answer. Validate the result with evidence, edge cases, and tests before finalizing.
```

All scripts and saved audit artifacts stay local to the current machine.

## Requirements (Must)

- Previous required stage ledger gate: Stage `00` has no previous stage; verify `current_stage: 00` and linked artifacts.
- Find every readable `SKILL.md` under the configured roots.
- Verify every configured root from `plan_doc`. If any root is unreadable or cannot be verified, mark Stage `00` as `blocked` unless the user explicitly approved reduced scope for this run.
- Deduplicate discovered `SKILL.md` files by canonical/resolved path before assigning `skill_id`; `/Users/daniildegtyarev/.codex/skills/.system` may overlap `/Users/daniildegtyarev/.codex/skills`.
- Compute or record `sha256` for every discovered skill.
- Assign `skill_id`, `source`, `skill_type` and `batch_id`.
- Create a batch plan that covers every canonical discovered skill exactly once.
- Do not edit source skills/plugins.
- Do not create any scratch/local-state directory unless the user explicitly approves the path and lifecycle first.
- Update `stage_ledger` after validation and before final report.

## Final Output

Respond in Russian with:

1. **Результат Stage 00**
2. **Покрытие библиотеки**
3. **Batch plan**
4. **Blockers or missing roots**
5. **Quality gates**
6. **File manifest**
7. **Next-stage handoff**
