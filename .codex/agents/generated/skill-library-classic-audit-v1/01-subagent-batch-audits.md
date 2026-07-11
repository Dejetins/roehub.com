---
prompt_name: 01-subagent-batch-audits
repo: roehub.com
branch: main
scope: "Run main-model and subagent read-only classic audits for every inventoried skill batch."
model_preferences:
  primary_agent_model: gpt-5.5
  reasoning_effort: xhigh
language:
  implementation: docs
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and subagent/cold-head rules"
    - path: docs/architecture/agents/skill-library-classic-audit-v1.md
      why: "plan_doc and audit rubric"
    - path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md
      why: "stage_ledger and Stage 00 gate"
    - path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md
      why: "inventory and batch plan"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  subagent_review_per_skill: true
  subagent_evidence_ref_per_skill: true
  coverage_reconciliation_required: true
  hash_drift_check_required: true
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
  - "Do not apply improvements."
  - "Do not edit source skills/plugins."
  - "Do not shrink coverage below the Stage 00 inventory."
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
    - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/01-subagent-batch-audits.md"
    - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md"
    - "docs/architecture/README.md if docs index changes"
  foreign_changes_policy: "ignore and preserve unrelated changes from other chats"
  mixed_file_policy: "stage only owned hunks; block that file if safe hunk separation is impossible"
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"
validation_strategy:
  depth: architecture_review
  e2e_required: true
  acceptance_surfaces:
    - "main-model review for every skill"
    - "subagent review for every skill"
    - "coverage reconciliation against Stage 00 inventory"
  evidence_target: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/01-subagent-batch-audits.md
proof_boundary:
  required_when: "Mac Studio, runtime, deploy or production smoke is not in scope"
  label: none
  changed_code_production_claim_allowed: false
  blocked_or_deferred_reason: "N/A"
stage_execution_ledger:
  path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md
  plan_doc: docs/architecture/agents/skill-library-classic-audit-v1.md
  current_stage: "01"
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
      - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/01-subagent-batch-audits.md"
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
  - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/01-subagent-batch-audits.md"
  - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/README.md"
safety_notes:
  - "Subagents are read-only reviewers."
  - "No source skill/plugin edits are allowed."
---

# Task

Run Stage `01`: audit every inventoried skill with both the main model and
clean-context subagents.

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

- Previous required stage ledger gate: confirm Stage `00` is accepted and `current_stage` allows Stage `01`.
- Re-check `sha256` for every skill before review and compare it with Stage `00` `inventory_sha256`.
- If any skill hash changed, mark that skill and Stage `01` as `blocked` until inventory is refreshed or the user explicitly approves reviewing the changed version.
- For every skill from Stage `00`, produce a main-model review.
- For every skill from Stage `00`, obtain at least one clean-context subagent review.
- Record a `subagent_evidence_ref` for every subagent review, pointing to the report section, subagent id, transcript reference, or other durable local evidence available in this environment.
- Maintain a per-skill coverage reconciliation table with `skill_id`, `batch_id`, `inventory_sha256`, `review_sha256`, `hash_drift_status`, `main_review_status`, `subagent_review_status`, `subagent_evidence_ref`, `clean_context_input_scope`, and `coverage_status`.
- If subagents are unavailable, mark Stage `01` as `blocked`; do not continue to Stage `02`.
- Every review must produce findings, strengths, risks and improvement proposals.
- Record disagreements between main model and subagent.
- Do not edit source skills/plugins.
- Update `stage_ledger` after validation and before final report.

## Subagent Review Schema

Each subagent result must include:

- `skill_id`
- `batch_id`
- `path`
- `inventory_sha256`
- `review_sha256`
- `hash_drift_status`: `same | changed | blocked`
- `clean_context_input_scope`
- `subagent_evidence_ref`
- `verdict`: `ok | improve | split | merge_or_deprecate | blocked`
- `what_works`
- `top_findings`
- `improvement_proposals`
- `priority`
- `risk_if_unchanged`

## Final Output

Respond in Russian with:

1. **Результат Stage 01**
2. **Subagent coverage**
3. **Findings by batch**
4. **Conflicts between main model and subagents**
5. **Quality gates**
6. **File manifest**
7. **Next-stage handoff**
