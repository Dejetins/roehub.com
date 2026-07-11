---
prompt_name: 02-consolidated-improvement-backlog
repo: roehub.com
branch: main
scope: "Consolidate classic audit findings into one improvement backlog covering every skill."
model_preferences:
  primary_agent_model: gpt-5.5
  reasoning_effort: xhigh
language:
  implementation: docs
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and final reporting"
    - path: docs/architecture/agents/skill-library-classic-audit-v1.md
      why: "plan_doc and per-skill schema"
    - path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md
      why: "stage_ledger and Stage 01 gate"
    - path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md
      why: "full skill inventory"
    - path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/01-subagent-batch-audits.md
      why: "main-model and subagent findings"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  improvement_proposal_per_skill: true
  what_works_per_skill: true
  coverage_reconciliation_required: true
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
  - "Do not apply improvements in this stage."
  - "Do not edit source skills/plugins."
  - "Do not hide skills with no severe findings; every skill needs a row."
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
    - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md"
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
    - "coverage reconciliation against Stage 00 inventory"
    - "improvement proposal for every skill"
    - "ledger closure"
  evidence_target: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md
proof_boundary:
  required_when: "Mac Studio, runtime, deploy or production smoke is not in scope"
  label: none
  changed_code_production_claim_allowed: false
  blocked_or_deferred_reason: "N/A"
stage_execution_ledger:
  path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md
  plan_doc: docs/architecture/agents/skill-library-classic-audit-v1.md
  current_stage: "02"
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
      - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md"
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
  - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md"
  - "docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/README.md"
safety_notes:
  - "No source skill/plugin edits are allowed."
  - "Every discovered skill must appear in the final backlog."
---

# Task

Run Stage `02`: consolidate all main-model and subagent findings into one
improvement backlog covering every discovered skill.

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

- Previous required stage ledger gate: confirm Stage `01` is accepted and `current_stage` allows Stage `02`.
- Reconcile Stage `01` output against Stage `00` inventory; every skill must have a final row.
- Verify every final row includes `what_works`, `subagent_evidence_ref`, `hash_drift_status`, `coverage_status`, and at least one improvement proposal.
- Propose improvements for every skill. If no change is recommended, write a low-priority no-op/polish recommendation.
- Prioritize improvements as `P0`, `P1`, `P2` or `P3`.
- Do not apply improvements or edit source skills/plugins.
- Update `stage_ledger` to `completed` only if coverage is complete and evidence is recorded.

## Final Output

Respond in Russian with:

1. **Результат Stage 02**
2. **Полнота покрытия**
3. **Top improvement themes**
4. **Per-skill backlog**
5. **Quality gates**
6. **File manifest**
7. **Closure / next action**
