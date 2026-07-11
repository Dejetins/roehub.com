---
prompt_name: 00-baseline-inventory-and-rubric
repo: roehub.com
branch: main
scope: "Freeze the local skill/plugin benchmark inventory, rubric, source methodology snapshot and clean-context eval cases."
model_preferences:
  primary_agent_model: gpt-5.5
  reasoning_effort: xhigh
  clean_context_evaluator: "Codex subagents on gpt-5.5 xhigh for clean-context gates; any fallback must be recorded explicitly"
language:
  implementation: docs
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, prompt-pack execution policy, local-only and branch/worktree rules"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md
      why: "plan_doc and target benchmark contract"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md
      why: "stage_ledger, current stage and next-stage gate"
  task_entrypoints:
    - path: /Users/daniildegtyarev/.codex/skills
      why: "local installed skills inventory; read only selected SKILL.md files after manifest sampling"
    - path: /Users/daniildegtyarev/.codex/plugins/cache
      why: "local plugin-contributed skills inventory; read only selected SKILL.md files after manifest sampling"
  external_sources:
    - url: https://github.com/crimeacs/auto-improve
      why: "source methodology reference"
    - url: https://raw.githubusercontent.com/crimeacs/auto-improve/main/README.md
      why: "auto-improve loop and usage defaults"
    - url: https://raw.githubusercontent.com/crimeacs/auto-improve/main/criteria/README.md
      why: "rubric construction rules"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  no_source_skill_edits: true
  local_only_artifacts: true
  clean_context_subagent_plan: true
task_toggles:
  allow_branch_creation: false
  allow_worktree: false
  allow_stash: false
  allow_external_llm_api_from_python: false
  allow_source_skill_edits: false
skill_routing:
  - skill: staged-plan-runner
    use_when: "executing this prompt as part of the existing plan_doc/prompt_pack_dir/stage_ledger"
    timing: "before stage actions"
    reason: "owns current-stage gating and goal_driven continuation rules"
  - skill: architecture-review
    use_when: "reviewing inventory/rubric completeness and fact-vs-inference discipline"
    timing: "during validation"
    reason: "owns evidence discipline for audit reports"
target_envs:
  - local checkout
  - local Codex home
required_literals:
  - "Before solving or editing, emit one short commentary update stating the exact reasoning/root-cause check you are doing."
  - "Do thought experiments before making changes."
  - "Use them to test competing explanations, edge cases, failure modes, and whether the apparent fix would actually solve the root cause."
  - "Do not stop at the first plausible explanation."
  - "Do not stop after a superficial answer. Validate the result with evidence, edge cases, and tests before finalizing."
  - "all scripts and saved benchmark artifacts stay local to the current machine"
non_goals:
  - "Do not implement the Python benchmark harness in Stage 00."
  - "Do not edit source skills/plugins in Stage 00."
  - "Do not create branches, worktrees, stashes, temporary repo checkouts, or auxiliary workflow folders."
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
    - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/00-baseline-inventory-and-rubric.md"
    - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md"
    - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md"
    - "docs/architecture/README.md if docs index changes"
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
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index"
    expect: "updates docs/architecture/README.md if needed"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes for owned Markdown changes"
validation_strategy:
  depth: architecture_review
  e2e_required: true
  acceptance_surfaces:
    - "local target manifest completeness"
    - "fixed 100-point rubric"
    - "clean-context eval case design"
    - "stage_ledger handoff"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/00-baseline-inventory-and-rubric.md
proof_boundary:
  required_when: "Mac Studio, deploy, target-host or production smoke is not in scope"
  label: none
  changed_code_production_claim_allowed: false
  blocked_or_deferred_reason: "N/A"
stage_execution_ledger:
  path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md
  plan_doc: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md
  current_stage: "00"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
prompt_pack_execution:
  mode: goal_driven
  plan_doc: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md
  prompt_pack_dir: .codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/
  stage_ledger: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md
  goal_mode_optional: true
  goal_artifact_required: false
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    code: []
    config_infra_migrations: []
    docs_runbooks:
      - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/00-baseline-inventory-and-rubric.md"
      - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md"
      - "docs/architecture/README.md"
    prompt_artifacts: []
    ledger_and_evidence:
      - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/00-baseline-inventory-and-rubric.md"
  final_report_required_fields:
    - created
    - modified
    - deleted
    - outside_expected_paths
    - outside_expected_paths_justification
    - foreign_changes_excluded
    - mixed_files
expected_primary_touches:
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/00-baseline-inventory-and-rubric.md"
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Keep raw skill/plugin content and subagent prompts free of secrets."
  - "Record absolute local paths only when needed for reproducibility; do not expose credentials or env dumps."
---

# Task

Run Stage `00` for `Skill/Plugin Auto-Improve Benchmark v1`: freeze the local
target inventory, methodology snapshot, rubric and clean-context eval cases.

Before doing anything else, emit this short commentary update requirement in the
active Codex thread and follow it:

```text
Before solving or editing, emit one short commentary update stating the exact reasoning/root-cause check you are doing.

Do thought experiments before making changes.
Use them to test competing explanations, edge cases, failure modes, and whether the apparent fix would actually solve the root cause.
Do not stop at the first plausible explanation.

Do not stop after a superficial answer. Validate the result with evidence, edge cases, and tests before finalizing.
```

All scripts and saved benchmark artifacts stay local to the current machine. If
Python is run, it runs in this local checkout only. Subagents may be used only as
clean-context evaluators for sanitized prompt/answer packets; store their
verdicts back into local artifacts.

Done means:

- Stage `00` report exists and records the source methodology snapshot.
- A bounded target manifest is frozen with skill/plugin type classification.
- The 100-point rubric and fixed eval cases are frozen before Stage `01`.
- The stage ledger marks Stage `00` accepted or blocked with evidence.

## Requirements (Must)

- Read `stage_ledger` first and confirm `current_stage: 00`; if not, update the Stage `00` report as blocked and stop.
- Previous required stage ledger gate: Stage `00` has no previous stage; verify the ledger says `current_stage: 00` and block if the plan_doc, prompt_pack_dir or stage_ledger links are missing or inconsistent.
- Do not edit source skill/plugin files.
- Inspect local skill/plugin inventory by listing files first, then reading only enough selected `SKILL.md` files to classify targets.
- Include at least one target from each available high-value type when feasible: `workflow_skill`, `research_skill`, `coding_skill`, `review_skill`, `artifact_skill`, `plugin_tool_skill`.
- Cap the first benchmark batch to a reviewable size, recommended 6 to 12 target files.
- Read `crimeacs/auto-improve` README and criteria guide from the source URLs. Record observed URLs, access date, and commit SHA if it can be obtained without creating a local clone/folder.
- Freeze the exact scoring rubric from `plan_doc`; if adjusted, keep dimensions summing to 100 and explain why.
- Define fixed eval cases per target type. Each case must specify input prompt, expected behavior, failure conditions, and scoring notes.
- Define subagent clean-context packet shape: what sanitized target content/context the evaluator receives and what JSON verdict it must return.
- Update `stage_ledger` after validation and before final report.
- Keep local-only and redaction constraints explicit.

## Work Plan

1. Verify the stage gate from `stage_ledger`.
2. List local skills/plugins and choose a bounded target set.
3. Read only selected target files needed to classify skill types and baseline risks.
4. Read the external auto-improve methodology URLs and summarize only the transferable method.
5. Draft the Stage `00` report with manifest, rubric, eval cases, clean-context packet schema and blockers.
6. Update `stage_ledger`: Stage `00` accepted only if Stage `01` can implement the harness without hidden assumptions.
7. Run docs/index and diff checks.

## Acceptance Criteria

- Report includes target table with path, type, baseline hash if computed, reason for inclusion and whether source edits are allowed later.
- Report includes rubric table with 100 total points.
- Report includes fixed eval cases for every selected target type.
- Report includes auto-improve methodology transfer notes: separate evaluator, best-of-N, pairwise champion gate, keep/discard and iteration log.
- Stage `01` is allowed only if report and ledger provide enough details to implement `tools/codex_quality_benchmark/` locally.

# Final Output

Respond in Russian with:

1. **Результат Stage 00**
2. **Target manifest**
3. **Rubric and eval cases**
4. **Local-only and subagent contract**
5. **Quality gates**
6. **File manifest**
7. **Next-stage handoff**
