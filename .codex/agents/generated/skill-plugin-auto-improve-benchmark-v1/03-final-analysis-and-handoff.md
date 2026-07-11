---
prompt_name: 03-final-analysis-and-handoff
repo: roehub.com
branch: main
scope: "Close the skill/plugin benchmark cycle with final score analysis, candidate disposition and optional scoped application handoff."
model_preferences:
  primary_agent_model: gpt-5.5
  reasoning_effort: xhigh
  clean_context_evaluator: "Codex subagents on gpt-5.5 xhigh for apply-ready evidence; fallback-only Stage 02 can only be report/defer/reject"
language:
  implementation: docs
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, scoped change policy and final reporting"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md
      why: "plan_doc and acceptance contract"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md
      why: "stage_ledger, Stage 02 gate and closure rules"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/02-ten-iteration-auto-improve-run.md
      why: "benchmark results and local run directory"
  task_entrypoints:
    - path: tools/codex_quality_benchmark
      why: "local report regeneration if needed"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  local_only_artifacts: true
  explicit_candidate_disposition: true
task_toggles:
  allow_branch_creation: false
  allow_worktree: false
  allow_stash: false
  allow_source_skill_edits_without_user_approval: false
  allow_external_llm_api_from_python: false
skill_routing:
  - skill: staged-plan-runner
    use_when: "executing this prompt as part of the existing plan_doc/prompt_pack_dir/stage_ledger"
    timing: "before stage actions"
    reason: "owns current-stage gating and goal_driven closure rules"
  - skill: pre-ship-gate
    use_when: "the user explicitly asks to stage, commit, publish or hand off changes after applying approved skill edits"
    timing: "before ship only"
    reason: "owns release-readiness checks"
target_envs:
  - local checkout
  - local Codex home if user explicitly approves applying global skill edits
required_literals:
  - "Before solving or editing, emit one short commentary update stating the exact reasoning/root-cause check you are doing."
  - "Do thought experiments before making changes."
  - "Use them to test competing explanations, edge cases, failure modes, and whether the apparent fix would actually solve the root cause."
  - "Do not stop at the first plausible explanation."
  - "Do not stop after a superficial answer. Validate the result with evidence, edge cases, and tests before finalizing."
  - "all scripts and saved benchmark artifacts stay local to the current machine"
non_goals:
  - "Do not apply global skill/plugin edits without explicit user approval."
  - "Do not create branches, worktrees or stashes."
  - "Do not publish or deploy Roehub runtime changes."
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
    - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/03-final-analysis-and-handoff.md"
    - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md"
    - "docs/architecture/README.md if docs index changes"
    - "approved source skill/plugin files only if user explicitly approves applying candidates"
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
  - cmd: "uv run python -m tools.codex_quality_benchmark.cli summarize --run-dir <run_dir>"
    expect: "regenerates summary from saved local results"
  - cmd: "uv run python -m tools.docs.generate_docs_index"
    expect: "updates docs/architecture/README.md if needed"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"
validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "local benchmark summary regeneration"
    - "candidate disposition table"
    - "stage_ledger closure"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/03-final-analysis-and-handoff.md
proof_boundary:
  required_when: "Mac Studio, deploy, target-host or production smoke is not in scope"
  label: none
  changed_code_production_claim_allowed: false
  blocked_or_deferred_reason: "N/A"
stage_execution_ledger:
  path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md
  plan_doc: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md
  current_stage: "03"
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
    code:
      - "approved winning candidate source skill/plugin files from Stage 02, only when the user explicitly approves applying them"
    config_infra_migrations: []
    docs_runbooks:
      - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/03-final-analysis-and-handoff.md"
      - "docs/architecture/README.md"
    prompt_artifacts: []
    ledger_and_evidence:
      - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md"
  final_report_required_fields:
    - created
    - modified
    - deleted
    - outside_expected_paths
    - outside_expected_paths_justification
    - foreign_changes_excluded
    - mixed_files
expected_primary_touches:
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/03-final-analysis-and-handoff.md"
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/README.md"
safety_notes:
  - "Global skill/plugin edits are outside Roehub git unless the user explicitly asks to apply them."
  - "Summaries may cite local run paths but must not embed raw secrets or large evaluator transcripts."
---

# Task

Run Stage `03`: close the benchmark cycle with final score analysis, candidate
disposition and next-action handoff.

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

## Requirements (Must)

- Verify from `stage_ledger` that Stage `02` is accepted, or accepted_for_learning for report-only closure, and current stage allows `03`; otherwise write Stage `03` as blocked and stop.
- Previous required stage ledger gate: before final analysis, confirm Stage `02` is accepted, or accepted_for_learning for report-only closure, in the ledger and `current_stage` allows Stage `03`; if not, update Stage `03` as blocked and stop.
- Regenerate or verify the final benchmark summary from saved local artifacts, not from memory.
- Produce a candidate disposition table: `apply_now`, `defer`, `reject`, `needs_user_approval`.
- `apply_now` is allowed only when Stage `02` is fully `accepted` with clean-context subagent or explicitly equivalent isolated evaluator evidence. If Stage `02` is only `accepted_for_learning`, every candidate disposition must be `defer`, `reject`, or `needs_user_approval`.
- Do not apply global skill/plugin edits unless the user explicitly approved applying them in this run.
- If edits are approved, apply only exact winning candidate files/hunks and record contract impact per file.
- Keep raw `.codex/tmp` data out of durable docs except summarized paths and hashes.
- Update `stage_ledger` to `completed` only when final report and candidate disposition are complete.

## Acceptance Criteria

- Final Stage `03` report has champion table, score deltas, rejected approaches and residual risks.
- Every candidate has a disposition and reason.
- Source skill/plugin files are unchanged unless explicit user approval is recorded.
- Ledger closure state is accurate: `completed` if done, `blocked` if missing evidence or approval.
- Final report states what can be run next.

# Final Output

Respond in Russian with:

1. **Результат Stage 03**
2. **Final score table**
3. **Candidate disposition**
4. **Applied or deferred files**
5. **Quality gates**
6. **File manifest**
7. **Closure / next action**
