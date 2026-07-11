---
prompt_name: 02-ten-iteration-auto-improve-run
repo: roehub.com
branch: main
scope: "Run exactly 10 auto-improve-style iteration attempts across selected skill/plugin targets with clean-context subagent evaluation and local score logs."
model_preferences:
  primary_agent_model: gpt-5.5
  reasoning_effort: xhigh
  clean_context_evaluator: "Codex subagents on gpt-5.5 xhigh for full acceptance; fallback can only be blocked or accepted_for_learning"
language:
  implementation: docs
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, scoped change policy and redaction rules"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md
      why: "plan_doc, metrics and iteration approaches"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md
      why: "stage_ledger, Stage 01 gate and Stage 02 status"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/00-baseline-inventory-and-rubric.md
      why: "target manifest, rubric and eval cases"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/01-local-benchmark-harness.md
      why: "harness commands and validation evidence"
  task_entrypoints:
    - path: tools/codex_quality_benchmark
      why: "local scoring harness"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  local_only_artifacts: true
  clean_context_subagents_required: true
  fixed_metrics_each_iteration: true
task_toggles:
  allow_branch_creation: false
  allow_worktree: false
  allow_stash: false
  allow_source_skill_edits: false
  allow_external_llm_api_from_python: false
skill_routing:
  - skill: staged-plan-runner
    use_when: "executing this prompt as part of the existing plan_doc/prompt_pack_dir/stage_ledger"
    timing: "before stage actions"
    reason: "owns current-stage gating and goal_driven continuation rules"
  - skill: prompt-manager
    use_when: "candidate changes alter prompt/skill instructions and need executable prompt quality checks"
    timing: "during candidate review only"
    reason: "owns prompt artifact readiness and skill-routing instruction quality"
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
  - "Do not directly overwrite installed skill/plugin files in Stage 02."
  - "Do not trust self-evaluation from the mutator context."
  - "Do not commit raw .codex/tmp benchmark state."
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
    - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/02-ten-iteration-auto-improve-run.md"
    - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/benchmark-summary-*.md"
    - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md"
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
  - cmd: "uv run python -m tools.codex_quality_benchmark.cli validate-manifest --manifest <run_dir>/manifest.json"
    expect: "passes"
  - cmd: "uv run python -m tools.codex_quality_benchmark.cli aggregate --run-dir <run_dir>"
    expect: "recomputes results.tsv from saved evaluator JSON"
  - cmd: "uv run python -m tools.codex_quality_benchmark.cli summarize --run-dir <run_dir>"
    expect: "writes summary.md"
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
    - "clean-context subagent evaluations"
    - "local score aggregation"
    - "pairwise champion gate"
    - "reproducible TSV/JSONL/Markdown reports"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/02-ten-iteration-auto-improve-run.md
proof_boundary:
  required_when: "Mac Studio, deploy, target-host or production smoke is not in scope"
  label: none
  changed_code_production_claim_allowed: false
  blocked_or_deferred_reason: "N/A"
stage_execution_ledger:
  path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md
  plan_doc: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md
  current_stage: "02"
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
      - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/02-ten-iteration-auto-improve-run.md"
      - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/benchmark-summary-*.md"
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
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/02-ten-iteration-auto-improve-run.md"
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/benchmark-summary-*.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Subagent packets must be sanitized and bounded; no secrets or env dumps."
  - "Candidate versions live in local run state until Stage 03 applies approved edits."
---

# Task

Run Stage `02`: execute exactly 10 auto-improve-style iteration attempts over
the selected local skill/plugin target batch.

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

- Verify from `stage_ledger` that Stage `01` is accepted and current stage allows `02`; otherwise write Stage `02` as blocked and stop.
- Previous required stage ledger gate: before running benchmark iterations, confirm Stage `01` is accepted in the ledger and `current_stage` allows Stage `02`; if not, update Stage `02` as blocked and stop.
- Use the fixed target manifest, rubric and eval cases from Stage `00`.
- Use the local harness from Stage `01` for scoring and reports.
- Create a local run directory under `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/<run_id>/`.
- Snapshot every baseline as `v00` with `sha256`.
- Run exactly 10 iteration approaches in the exact order from `plan_doc`.
- If an approach cannot safely produce a candidate, still write that iteration row as `no_op` or `blocked` with the current champion hash, score, fixed metrics and decision reason.
- For each target and iteration, produce candidate text as a separate local version file. Do not overwrite the original skill/plugin file.
- Use clean-context subagents for full accepted evaluations. Each subagent must receive only sanitized target/candidate/eval-case context and must return structured JSON.
- If clean-context subagents or an explicitly equivalent isolated evaluator are unavailable, mark Stage `02` as `blocked` or `accepted_for_learning`; do not mark it `accepted`, do not produce apply-ready candidates, and require Stage `03` to summarize/defer/reject only.
- Score every version with the same rubric and cases.
- Run pairwise candidate-vs-champion checks in both orderings; keep candidate only on strict `2-0`.
- Record every keep/discard/block in local `events.jsonl` and `results.tsv`.
- Write a durable Stage `02` report summarizing the run and linking the local run directory path. Do not paste large raw evaluator transcripts.
- Update `stage_ledger` after validation and before final report.

## Thought Experiments Required Before Candidate Edits

For each iteration approach, briefly test:

- Could this edit improve routing but harm clean-context execution?
- Could the skill become over-specific and miss valid tasks?
- Could safety text hide the actual workflow?
- Could examples become stale or dominate the instruction?
- Could the same metric be inflated without a real quality gain?

If the apparent fix does not survive these thought experiments, do not produce
that candidate; record a blocked/no-op iteration.

## Acceptance Criteria

- Each selected target has baseline row `v00` plus iteration rows `1..10`.
- Each version row has score, hash, dimension scores, eval pass count and decision.
- Accepted candidates pass the strict pairwise gate.
- Rejected candidates have a reason.
- The run can be summarized again from saved local JSON/TSV without subagent memory.
- Stage `03` can decide whether to apply, defer or reject winning candidates.

# Final Output

Respond in Russian with:

1. **Результат Stage 02**
2. **Run directory and score summary**
3. **Iteration decisions**
4. **Clean-context evaluator evidence**
5. **Quality gates**
6. **File manifest**
7. **Next-stage handoff**
