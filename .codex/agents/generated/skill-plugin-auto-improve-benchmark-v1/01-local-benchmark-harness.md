---
prompt_name: 01-local-benchmark-harness
repo: roehub.com
branch: main
scope: "Implement the local Python benchmark harness for skill/plugin version scoring and pairwise keep/discard decisions."
model_preferences:
  primary_agent_model: gpt-5.5
  reasoning_effort: xhigh
  clean_context_evaluator: "Codex subagents on gpt-5.5 xhigh for clean-context gates; any fallback must be recorded explicitly"
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, Python gates and scoped change policy"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md
      why: "plan_doc and harness architecture"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md
      why: "stage_ledger, Stage 00 gate and Stage 01 status"
    - path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/00-baseline-inventory-and-rubric.md
      why: "frozen target manifest, rubric and eval cases"
  task_entrypoints:
    - path: tools
      why: "existing repo tooling style"
    - path: pyproject.toml
      why: "Python version and test/lint dependencies"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  local_only_artifacts: true
  no_external_llm_api_by_default: true
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
  - skill: backend-quality-gates
    use_when: "Python harness or tests are changed"
    timing: "verification"
    reason: "owns focused Python lint/test gate selection"
target_envs:
  - local checkout
required_literals:
  - "Before solving or editing, emit one short commentary update stating the exact reasoning/root-cause check you are doing."
  - "Do thought experiments before making changes."
  - "Use them to test competing explanations, edge cases, failure modes, and whether the apparent fix would actually solve the root cause."
  - "Do not stop at the first plausible explanation."
  - "Do not stop after a superficial answer. Validate the result with evidence, edge cases, and tests before finalizing."
  - "all scripts and saved benchmark artifacts stay local to the current machine"
non_goals:
  - "Do not run the 10 improvement iterations in Stage 01."
  - "Do not edit source skills/plugins in Stage 01."
  - "Do not add an external LLM provider dependency."
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
    - "tools/codex_quality_benchmark/**"
    - "tests/unit/tools/test_codex_quality_benchmark*.py"
    - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/01-local-benchmark-harness.md"
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
  - cmd: "uv run ruff check tools/codex_quality_benchmark tests/unit/tools/test_codex_quality_benchmark*.py"
    expect: "passes when files exist"
  - cmd: "uv run pytest -q tests/unit/tools/test_codex_quality_benchmark*.py"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index"
    expect: "updates docs/architecture/README.md if needed"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces:
    - "local CLI fixture run"
    - "score aggregation from saved JSON"
    - "pairwise keep/discard deterministic fixture"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/01-local-benchmark-harness.md
proof_boundary:
  required_when: "Mac Studio, deploy, target-host or production smoke is not in scope"
  label: none
  changed_code_production_claim_allowed: false
  blocked_or_deferred_reason: "N/A"
stage_execution_ledger:
  path: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md
  plan_doc: docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md
  current_stage: "01"
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
      - "tools/codex_quality_benchmark/**"
      - "tests/unit/tools/test_codex_quality_benchmark*.py"
    config_infra_migrations: []
    docs_runbooks:
      - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/01-local-benchmark-harness.md"
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
  - "tools/codex_quality_benchmark/__init__.py"
  - "tools/codex_quality_benchmark/cli.py"
  - "tools/codex_quality_benchmark/models.py"
  - "tools/codex_quality_benchmark/manifest.py"
  - "tools/codex_quality_benchmark/scoring.py"
  - "tools/codex_quality_benchmark/pairwise.py"
  - "tools/codex_quality_benchmark/reports.py"
  - "tests/unit/tools/test_codex_quality_benchmark.py"
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/01-local-benchmark-harness.md"
  - "docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/README.md"
safety_notes:
  - "The harness may parse local JSON/Markdown/TSV only; do not call external LLM APIs by default."
  - "Raw run state belongs under .codex/tmp/skill-plugin-auto-improve-benchmark-v1/ and must not be committed."
---

# Task

Run Stage `01`: implement a local deterministic Python harness for benchmark
manifests, version snapshots, score aggregation, pairwise decisions and report
generation.

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

- Verify from `stage_ledger` that Stage `00` is accepted and current stage allows `01`; otherwise write Stage `01` as blocked and stop.
- Previous required stage ledger gate: before implementation, confirm Stage `00` is accepted in the ledger and `current_stage` allows Stage `01`; if not, update Stage `01` as blocked and stop.
- Implement a stdlib-first local harness under `tools/codex_quality_benchmark/`; use existing project dependencies only if justified.
- Accept saved evaluator JSON as input; do not require live model/API calls.
- Provide deterministic score aggregation from dimension scores.
- Provide pairwise champion decision logic with strict `2-0` candidate keep rule.
- Provide report generation for `results.tsv`, `events.jsonl` and `summary.md`.
- Include tests with tiny local fixtures.
- Do not edit source skill/plugin files.
- Update stage report and `stage_ledger` after validation.

## Minimum CLI Shape

The exact command names may vary, but the harness must support these operations:

```bash
uv run python -m tools.codex_quality_benchmark.cli validate-manifest --manifest <path>
uv run python -m tools.codex_quality_benchmark.cli aggregate --run-dir <path>
uv run python -m tools.codex_quality_benchmark.cli summarize --run-dir <path>
```

The CLI must fail closed on:

- missing target/version hash;
- dimension scores not summing to expected rubric dimensions;
- candidate keep without `2-0` pairwise win;
- secret/locality violation marked severe;
- incomplete evaluator JSON.

## Acceptance Criteria

- Focused tests prove score aggregation, pairwise keep/discard, TSV writing and blocked malformed inputs.
- A sample fixture run produces local `results.tsv`, `events.jsonl` and `summary.md`.
- Stage `02` has enough command-level instructions to run 10 iterations.
- Stage ledger records validation evidence and next-stage handoff.

# Final Output

Respond in Russian with:

1. **Результат Stage 01**
2. **Harness API and CLI**
3. **Local fixture evidence**
4. **Quality gates**
5. **File manifest**
6. **Next-stage handoff**
