---
prompt_name: backtest_job_runner_v1_00_readiness_backlog_and_contract_freeze
repo: roehub.com
branch: main
scope: "R0: readiness inventory for `backtest-job-runner-v1` before runtime implementation."

language:
  implementation: python_docs_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "runner source of truth"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py
      why: "current worker use-case seam"
      inspect_symbols:
        - BacktestJobWorkerUseCase
        - BacktestJobExecutor
    - path: src/trading/contexts/backtest/application/ports/lazy_trades_materializations.py
      why: "current lazy materialization port"
      inspect_symbols:
        - BacktestLazyTradesMaterializationRepository
        - BacktestLazyTradesMaterializationTask
    - path: scripts/macos/reload_launchd_services.sh
      why: "current service reload behavior"
      inspect_symbols:
        - prod_services
        - collect_worker_services
    - path: tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py
      why: "current worker behavior coverage"
      inspect_symbols:
        - test_worker_claims_updates_progress_executes_and_finishes_job
  conditional_bundles:
    storage_schema:
      read_when: "materialization schema or queue indexes need confirmation"
      paths:
        - alembic/versions/20260511_0010_backtest_lazy_trades_materializations_v1.py
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/lazy_trades_materialization_repository.py
    production_state:
      read_when: "production backlog or Mac Studio service state is being inspected"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
    api_stage85:
      read_when: "enqueue-only create boundary is ambiguous"
      paths:
        - apps/api/routes/backtests.py
        - src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
  consult_if_needed:
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      read_when: "Stage 09 result/materialization dependency is unclear"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "artifact/runtime contract is unclear"

style_references: []

hard_requirements:
  no_runtime_behavior_change: true
  current_code_inventory_required: true
  backlog_policy_required: true
  separate_web_ui_from_runner: true
  no_publish_by_default: true

task_toggles:
  implementation_changes_allowed: false
  docs_patch_allowed_if_drift_found: true
  inspect_prod_backlog_if_credentials_available: true
  publish_after_success: false

skill_routing:
  - skill: architecture-review
    use_when: "checking plan/code/runtime drift without implementation"
    timing: "during investigation"
    reason: "readiness review"
  - skill: contract-impact-analysis
    use_when: "classifying queue/API/schema/runtime contract gaps"
    timing: "before final report"
    reason: "runner crosses public API and persistence boundaries"
  - skill: backend-quality-gates
    use_when: "running focused readiness gates"
    timing: "during verification"
    reason: "confirm existing worker/API baseline"

target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "backtest-job-runner-v1"
  - "BacktestJobWorkerUseCase"
  - "DatabaseBacktestJobExecutionTrigger"
  - "backtest_lazy_trades_materializations"
  - "ROEHUB_BACKTEST_RUNNER_CONCURRENCY=1"
  - "127.0.0.1:9204/metrics"

non_goals:
  - "Do not implement the worker process in this prompt."
  - "Do not change `/backtests` UI."
  - "Do not process existing queued jobs."
  - "Do not alter launchd services."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Current contract"
    - "Backlog state"
    - "Gaps"
    - "Contract impact"
    - "Tests"
    - "Docs"
    - "Risks"
    - "Handoff"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"
    expect: "passes or exact pre-existing failure classification"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest/application/use_cases"
    expect: "passes or exact pre-existing failure classification"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md only if drift is found"
  - ".codex/agents/generated/backtest-job-runner-v1/*.md only if prompt drift is found"

possible_secondary_touches:
  - "docs/runbooks/mac-studio-native-backend-operations.md if runtime docs drift is found"
  - "docs/runbooks/mac-studio-monitoring-plan.md if monitoring docs drift is found"

safety_notes:
  - "Existing queued production jobs must be inspected before enabling a runner."
  - "A controlled smoke job, not an old stuck job, is the primary production acceptance target."
---

# Task

Prepare the implementation surface for `backtest-job-runner-v1` without changing runtime behavior.

Done means:

- current code/docs/runtime gaps are inventoried;
- existing queued backlog handling is classified;
- next prompts can implement R1-R5 without rediscovering the baseline;
- no runtime service is started, stopped, or modified.

## Context / Current State

The production plan says Stage 8.5 already made `POST /api/backtests/jobs` enqueue-only. `BacktestJobWorkerUseCase` exists as the application seam, while the standalone production process `apps/worker/backtest_job_runner` does not exist yet.

The current materialization table/port may already exist. Treat existing code as current fact and the plan as target state. If they disagree, record the drift and patch only docs/prompt text if needed.

## Requirements (Must)

- Inspect current worker, queue trigger, materialization port/schema, launchd reload behavior and focused tests.
- Verify that API create path remains enqueue-only and no full compute runs in API request path.
- Inspect whether `backtest_lazy_trades_materializations` storage is already implemented.
- Inspect production backlog only if access is available and safe. Do not let runner process old jobs in this prompt.
- Classify blockers for R1-R5 as `blocker`, `must-fix`, `non-blocker`, or `already satisfied`.
- If docs say a missing component is already implemented, correct docs narrowly.

## Requirements (Should)

- Record exact files/symbols that later prompts should touch.
- Record any known queued job ids only as evidence, not as acceptance targets.

## Requirements (Nice-to-have)

- Include a one-line suggested execution order for R1-R5 if any dependency changed.

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
- `<= ~45k tokens`

Stop reading once the runner baseline, backlog policy, touched files and acceptance gates are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Read conditional bundles only when the stated condition applies.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-review`: use during investigation for docs/code/runtime drift.
- `contract-impact-analysis`: use before final report if public API, schema, config or runtime workflow impact is found.
- `backend-quality-gates`: use during verification for focused tests/lint.

1. Inspect current queue and worker seams.
2. Inspect materialization schema/port only if needed.
3. Inspect launchd reload behavior.
4. Optionally inspect safe production backlog state.
5. Run focused gates.
6. Patch docs only for concrete drift.
7. Report R1-R5 handoff.

# Acceptance criteria (Definition of Done)

- Current runner baseline is described with exact files/symbols.
- Backlog policy is explicit.
- R1-R5 dependencies are clear.
- No runtime behavior changed.
- Focused gates were run or failures were classified.

# Implementation constraints

- Do not edit application code.
- Do not start or install launchd services.
- Do not run any backtest job.
- Preserve unrelated dirty worktree changes.

# Files to indicate (expected touched areas)

Use `expected_primary_touches` and `possible_secondary_touches` from front matter.

# Non-goals

See front matter `non_goals`.

# Quality gates (must run and pass)

Run the `quality_gates` commands from front matter. If a gate fails, classify it as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.

# Final output: report format (strict)

Report in Russian using the `final_report_format.sections` order. Distinguish observed facts, inference and assumptions.
