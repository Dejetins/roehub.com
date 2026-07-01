---
prompt_name: backtest_job_runner_v1_02_implement_full_job_runner_process
repo: roehub.com
branch: main
scope: "R2: implement standalone `backtest-job-runner` process for full queued backtest jobs."

language:
  implementation: python_worker_runtime
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "R2 source of truth"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py
      why: "worker use-case to execute"
      inspect_symbols:
        - BacktestJobWorkerUseCase
        - BacktestJobExecutor
    - path: src/trading/contexts/backtest/application/services/v2/job_orchestration.py
      why: "runtime executor service"
      inspect_symbols:
        - BacktestRuntimeJobOrchestrationService
        - BacktestJobExecutionResult
    - path: apps/worker/market_data_ws/main/main.py
      why: "existing worker CLI/signal pattern"
      inspect_symbols:
        - main
        - _install_signal_handlers
    - path: apps/api/wiring/modules/backtest.py
      why: "current backtest service wiring to mirror"
      inspect_symbols:
        - _build_jobs_use_case
  conditional_bundles:
    postgres_wiring:
      read_when: "building worker repository wiring"
      paths:
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_lease_repository.py
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/gateway.py
    runtime_services:
      read_when: "orchestration dependencies need explicit construction"
      paths:
        - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
        - src/trading/contexts/backtest/application/services/v2/combo_planning.py
        - src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
        - src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
        - src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py
    tests:
      read_when: "adding runner process tests"
      paths:
        - tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py
        - tests/unit/apps/worker
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "artifact runtime wiring or benchmark acceptance is ambiguous"
    - path: scripts/backtest/run_stage_8_5_create_path_load_smoke.py
      read_when: "need enqueue-only load-smoke pattern"

style_references: []

hard_requirements:
  one_process_concurrency_one: true
  no_full_compute_in_api: true
  heartbeat_and_lease_required: true
  graceful_shutdown_required: true
  max_jobs_per_process_required: true
  metrics_endpoint_required: true
  no_launchd_changes_here: true

task_toggles:
  implement_worker_process: true
  implement_full_job_execution: true
  implement_lazy_detail_execution: false
  implement_launchd: false
  publish_after_success: true

skill_routing:
  - skill: architecture-design
    use_when: "wiring worker composition root and runtime boundaries"
    timing: "before implementation"
    reason: "ports/adapters and process boundary"
  - skill: contract-impact-analysis
    use_when: "changing runtime workflow, config or worker DTO/status behavior"
    timing: "before final report"
    reason: "runtime workflow contract"
  - skill: backend-quality-gates
    use_when: "running worker/use-case tests and lint/type gates"
    timing: "during verification"
    reason: "backend runtime correctness"
  - skill: backend-performance-evidence
    use_when: "claiming worker loop latency, memory or process recycle behavior"
    timing: "during verification"
    reason: "worker is performance/memory-sensitive"
  - skill: publish-ci-deploy
    use_when: "all gates pass and publish_after_success is true"
    timing: "before ship"
    reason: "direct-main delivery"

target_envs:
  - local-dev
  - github-actions
  - mac-studio

required_literals:
  - "apps/worker/backtest_job_runner"
  - "BacktestJobWorkerUseCase"
  - "BacktestRuntimeJobOrchestrationService"
  - "ROEHUB_BACKTEST_RUNNER_CONCURRENCY=1"
  - "ROEHUB_BACKTEST_RUNNER_MAX_JOBS_PER_PROCESS"
  - "backtest_runner_tasks_claimed_total"
  - "backtest_runner_last_success_unixtime"

non_goals:
  - "Do not add launchd plist or Prometheus target in this prompt."
  - "Do not implement lazy detail materialization execution here."
  - "Do not modify `/backtests` UI."
  - "Do not add Celery/RQ/Kafka/Redis Streams."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Scope"
    - "Design"
    - "Contract impact"
    - "Tests"
    - "Performance"
    - "Runtime evidence"
    - "Risks"
    - "Handoff"
    - "Publish/deploy"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py tests/unit/apps/worker"
    expect: "passes; if tests/unit/apps/worker does not exist yet, add focused tests or classify"
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/worker src/trading/contexts/backtest tests/unit/contexts/backtest tests/unit/apps/worker"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"

expected_primary_touches:
  - "apps/worker/backtest_job_runner/main/main.py"
  - "apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py"
  - "src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py"
  - "tests/unit/apps/worker/backtest_job_runner/**"
  - "tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"

possible_secondary_touches:
  - "apps/worker/backtest_job_runner/__init__.py"
  - "apps/worker/backtest_job_runner/wiring/__init__.py"
  - "apps/worker/backtest_job_runner/wiring/modules/__init__.py"
  - "src/trading/contexts/backtest/application/services/v2/__init__.py"
  - "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md"

safety_notes:
  - "Worker loop must be locally runnable but not installed as production service in this prompt."
  - "A successful local test is not production acceptance; Mac Studio smoke belongs to R5."
---

# Task

Implement the standalone full-job `backtest-job-runner` process.

Done means:

- `python -m apps.worker.backtest_job_runner.main.main` can run a worker loop;
- the worker claims at most one full job at a time;
- it executes through `BacktestJobWorkerUseCase` and `BacktestRuntimeJobOrchestrationService`;
- it heartbeats, exits gracefully, and supports max-jobs recycle;
- it exposes low-cardinality Prometheus metrics locally.

## Context / Current State

The API already persists queued jobs and uses `DatabaseBacktestJobExecutionTrigger` as an explicit no-op trigger. `BacktestJobWorkerUseCase` currently has a minimal `run_next()` path over claim/progress/executor/finish.

This prompt creates the missing process/composition root. It does not install launchd and does not process production backlog.

## Requirements (Must)

- Add `apps/worker/backtest_job_runner` package with CLI entrypoint, logging, signal handling and runtime loop.
- Wire Postgres job repository, lease repository, preflight service and `BacktestRuntimeJobOrchestrationService`.
- Enforce v1 effective compute concurrency `1`.
- Implement poll interval, empty backoff, lease seconds, heartbeat interval, max runtime and max jobs per process from env/defaults.
- Ensure heartbeat is updated during long execution or at safe stage boundaries. If current executor cannot heartbeat inside tight kernels, document and test the best available boundary.
- Keep terminal writes guarded by lease owner semantics.
- Add structured logs without secrets/full payloads.
- Add metrics with no high-cardinality labels.
- Add tests for no-job idle, claim/success, executor failure, graceful stop, max-jobs recycle and lost lease/failed finish handling where feasible.

## Requirements (Should)

- Provide a `--run-once` or equivalent local smoke mode for tests/operators.
- Return non-zero on unrecoverable wiring/config errors.
- Keep process wiring similar to existing `market_data_ws` and `strategy_live_runner` workers.

## Requirements (Nice-to-have)

- Include a local fake-executor test mode only if it does not risk production confusion.

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
- `<= ~50k tokens`

Stop reading once worker composition, env config, tests and acceptance gates are bounded.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-design`: use before implementation for composition root and process boundary.
- `contract-impact-analysis`: use before final report for runtime/config workflow impact.
- `backend-quality-gates`: use during verification.
- `backend-performance-evidence`: use if claiming loop latency, memory or recycle behavior.
- `publish-ci-deploy`: use before ship only after all gates pass.

1. Inspect existing worker process patterns.
2. Extend worker use-case only where required for loop/heartbeat correctness.
3. Add backtest runner CLI and wiring.
4. Add metrics and structured logging.
5. Add focused tests.
6. Run gates.
7. Publish via direct-main flow if fully green.

# Acceptance criteria (Definition of Done)

- Local worker process can start in a controlled no-job mode.
- Full job execution path is wired through existing application services.
- API create remains enqueue-only.
- Worker does not run more than one job concurrently.
- Metrics endpoint starts on configured port in local tests/smoke.
- Focused tests pass.

# Implementation constraints

- Domain/application logic must not depend on launchd or HTTP.
- Worker composition root owns adapters and environment parsing.
- Do not introduce broker dependencies.
- Do not alter public API routes except for a bug uncovered by focused tests.

# Files to indicate (expected touched areas)

Use `expected_primary_touches` and `possible_secondary_touches` from front matter.

# Non-goals

See front matter `non_goals`.

# Quality gates (must run and pass)

Run the `quality_gates` commands from front matter. If a gate fails, classify it as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.

# Final output: report format (strict)

Report in Russian using the `final_report_format.sections` order. Include exact commands, local run evidence and direct-main publish/deploy state if executed.
