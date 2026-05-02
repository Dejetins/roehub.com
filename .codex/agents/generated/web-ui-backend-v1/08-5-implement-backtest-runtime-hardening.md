---
prompt_name: web_ui_backend_v1_08_5_backtest_runtime_hardening
repo: roehub.com
branch: current
scope: "Этап 8.5: убрать sync_inline compute из API request path перед публичным backtest UI rollout."

language:
  implementation: python_fastapi_worker_postgres
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "DDD, performance, contracts, gates, Mac Studio policy"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 8.5 source of truth"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "canonical backtest jobs, runtime and benchmark contract"
  task_entrypoints:
    - path: apps/api/wiring/modules/backtest.py
      why: "current sync_inline executor wiring"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "create, idempotency, claim, executor path"
      inspect_symbols:
        - BacktestJobsUseCase
        - BacktestJobExecutor
    - path: src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      why: "job repository queue/claim/lease contract"
    - path: apps/api/routes/backtests.py
      why: "public job create/cancel routes"
    - path: tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py
      why: "use-case behavior tests"
  conditional_bundles:
    worker_runtime:
      read_when: "when implementing worker trigger/adapter or queue claim loop"
      paths:
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_lease_repository.py
        - apps/worker
        - scripts/macos
    benchmark_policy:
      read_when: "if compute path, timers, or verified hot path changes"
      paths:
        - docs/architecture/backtest/benchmark_iterations/README.md
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_summary.md
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/benchmark_summary.md
  consult_if_needed:
    - path: docs/runbooks/mac-studio-native-backend-operations.md
      read_when: "deployment/runtime verification or Mac Studio service impact is ambiguous"

hard_requirements:
  api_create_bounded: true
  no_long_compute_in_api_request_path: true
  idempotency_replay_no_duplicate_enqueue: true
  cancel_idempotent: true
  request_hash_cache_identity_unchanged: true
  macstudio_policy_if_compute_touched: true
  load_smoke_required: true

task_toggles:
  implement_worker_trigger_or_adapter: true
  implement_job_state_tests: true
  implement_public_api_breaking_change: false
  publish_after_success: true

skill_routing:
  - skill: architecture-design
    use_when: "worker trigger/queue/adapter boundary is not already clear from existing code"
    timing: "before implementation"
    reason: "runtime workflow boundary must be designed before edits"
  - skill: contract-impact-analysis
    use_when: "changing job create response/status, states, idempotency, request hash/cache identity, queue metadata, persisted schema"
    timing: "before implementation and final report"
    reason: "backtest jobs are public API and persistence contracts"
  - skill: backend-performance-evidence
    use_when: "validating API create latency, CPU saturation, benchmark impact, or Mac Studio evidence"
    timing: "during verification"
    reason: "this stage is performance/runtme hardening"
  - skill: backend-quality-gates
    use_when: "running use-case/API/repository/worker tests, ruff, pyright"
    timing: "during verification"
    reason: "backend correctness gates"
  - skill: root-cause-debugging
    use_when: "idempotency, job states, worker claim, or performance smoke fails"
    timing: "only after a concrete failure"
    reason: "must isolate root cause before changing compute path"
  - skill: publish-ci-deploy
    use_when: "all local gates, load smoke, and any required Mac Studio benchmark evidence pass"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - github-actions
  - macstudio

required_literals:
  - "sync_inline"
  - "queued"
  - "running"
  - "succeeded"
  - "failed"
  - "cancelled"
  - "Idempotency-Key"
  - "request_hash"

non_goals:
  - "Do not implement browser results UI; Stage 9 owns it."
  - "Do not change canonical request hash."
  - "Do not store full trades in top rows."
  - "Do not claim benchmark acceptance from local tests alone if compute path changes."

final_report_format:
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright/performance evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py tests/unit/contexts/backtest/application/ports tests/unit/contexts/backtest/adapters/outbound/persistence/postgres"
    expect: "passes; adjust to existing focused tests if directory names differ"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/api/wiring/modules/backtest.py"
  - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
  - "src/trading/contexts/backtest/application/ports/backtest_job_repositories.py"
  - "src/trading/contexts/backtest/**"
  - "apps/api/routes/backtests.py"
  - "tests/unit/apps/api/test_backtests_routes.py"
  - "tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py"

possible_secondary_touches:
  - "apps/worker/**"
  - "alembic/versions/*.py"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "scripts/backtest/*"

safety_notes:
  - "If full worker queue is not ready, document transitional adapter, timeout guard, and public rollout limitation."
  - "Do not silently change external response shape/status."
  - "Mac Studio evidence is required if verified compute path changes."
---

# Task

Implement Stage 8.5 backtest runtime hardening.

Done means:

- `POST /api/backtests/jobs` returns after validation/persistence/enqueue, not after full compute;
- idempotency replay does not enqueue duplicate work;
- cancel remains deterministic for queued/running/terminal jobs;
- UI can show `queued/running/succeeded|failed|cancelled`;
- request hash/cache identity are unchanged;
- load smoke shows API process is not CPU-saturated by create path;
- Mac Studio benchmark policy is followed if compute path is touched.

## Context / Current State

- Current wiring builds `BacktestRuntimeJobOrchestrationService` in API process.
- `BacktestJobsUseCase.create()` can execute via `sync_inline`.
- Public UI must treat create as async job flow.

## Requirements (Must)

- Remove long-running compute from API request path or document a transitional adapter with explicit rollout ban.
- Preserve public jobs API compatibility where possible.
- Add tests for queued create, idempotency replay, cancel, worker claim/update.
- Run performance/load evidence.
- Use `publish-ci-deploy` only after all required evidence passes.

## Requirements (Should)

- Keep worker trigger behind a port/adapter.
- Keep API response fast and bounded.
- Keep state transitions deterministic.

## Requirements (Nice-to-have)

- Add progress event/polling bridge only if it does not broaden the stage.

# Context acquisition protocol

Read `.codex/AGENTS.md`, Stage 8.5, backtest runtime doc, then task entrypoints. Expand to worker/runtime only after bounding the queue/adapter design.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Classify current sync_inline path and target queue/adapter boundary.
2. Design minimal compatible runtime transition.
3. Implement use case/wiring/worker changes.
4. Add focused tests.
5. Run load smoke and benchmark evidence if compute path changed.
6. Run quality gates.
7. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- API create path is bounded by validation/persistence/enqueue.
- Current job states do not regress.
- No full result/trades payload is stored in top rows.
- Focused local tests pass.
- Capacity/load evidence records create path behavior.
- Any compute-path change has Mac Studio benchmark evidence or a documented blocker.

# Implementation constraints

## API / contracts

- Public API contract: `compatible-change` if shape/status stays compatible.
- Runtime workflow: `compatible-change` or documented `unknown`.
- Request hash/cache identity: `none`.
- Persisted schema: `none` or additive `compatible-change`.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- UI results page.
- AI configurator.
- New scoring algorithms.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py
uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Add the strongest feasible load/benchmark command and record exact command/output path in the final report.

# Final output: report format (strict)

Report in Russian: `Intent`, `Scope`, `Design`, `Contract impact`, `Tests`, `Performance evidence`, `Runtime evidence`, `Risks`, `Handoff`, `Publish/deploy`.
