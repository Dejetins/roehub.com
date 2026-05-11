---
prompt_name: backtest_job_runner_v1_03_implement_lazy_detail_materialization_worker
repo: roehub.com
branch: main
scope: "R3: implement lazy trades materialization queue claim/execution and integrate it into `backtest-job-runner`."

language:
  implementation: python_fastapi_worker
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "R3 source of truth"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ports/lazy_trades_materializations.py
      why: "materialization port to extend"
      inspect_symbols:
        - BacktestLazyTradesMaterializationRepository
    - path: src/trading/contexts/backtest/adapters/outbound/persistence/postgres/lazy_trades_materialization_repository.py
      why: "Postgres materialization adapter"
      inspect_symbols:
        - PostgresBacktestLazyTradesMaterializationRepository
    - path: src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py
      why: "detail recompute/cache service"
      inspect_symbols:
        - BacktestLazyTradesDetailService
    - path: apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py
      why: "runner loop integration point"
      inspect_symbols:
        - build_backtest_job_runner_app
  conditional_bundles:
    api_cache_miss:
      read_when: "POST /trades cache-miss still computes synchronously"
      paths:
        - apps/api/routes/backtests.py
        - apps/api/dto/backtests.py
        - src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
    schema:
      read_when: "queue claim/finish columns or indexes are insufficient"
      paths:
        - alembic/versions/20260511_0010_backtest_lazy_trades_materializations_v1.py
        - tests/unit/apps/migrations/test_bootstrap_apply_flow.py
    tests:
      read_when: "adding materialization worker tests"
      paths:
        - tests/unit/apps/api/test_backtests_routes.py
        - tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py
        - tests/unit/apps/worker/backtest_job_runner
  consult_if_needed:
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      read_when: "Stage 09 UI materialization status contract is unclear"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "lazy trades cache identity is unclear"

style_references: []

hard_requirements:
  cache_hit_200_cache_miss_202: true
  no_sync_lazy_recompute_in_api_on_cache_miss: true
  claim_finish_detail_tasks_required: true
  anti_starvation_required: true
  owner_scope_required: true
  cache_write_atomicity_required: true
  no_full_trades_in_top_rows: true

task_toggles:
  implement_lazy_detail_execution: true
  implement_api_cache_miss_status_if_missing: true
  implement_worker_process_changes: true
  implement_web_ui: false
  publish_after_success: true

skill_routing:
  - skill: architecture-design
    use_when: "extending materialization port/use-case and runner scheduling"
    timing: "before implementation"
    reason: "queue boundary and dependency direction"
  - skill: contract-impact-analysis
    use_when: "changing POST /trades status, DTOs, schema or cache identity"
    timing: "before final report"
    reason: "public API and persistence contract"
  - skill: backend-quality-gates
    use_when: "running focused API/worker/service tests"
    timing: "during verification"
    reason: "backend correctness"
  - skill: backend-performance-evidence
    use_when: "validating cache miss does not block API or detail tasks stay bounded"
    timing: "during verification"
    reason: "interactive detail path"
  - skill: publish-ci-deploy
    use_when: "all gates pass and publish_after_success is true"
    timing: "before ship"
    reason: "direct-main delivery"

target_envs:
  - local-dev
  - github-actions
  - mac-studio

required_literals:
  - "backtest_lazy_trades_materializations"
  - "lazy_detail"
  - "full_job"
  - "202"
  - "retry_after_seconds"
  - "backtest_lazy_trades_cache_total"
  - "ROEHUB_BACKTEST_DETAIL_SYNC_FALLBACK_ENABLED=false"

non_goals:
  - "Do not add `/backtests` UI result panels in this prompt."
  - "Do not split into `backtest-detail-runner` process in v1."
  - "Do not store full trades in `backtest_job_top_variants`."
  - "Do not make exchange API calls."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Scope"
    - "Design"
    - "Contract impact"
    - "Migrations"
    - "Tests"
    - "Performance"
    - "Runtime evidence"
    - "Risks"
    - "Handoff"
    - "Publish/deploy"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py tests/unit/apps/worker/backtest_job_runner"
    expect: "passes; if worker test dir does not exist yet, add focused tests or classify"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/worker src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/worker tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/ports/lazy_trades_materializations.py"
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/lazy_trades_materialization_repository.py"
  - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
  - "apps/api/routes/backtests.py"
  - "apps/api/dto/backtests.py"
  - "apps/worker/backtest_job_runner/**"
  - "tests/unit/apps/api/test_backtests_routes.py"
  - "tests/unit/apps/worker/backtest_job_runner/**"

possible_secondary_touches:
  - "alembic/versions/20260511_0010_backtest_lazy_trades_materializations_v1.py"
  - "src/trading/contexts/backtest/application/services/v2/result_series.py"
  - "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md"

safety_notes:
  - "Cache miss must enqueue/status, not block API."
  - "Detail task priority cannot starve full jobs."
---

# Task

Implement lazy trades materialization execution for `backtest-job-runner`.

Done means:

- cache hit returns current detail payload;
- cache miss returns a typed `202` materialization status without API-process recompute;
- runner can claim and execute `lazy_detail` tasks;
- runner writes cache atomically and marks materialization terminal;
- scheduler alternates `lazy_detail` and `full_job` according to anti-starvation policy.

## Context / Current State

The repository already contains a planned materialization table/port in the current checkout. Verify current state first and reuse existing pieces. Do not create a duplicate queue.

The runner plan says one process handles `full_job` and `lazy_detail` in v1. Cache-miss lazy detail gets interactive priority but cannot starve full jobs: after at most 5 `lazy_detail` tasks while full jobs are queued, take one `full_job`.

## Requirements (Must)

- Extend materialization repository/port with claim, heartbeat/progress if needed, terminal success/failure/cancel operations.
- Ensure claim uses safe locking and lease semantics comparable to full jobs.
- Integrate detail tasks into the runner loop from prompt 02.
- Ensure API `POST /trades` cache miss returns typed status and `retry_after_seconds`.
- Ensure `GET /trades`/series/stat endpoints can surface status/degraded state and do not trigger heavy recompute.
- Preserve cache key identity and public `variant_key`.
- Add tests for cache hit, cache miss 202, idempotent task replay, claim/success/failure, second cached read, ownership and anti-starvation.

## Requirements (Should)

- Keep lazy materialization bounded to one variant.
- Include TTL/cache_status fields in terminal task state.
- Include metrics for cache hit/miss/materialization result without high-cardinality labels.

## Requirements (Nice-to-have)

- Add a local run-once mode that processes one detail task for smoke tests.

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

Stop reading once materialization queue operations, API status behavior and runner integration points are bounded.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-design`: use before implementation for queue/runner scheduling boundary.
- `contract-impact-analysis`: use before final report for API/DTO/schema/cache impacts.
- `backend-quality-gates`: use during verification.
- `backend-performance-evidence`: use when validating non-blocking cache miss or detail latency.
- `publish-ci-deploy`: use before ship only after all gates pass.

1. Inspect existing materialization schema/port/API behavior.
2. Add missing claim/finish/replay operations.
3. Wire detail task execution into runner loop.
4. Update API cache-miss/status behavior if missing.
5. Add tests.
6. Run gates.
7. Publish via direct-main flow if fully green.

# Acceptance criteria (Definition of Done)

- API cache miss does not compute lazy trades synchronously.
- Runner processes one lazy detail task to terminal state in tests.
- Anti-starvation rule is tested or explicitly evidenced.
- Cache hit remains compatible.
- No full trades are stored in top variant rows.

# Implementation constraints

- Keep application ports free of Postgres-specific details.
- Keep API status DTO explicit; do not overload error payloads for normal pending state.
- Do not add browser UI work.
- Do not add external queue/broker.

# Files to indicate (expected touched areas)

Use `expected_primary_touches` and `possible_secondary_touches` from front matter.

# Non-goals

See front matter `non_goals`.

# Quality gates (must run and pass)

Run the `quality_gates` commands from front matter. If a gate fails, classify it as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.

# Final output: report format (strict)

Report in Russian using the `final_report_format.sections` order. Include exact API status behavior, tests, performance evidence and publish/deploy state.
