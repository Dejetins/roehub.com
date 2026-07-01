---
prompt_name: backtest_job_runner_v1_01_implement_admission_quota_contracts
repo: roehub.com
branch: main
scope: "R1: implement tier-based admission control and quota contracts for backtest jobs and lazy detail materialization."

language:
  implementation: python_fastapi_sqlalchemy
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "R1 source of truth"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "job create/preflight orchestration"
      inspect_symbols:
        - BacktestJobsUseCase
        - create
        - preflight
    - path: apps/api/routes/backtests.py
      why: "public API error/status mapping"
      inspect_symbols:
        - build_backtests_router
    - path: apps/api/dto/backtests.py
      why: "public DTO/error envelope additions"
      inspect_symbols:
        - BacktestJobResponse
    - path: src/trading/shared_kernel/primitives/paid_level.py
      why: "tier source primitive"
      inspect_symbols:
        - PaidLevel
  conditional_bundles:
    persistence_indexes:
      read_when: "quota reads require new indexes or repository methods"
      paths:
        - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py
        - alembic/versions/20260222_0003_backtest_jobs_v1.py
        - alembic/versions/20260511_0010_backtest_lazy_trades_materializations_v1.py
    lazy_detail_quota:
      read_when: "lazy detail materialization quota is implemented in this stage"
      paths:
        - src/trading/contexts/backtest/application/ports/lazy_trades_materializations.py
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/lazy_trades_materialization_repository.py
    tests:
      read_when: "adding focused tests"
      paths:
        - tests/unit/apps/api/test_backtests_routes.py
        - tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py
  consult_if_needed:
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      read_when: "Web UI refresh/rate-limit expectations are unclear"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "request cost or top_n semantics are unclear"

style_references: []

hard_requirements:
  config_driven_tier_policy: true
  idempotency_replay_does_not_consume_quota: true
  owner_scope_required: true
  no_api_compute_regression: true
  exact_error_codes_required: true

task_toggles:
  implement_backend_api: true
  implement_persisted_indexes_if_needed: true
  implement_runner_process: false
  implement_web_ui: false
  publish_after_success: true

skill_routing:
  - skill: architecture-design
    use_when: "introducing quota service/ports or persistence query boundaries"
    timing: "before implementation"
    reason: "service boundary and dependency direction"
  - skill: contract-impact-analysis
    use_when: "changing API errors, DTOs, schema, config or cache identity"
    timing: "before final report"
    reason: "public API and persisted contract"
  - skill: backend-quality-gates
    use_when: "running focused tests/lint/type checks"
    timing: "during verification"
    reason: "backend correctness"
  - skill: backend-performance-evidence
    use_when: "quota queries or indexes are added"
    timing: "during verification"
    reason: "admission path must stay cheap"
  - skill: publish-ci-deploy
    use_when: "all gates pass and publish_after_success is true"
    timing: "before ship"
    reason: "direct-main delivery"

target_envs:
  - local-dev
  - github-actions
  - mac-studio

required_literals:
  - "free"
  - "base"
  - "pro"
  - "ultra"
  - "429 backtest.rate_limited"
  - "422 backtest.request_too_expensive"
  - "503 backtest.queue_saturated"
  - "409"
  - "retry_after_seconds"

non_goals:
  - "Do not implement the runner loop here."
  - "Do not change `/backtests` UI."
  - "Do not add external queue/broker dependencies."
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
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest/application/use_cases"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/** quota/admission service"
  - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
  - "apps/api/routes/backtests.py"
  - "apps/api/dto/backtests.py"
  - "tests/unit/apps/api/test_backtests_routes.py"
  - "tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/ports/backtest_job_repositories.py"
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py"
  - "alembic/versions/*backtest*quota*"
  - "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md"

safety_notes:
  - "Admission control must reject before persistence unless idempotent replay is found."
  - "Request hash/cache identity must remain stable for equivalent requests."
---

# Task

Implement R1 admission control and quota contracts for backtest job creation and lazy detail materialization.

Done means:

- `free|base|pro|ultra` tier policy is enforced on job create/preflight paths;
- idempotent replay does not consume a new quota slot;
- expected `429/422/503/409` semantics are tested;
- no runner process or UI work is introduced.

## Context / Current State

`CurrentUserPrincipal.paid_level` / `identity_users.paid_level` is the source of tier. The runner plan defines v1 defaults:

- `free`: active full jobs 2, creates/hour 5, max `top_n` 20, max arity 2, max range 365d, active lazy details 2, lazy/hour 10, min autorefresh 60s;
- `base`: active full jobs 5, creates/hour 15, max `top_n` 50, max arity 3, max range 730d, active lazy details 5, lazy/hour 30, min autorefresh 30s;
- `pro`: active full jobs 20, creates/hour 60, max `top_n` 100, max arity 7, artifact coverage range, active lazy details 20, lazy/hour 120, min autorefresh 15s;
- `ultra`: active full jobs 50, creates/hour 240, max `top_n` 250, max arity 10, artifact coverage range, active lazy details 50, lazy/hour 500, min autorefresh 10s.

The implementation must be config-driven so commercial values can change without code changes.

## Requirements (Must)

- Add a small admission/tier policy service in the backtest application layer.
- Enforce active/queued/running full job limits and creates/hour on create path.
- Enforce request cost limits: `top_n`, arity and date range.
- Preserve idempotency: exact replay returns existing job and does not consume quota.
- Add lazy detail quota hooks if materialization create path exists; if not, expose a reusable service method for R3.
- Return typed errors:
  - `429 backtest.rate_limited` with `retry_after_seconds`, `limit_scope`, `paid_level`;
  - `422 backtest.request_too_expensive`;
  - `503 backtest.queue_saturated`;
  - `409` for same idempotency key with different request.
- Add indexes/repository methods only if quota reads cannot be cheap with existing schema.

## Requirements (Should)

- Keep policy defaults centralized and easy to override from environment/config.
- Add tests for all tiers and boundary values.
- Keep API create p95 cheap; do not add artifact compute to admission.

## Requirements (Nice-to-have)

- Include a tiny unit test proving unknown tier fails closed or normalizes safely.

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

Stop reading once admission data sources, touched files and error contracts are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-design`: use before implementation if adding service/port/query boundaries.
- `contract-impact-analysis`: use before final report for API, DTO, schema, config and cache identity.
- `backend-quality-gates`: use during verification.
- `backend-performance-evidence`: use if quota queries/indexes are added.
- `publish-ci-deploy`: use before ship only after all gates pass.

1. Inspect current create/idempotency/preflight flow.
2. Design minimal tier policy and quota reads.
3. Implement policy wiring and errors.
4. Add indexes/migration only if needed.
5. Add focused tests for tiers and idempotency.
6. Run gates.
7. Publish via direct-main flow if fully green.

# Acceptance criteria (Definition of Done)

- Tier limits are enforced and tested.
- Idempotency replay is compatible and quota-safe.
- Error payloads are typed and stable.
- No runner loop/UI code is added.
- Performance impact of admission reads is bounded or explicitly evidenced.

# Implementation constraints

- Business policy belongs in application/service layer, not route handlers.
- Routers map validation/errors; they do not own quota rules.
- Persistence changes must be additive and rollbackable.
- Do not change public request hashes unless explicitly justified.

# Files to indicate (expected touched areas)

Use `expected_primary_touches` and `possible_secondary_touches` from front matter.

# Non-goals

See front matter `non_goals`.

# Quality gates (must run and pass)

Run the `quality_gates` commands from front matter. If a gate fails, classify it as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.

# Final output: report format (strict)

Report in Russian using the `final_report_format.sections` order. Include exact commands, results, contract classification and any publish/deploy state.
