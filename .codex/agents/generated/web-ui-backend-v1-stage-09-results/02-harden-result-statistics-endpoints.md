---
prompt_name: web_ui_backend_v1_stage09_02_harden_result_statistics_endpoints
repo: roehub.com
branch: main
scope: "Stage 09 backend result/statistics hardening: ensure all backtest result methods expected by Web UI are bounded, owner-scoped and materialization-aware."

language:
  implementation: python_fastapi
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and backend gates"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Stage 09 backend contract"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "result/statistics source contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "materialization/worker dependency"
    - path: .codex/agents/generated/web-ui-backend-v1/09-implement-backtests-results.md
      why: "parent Stage 09 prompt"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "result/statistics route declarations"
    - path: apps/api/dto/backtests.py
      why: "result/statistics DTOs"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "result/statistics use-case methods"
    - path: src/trading/contexts/backtest/application/services/v2/result_series.py
      why: "bounded series/stat/trades builders"
    - path: tests/unit/apps/api/test_backtests_routes.py
      why: "API result route tests"

hard_requirements:
  all_expected_statistics_methods_covered: true
  bounded_payloads_required: true
  owner_scope_required: true
  materialization_aware_cache_miss_required: true
  no_fake_unavailable_data: true
  public_variant_key_only: true
  pagination_required_for_trades: true

task_toggles:
  implement_backend_api: true
  implement_web_ui: false
  implement_migrations: false
  publish_after_success: true

package_contract:
  depends_on:
    - "01 materialization/status contract accepted or equivalent current code verified"
  owns:
    - "apps/api/routes/backtests.py"
    - "apps/api/dto/backtests.py"
    - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
    - "src/trading/contexts/backtest/application/services/v2/result_series.py"
    - "tests/unit/apps/api/test_backtests_routes.py"
  forbidden:
    - "apps/web/templates/**"
    - "apps/web/dist/**"
    - "new exchange API integrations"
    - "dashboard/settings/strategies pages"
  integration_points:
    - "Stage 09 UI can consume these endpoints"
    - "runner/materialization status is reflected consistently"
  handoff:
    - "stable endpoint matrix for prompt 03 UI integration"
    - "cache miss/status behavior for browser polling"

skill_routing:
  - skill: prompt-manager
    use_when: "following this generated prompt"
    timing: "startup and final report"
    reason: "prompt-pack discipline"
  - skill: contract-impact-analysis
    use_when: "changing route behavior/DTO/status codes"
    timing: "before final report"
    reason: "browser-visible API contract"
  - skill: backend-quality-gates
    use_when: "focused pytest/ruff/pyright verification"
    timing: "verification"
    reason: "backend route/use-case changes"
  - skill: backend-performance-evidence
    use_when: "validating bounded series/trades endpoints or load-sensitive paths"
    timing: "after functional tests"
    reason: "result endpoints can carry large payloads"
  - skill: publish-ci-deploy
    use_when: "all gates and required evidence pass"
    timing: "after verification"
    reason: "direct-main delivery"

target_envs:
  - local-dev
  - github-actions
  - mac-studio

required_literals:
  - "/api/backtests/jobs/{job_id}/summary"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/equity"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/drawdown"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats"
  - "GET /api/backtests/jobs/{job_id}/variants/{variant_key}/trades?page="
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv"
  - "variant_key"

non_goals:
  - "Do not redesign `/backtests` UI in this prompt."
  - "Do not invent values for missing trades/equity/stat data."
  - "Do not make exchange calls from result endpoints."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest"
    expect: "passes or focused known-pre-existing failures classified"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
---

# Task

Harden all backend result/statistics methods required for Stage 09 backtest results.

This prompt does not implement UI. It makes sure every method the current/future `/backtests` result UI needs has a stable, bounded, owner-scoped backend contract and does not accidentally trigger expensive synchronous work in the API process.

## Required Endpoint Matrix

Verify and harden these browser-visible endpoints:

- `GET /api/backtests/jobs/{job_id}/summary`
- `GET /api/backtests/jobs/{job_id}/variants/{variant_key}`
- `GET /api/backtests/jobs/{job_id}/variants/{variant_key}/equity`
- `GET /api/backtests/jobs/{job_id}/variants/{variant_key}/drawdown`
- `GET /api/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats`
- `GET /api/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats`
- `GET /api/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=...`
- `GET /api/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv`

Use backend router paths without a duplicated `/api` prefix.

## Requirements (Must)

- Preserve `summary` route and current UI summary behavior.
- Ensure every endpoint is owner-scoped.
- Ensure every endpoint uses public `variant_key`.
- Ensure chart/series endpoints have bounded points, explicit `max_points` behavior, stable ordering and no unbounded payload.
- Ensure paginated trades have stable ordering, limit bounds, page/cursor semantics documented in tests, and no full-table browser payload.
- Ensure CSV export is bounded or explicitly controlled and remains owner-scoped.
- Ensure cache miss/materialization pending states are typed and do not fake data.
- Add tests for every endpoint in the matrix, including missing data, cache/materialization pending, invalid job, invalid variant, ownership failure, and bounds.
- If an endpoint already exists and passes the contract, keep the diff minimal.

## Requirements (Should)

- Prefer shared read-model helpers in `result_series.py` over ad hoc route logic.
- Keep DTO names stable unless a compatibility fix requires new models.
- Include freshness/materialization metadata where useful for UI.

# Context Acquisition Protocol

Read `.codex/AGENTS.md`, Stage 09 plan section, materialization handoff from prompt 01 or current code, then the route/use-case/DTO/test files. Stop reading once each endpoint has an implementation and test target.

Reading budget: default `<= 12 files`, `<= ~70k tokens`.

# Work Plan

1. Build an endpoint-to-use-case-to-data-source table.
2. Identify missing tests or unsafe cache-miss paths.
3. Harden route/use-case/DTO behavior with minimal diffs.
4. Add focused tests for the endpoint matrix.
5. Run focused backend gates.
6. Run any small performance/bounds evidence needed for large response paths.
7. If all gates pass, use `publish-ci-deploy` direct-main flow.

# Acceptance Criteria

- Every endpoint in the matrix has explicit tests.
- No endpoint returns invented statistics when required data is unavailable.
- Trades endpoint is paginated and bounded.
- Chart/series endpoints are bounded and deterministic.
- Cache miss behavior is materialization-aware.
- Current `/backtests` UI summary/CSV behavior remains compatible.

# Performance Evidence

For changed hot paths, report at least one of:

- focused unit test proving max-points/limit bounds;
- payload size/row count assertion;
- local timing comparison for a representative in-memory result builder;
- explicit reason performance evidence is not applicable.

# publish-ci-deploy Direct-Main Delivery Contract

When all DoD, gates and evidence pass, run `publish-ci-deploy` in direct-main mode. Do not create a branch or PR. Publish directly to `main`, monitor CI/deploy, sync Mac Studio with `git pull --ff-only`, restart impacted services if needed, and smoke the deployed result endpoints.

Do not report successful publish/deploy while any of those steps remains pending.

# Final Output: Report Format

Report in Russian with these exact sections:

- `Intent`
- `Scope`
- `Endpoint matrix`
- `Design`
- `Contract impact`
- `Tests`
- `Performance`
- `Runtime evidence`
- `Risks`
- `Handoff`
- `Publish/deploy`
