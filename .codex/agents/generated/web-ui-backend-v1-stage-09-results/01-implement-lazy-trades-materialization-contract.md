---
prompt_name: web_ui_backend_v1_stage09_01_lazy_trades_materialization_contract
repo: roehub.com
branch: main
scope: "Stage 09 backend foundation: materialization/status contract for lazy trades and result data, without expanding UI panels."

language:
  implementation: python_fastapi_sqlalchemy
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and backend gates"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Stage 09 contract"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "backtest artifacts/result contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "runner/materialization target state"
    - path: .codex/agents/generated/web-ui-backend-v1/09-implement-backtests-results.md
      why: "parent prompt"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "public lazy trades/result routes"
    - path: apps/api/dto/backtests.py
      why: "request/response DTOs"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "lazy trades and result orchestration"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py
      why: "worker-side execution and future materialization handoff"
    - path: src/trading/contexts/backtest/application/services/v2/result_series.py
      why: "read-model builders"
    - path: src/trading/contexts/backtest/infrastructure
      why: "repositories/adapters/migrations if present"
    - path: tests/unit/apps/api/test_backtests_routes.py
      why: "route regression tests"
    - path: tests/unit/contexts/backtest
      why: "use-case/service tests"

hard_requirements:
  no_sync_lazy_recompute_in_api_on_cache_miss: true
  public_variant_key_only: true
  typed_materialization_status_required: true
  owner_scope_required: true
  idempotent_materialization_request: true
  runner_boundary_compatible: true
  do_not_expand_browser_ui: true

task_toggles:
  implement_backend_api: true
  implement_persisted_schema_if_missing: true
  implement_runner_process: false
  implement_web_ui: false
  publish_after_success: true

package_contract:
  depends_on:
    - "00 inventory completed or current code inspected directly"
    - "backtest job runner plan read"
  owns:
    - "apps/api/routes/backtests.py"
    - "apps/api/dto/backtests.py"
    - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
    - "src/trading/contexts/backtest/application/services/v2/result_series.py"
    - "src/trading/contexts/backtest/** materialization ports/adapters/repositories as needed"
    - "migrations for backtest_lazy_trades_materializations if missing"
    - "tests/unit/apps/api/test_backtests_routes.py"
    - "tests/unit/contexts/backtest/** focused materialization tests"
  forbidden:
    - "apps/web/templates/**"
    - "apps/web/dist/**"
    - "dashboard/settings/strategies UI code"
    - "exchange integration not required for backtest artifacts"
  integration_points:
    - "BacktestJobWorkerUseCase can later consume pending materializations"
    - "Stage 09 result endpoint hardening"
    - "runner prompt pack"
  handoff:
    - "materialization status DTO names"
    - "migration revision and rollback notes"
    - "queue/status semantics for result endpoints"

skill_routing:
  - skill: prompt-manager
    use_when: "executing this prompt pack task"
    timing: "startup and final report"
    reason: "prompt-pack implementation discipline"
  - skill: architecture-design
    use_when: "adding materialization boundary, ports/adapters, dependency direction"
    timing: "before coding"
    reason: "new backend boundary/schema"
  - skill: contract-impact-analysis
    use_when: "adding DTO/status/schema and changing cache-miss behavior"
    timing: "before final report"
    reason: "public API and persisted schema change"
  - skill: backend-quality-gates
    use_when: "running focused Python tests/lint/type gates"
    timing: "verification"
    reason: "backend implementation"
  - skill: publish-ci-deploy
    use_when: "all local gates pass and publish_after_success is true"
    timing: "after verification"
    reason: "direct-main delivery"

target_envs:
  - local-dev
  - github-actions
  - mac-studio

required_literals:
  - "backtest_lazy_trades_materializations"
  - "queued"
  - "running"
  - "completed"
  - "failed"
  - "cancelled"
  - "variant_key"
  - "request_hash"
  - "public_variant_key"
  - "POST /api/backtests/jobs/{job_id}/variants/{variant_key}/trades"

non_goals:
  - "Do not implement the launchd runner process here unless it already exists and only needs wiring to this contract."
  - "Do not fake trades/equity/monthly/symbol data when storage is unavailable."
  - "Do not change current `/backtests` UI."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest"
    expect: "passes or focused known-pre-existing failures classified"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
---

# Task

Implement the backend materialization/status contract required before Stage 09 result UI can safely use detailed trades/statistics.

The main goal is to remove the production risk where API result routes synchronously recompute lazy trades on cache miss. Cache miss must produce a typed materialization state that a worker can process, not long-running compute inside the request path.

## Context / Current State

Backtest jobs can store summary/top variants, while detailed trades and derived series can be expensive and lazy. Current result methods may already exist, but the cache-miss path must be safe for production.

`backtest-job-runner-production-plan-v1.md` owns the full runner process. This prompt owns the API/use-case/storage contract that the runner will consume or that existing runner code can reuse.

## Requirements (Must)

- Inventory whether `backtest_lazy_trades_materializations` or equivalent storage already exists. Reuse it if present; do not duplicate.
- If missing, add a persisted materialization table/model/migration with owner/account scope, `job_id`, `variant_key`, request identity, status, timestamps, attempts/error fields, and indexes needed for lookup and worker pickup.
- Make `POST /api/backtests/jobs/{job_id}/variants/{variant_key}/trades` idempotent for the same owner/job/variant/request shape.
- Preserve public `variant_key`; do not expose storage SHA as browser contract.
- On cache hit, return completed trades detail as before.
- On cache miss, return typed pending/materialization status or enqueue status without synchronous full recompute in the API process.
- Define status model for `queued`, `running`, `completed`, `failed`, `cancelled` and retryability/correlation id where available.
- Enforce owner scope: user cannot start/read another user's materialization.
- Add tests for cache hit, cache miss enqueue/status, idempotency, ownership 403/404, invalid variant, request size bounds, and no lazy service execution in API request path on miss.

## Requirements (Should)

- Keep migration rollback straightforward.
- Include status freshness timestamps and stable polling hints if available.
- Preserve existing successful behavior for already materialized/cached detail.

# API Endpoint Specification Checklist

Before coding, write the local contract in tests or implementation notes:

- `method/path`: browser-visible `/api/backtests/jobs/{job_id}/variants/{variant_key}/trades`, backend router path without duplicate `/api`;
- `owner scope`: current user/account and job ownership;
- `request DTO`: required/optional fields, defaults, validation, idempotency key or derived request identity, size limits;
- `response DTO`: cache-hit payload and pending/status payload;
- `status codes`: `200/202/400/401/403/404/409/422/429/500/503`;
- `error payload`: Roehub-compatible error envelope;
- `pagination`: explicit `none` for POST detail unless changed;
- `cache identity`: request hash/materialization identity;
- `compatibility`: classify the cache-miss behavior change.

# Context Acquisition Protocol

Read `.codex/AGENTS.md`, runner plan materialization sections, Stage 09 parent prompt, then route/use-case/storage/test files. Stop after the materialization boundary and required tests are clear.

Reading budget: default `<= 14 files`, `<= ~80k tokens`.

# Work Plan

1. Inspect current lazy trades detail implementation and storage.
2. Decide whether this is reuse, migration addition, or narrow hardening.
3. Add/adjust DTOs and use-case contract.
4. Add/adjust persistence adapter/migration if missing.
5. Update route behavior for cache miss without API-process recompute.
6. Add focused tests.
7. Run gates.
8. If all gates pass, use `publish-ci-deploy` direct-main flow.

# Acceptance Criteria

- Cache miss path no longer runs full lazy trades recompute inside API process.
- Cache hit path remains successful.
- Materialization request/status is owner-scoped and idempotent.
- Migration/schema/index/rollback notes are present if schema changed.
- Tests prove ownership, idempotency, cache-hit, cache-miss and no-sync-compute behavior.

# publish-ci-deploy Direct-Main Delivery Contract

When all DoD, gates, and required evidence pass, run `publish-ci-deploy` in direct-main mode. Do not create a delivery branch, draft PR, or PR-based merge path.

Successful terminal state requires: up-to-date `main`, only intended files staged/committed, local gates green before push, direct push to `origin/main`, GitHub Actions/deploy monitored to green, Mac Studio checkout synchronized with `origin/main`, impacted services restarted/reloaded only as needed, and post-restart smoke verification completed.

Do not report successful publish/deploy while push, CI/deploy monitoring, Mac Studio pull, required restart/reload, or smoke verification remains pending.

# Final Output: Report Format

Report in Russian with these exact sections:

- `Intent`
- `Scope`
- `Design`
- `Contract impact`
- `Migrations`
- `Tests`
- `Performance`
- `Runtime evidence`
- `Risks`
- `Handoff`
- `Publish/deploy`
