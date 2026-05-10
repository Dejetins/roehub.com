---
prompt_name: web_ui_backend_v1_08_5_backtest_runtime_hardening
repo: roehub.com
branch: main
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
      inspect_symbols:
        - BacktestJobRepository
        - BacktestJobLeaseRepository
    - path: src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_lease_repository.py
      why: "existing Postgres claim_next/SKIP LOCKED lease implementation for worker boundary"
    - path: apps/api/routes/backtests.py
      why: "public job create/cancel routes"
    - path: src/trading/contexts/backtest/application/dto/backtest_jobs.py
      why: "public job/status DTO surface and refresh metadata target"
  conditional_bundles:
    tests:
      read_when: "when adding job runtime/use-case/API tests"
      paths:
        - tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py
        - tests/unit/apps/api/test_backtests_routes.py
    worker_runtime:
      read_when: "when implementing worker trigger/adapter or queue claim loop"
      paths:
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

style_references:
  design_manifest:
    path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
    purpose: "визуальный source of truth для токенов, тем, layouts, density и accessibility"
  external_reference_root:
    path: /Users/daniildegtyarev/Projects/roehub_web_ui
    purpose: "reference screenshots/assets; inspect only stage-relevant pages"
  default_palette: terminal-orange
  theme_variants:
    - terminal-orange
    - graphite
    - matrix-green
    - high-contrast
  invariant_financial_colors: true
  default_locale: en
  secondary_locale: ru
  language_switch_required: true

hard_requirements:
  api_create_bounded: true
  no_long_compute_in_api_request_path: true
  idempotency_replay_no_duplicate_enqueue: true
  cancel_idempotent: true
  request_hash_cache_identity_unchanged: true
  macstudio_policy_if_compute_touched: true
  load_smoke_required: true
  ui_refresh_retry_window_required: true
  current_code_has_no_worker_use_case: true
  must_introduce_worker_boundary_from_existing_lease_port: true

task_toggles:
  implement_worker_use_case_and_trigger_adapter: true
  implement_job_state_tests: true
  implement_public_api_breaking_change: false
  publish_after_success: true

package_contract:
  depends_on:
    - "public backtest jobs API contract stable"
  owns:
    - "apps/api/wiring/modules/backtest.py"
    - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
    - "src/trading/contexts/backtest/application/ports/backtest_job_repositories.py"
    - "src/trading/contexts/backtest/application/dto/backtest_jobs.py"
    - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_lease_repository.py"
    - "src/trading/contexts/backtest/** worker trigger/queue/adapter additions"
    - "apps/worker/** backtest worker additions"
    - "tests/unit/apps/api/test_backtests_routes.py job runtime assertions"
    - "tests/unit/contexts/backtest/** job runtime tests"
  forbidden:
    - "browser results/configurator UI"
    - "AI configurator"
    - "scoring algorithm changes"
    - "canonical request hash changes"
  integration_points:
    - "job state transitions"
    - "worker trigger port/adapter"
    - "BacktestJobLeaseRepository.claim_next worker claim path"
    - "idempotent enqueue"
    - "bounded UI polling/refresh metadata for backtest job state"
    - "Mac Studio benchmark evidence if compute path changes"
  handoff:
    - "bounded async job create path and worker/queue contract for Stage 9/12"

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
  - "background_auto"
  - "claim_next"
  - "BacktestJobWorkerUseCase"
  - "execution_trigger"
  - "refresh_status"
  - "retry_after_seconds"

non_goals:
  - "Do not implement browser result endpoints/state; Stage 9 owns selected result state inside `/backtests`."
  - "Do not change canonical request hash."
  - "Do not store full trades in top rows."
  - "Do not let browser refresh/autorefresh trigger compute or bypass bounded job-state polling contracts."
  - "Do not claim benchmark acceptance from local tests alone if compute path changes."

final_report_format:
  - "Intent: что реализовано и почему это нужно пользователю"
  - "Scope: bounded capability, routes, modules, files, owns/forbidden compliance"
  - "Design: use cases, DTO, ports/adapters, migrations, JS modules, template fragments"
  - "Contract impact: public API, port, DTO, persisted schema, config, cache/request identity, browser-visible behavior, performance risk"
  - "Tests: exact commands, cwd, results, focused/lint/type/migration gates"
  - "Docs: updated docs or explicit reason no docs changed"
  - "Performance: touched hot paths, payload/latency/RSS/load checks, or explicit none"
  - "Runtime evidence: Playwright/browser, tests, inference, assumptions clearly separated"
  - "Risks: edge cases, migration/rollback, pre-existing/environmental/flaky failures"
  - "Handoff: stable exports, route includes, helpers, endpoint contracts for next agents"
  - "Publish/deploy: direct-main publish-ci-deploy terminal state; if successful, include direct push to origin/main, main CI/deploy monitoring, local main sync, Mac Studio git pull, impacted service restart/reload, and smoke verification evidence; otherwise exact blocker or reason it was skipped"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"
    expect: "passes after adding this worker-boundary test file; if the file is named differently, run the exact added worker-boundary test file and report the substitution"
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
  - "src/trading/contexts/backtest/application/dto/backtest_jobs.py"
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_lease_repository.py"
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
  - "Backtest job status/progress refresh must be bounded, cache/read-model friendly, and expose retry-window semantics for UI autorefresh."
  - "Mac Studio evidence is required if verified compute path changes."
---

# Task

Implement Stage 8.5 backtest runtime hardening.

Done means:

- `POST /api/backtests/jobs` returns after validation/persistence/enqueue, not after full compute;
- idempotency replay does not enqueue duplicate work;
- cancel remains deterministic for queued/running/terminal jobs;
- UI can show `queued/running/succeeded|failed|cancelled`;
- UI refresh/autorefresh can poll job state without overlapping compute, unbounded payloads or missing retry-window semantics;
- request hash/cache identity are unchanged;
- load smoke shows API process is not CPU-saturated by create path;
- Mac Studio benchmark policy is followed if compute path is touched.

## Context / Current State

- Current wiring builds `BacktestRuntimeJobOrchestrationService` in API process.
- `BacktestJobsUseCase.create()` can execute via `sync_inline`.
- Current source does **not** provide `BacktestJobWorkerUseCase`, `execution_trigger`, or an enqueue adapter yet; this stage must introduce the boundary instead of assuming it exists.
- Current source already provides `BacktestJobLeaseRepository.claim_next` and `PostgresBacktestJobLeaseRepository` with FIFO/SKIP LOCKED semantics; prefer this existing lease seam over introducing a broker.
- Historical `sync_inline` literals may remain for legacy persisted rows and migration compatibility, but the public API create path must stop using them for new jobs.
- Public UI must treat create as async job flow.

## Requirements (Must)

- Remove long-running compute from API request path or document a transitional adapter with explicit rollout ban.
- Change new job creation from `execution_mode="sync_inline"` to a background execution mode, normally `background_auto`, unless current persisted-contract evidence requires a different compatible literal.
- Remove `claim_for_inline_execution()` and `executor.execute()` from `BacktestJobsUseCase.create()`; creation must persist a queued job and trigger/signal worker execution only.
- Introduce a minimal worker use case around the existing lease port: `claim_next -> update progress/heartbeat -> executor.execute -> finish_with_top_variants`, with deterministic failure persistence.
- Preserve public jobs API compatibility where possible.
- Add tests for queued create, idempotency replay, cancel, worker claim/update.
- Ensure job status/progress polling is bounded and can return `refresh_status`, `generated_at`, `next_allowed_refresh_at`/`retry_after_seconds` or an equivalent typed retry-window contract for Stage 8/9 UI refresh.
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
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Classify current sync_inline path and target queue/adapter boundary.
2. Confirm with `rg` whether `BacktestJobWorkerUseCase`, `execution_trigger`, and enqueue adapter exist in the current checkout; if absent, implement them as new target-state code.
3. Design minimal compatible runtime transition using existing `BacktestJobLeaseRepository.claim_next`.
4. Implement use case/wiring/worker changes.
5. Add focused tests.
6. Run load smoke and benchmark evidence if executor/scoring/orchestration internals changed.
7. Run quality gates.
8. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- API create path is bounded by validation/persistence/enqueue.
- New create returns an existing job DTO with `state="queued"` for non-idempotent submissions unless the worker has already completed via a clearly documented asynchronous trigger outside the request path.
- New create rows do not use `execution_mode="sync_inline"`; legacy support may remain read-compatible.
- Current job states do not regress.
- No full result/trades payload is stored in top rows.
- Job status/progress refresh cannot trigger compute and can communicate retry-window/coalescing state to UI.
- Focused local tests pass.
- Capacity/load evidence records create path behavior.
- Any compute-path change has Mac Studio benchmark evidence or a documented blocker.

# Implementation constraints

## Agent package boundaries

- Treat `package_contract.owns` as the write allow-list for this prompt.
- Do not edit `package_contract.forbidden` areas. If an implementation truly needs one, stop and report the required integration point instead of broadening scope silently.
- Keep shared integration edits small and explicit: route includes, DTO exports, CSS tokens, JS core APIs, migration chain, edge config.
- In final report, state whether the diff stayed inside `owns`; list any integration-point edits separately.

## API endpoint specification checklist

Before coding any new endpoint or browser-visible API addition, write the local contract in the implementation notes/tests with:

- `method/path`: browser-visible `/api/...` path and actual backend router path without duplicate `/api` prefix;
- `owner scope`: current user/account resolution and authorization check;
- `request DTO`: required/optional fields, defaults, validation, idempotency key, size limits;
- `response DTO`: shape, nullable fields, enums, links, timestamps, pagination;
- `status codes`: expected `200/201/204/400/401/403/404/409/422/429/500/503` semantics where applicable;
- `error payload`: compatible `RoehubError` envelope, field errors, retryability/correlation id when available;
- `pagination`: cursor/keyset/page semantics, max limit, stable ordering, or explicit `none`;
- `cache identity`: request hash/cache key/persistence identity impact or explicit `none`;
- `compatibility`: `none`, `compatible-change`, `breaking-change`, or `unknown` with migration/deprecation notes.

## Browser runtime evidence checklist

For every browser-visible change, collect and report runtime evidence:

- desktop screenshot, normally around `1440x1000`;
- mobile screenshot, normally around `390x844`;
- `snapshot` after the key state;
- console errors absent;
- failed same-origin network requests absent except expected auth redirects;
- auth state/protected route behavior verified when the page is protected;
- theme switcher changes base/accent/state but not financial colors;
- primary workflow has no overlapping requests;
- chart/canvas/SVG pages include a nonblank check;
- final report separates observed browser evidence, automated test evidence, inference, and assumptions.

## Gate failure classification

- Classify every failing gate as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.
- Do not run `publish-ci-deploy` with unresolved `introduced` failures or missing required browser/performance evidence.
- If a failure is pre-existing or environmental, include exact command, failure summary, and why it does or does not block this stage.

## API / contracts

- Public API contract: `compatible-change` if shape/status stays compatible.
- Runtime workflow: `compatible-change` or documented `unknown`.
- Request hash/cache identity: `none`.
- Persisted schema: `none` or additive `compatible-change`.
- Historical `sync_inline` storage literals: keep readable unless a migration/backfill is explicitly implemented and reversible.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- UI results page.
- AI configurator.
- New scoring algorithms.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py
uv run pytest -q tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py
uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Add the strongest feasible load/benchmark command and record exact command/output path in the final report.

# i18n / language contract

The Web UI v1 is multilingual. Every prompt in this pack must preserve this contract:

- default locale is `en`; secondary locale is `ru`;
- any new user-visible copy introduced by this stage must have both `en` and `ru` strings through the shared locale catalog/helper;
- do not localize routes, `/api/*` paths, DTO fields, enum values, market symbols, strategy ids, `job_id`, `variant_key`, config keys, or metric identifiers;
- rendered pages must keep `<html lang>` and root `data-locale` aligned with the selected locale;
- the language switcher must remain available from shell/account controls and must not compete with primary navigation;
- browser QA for any stage that adds or changes visible copy must include default `en` evidence and either `ru` locale-switch evidence or an explicit blocker;
- final report must state i18n impact: locale keys/catalogs touched, fallback behavior, and whether language-switch evidence was collected.

# publish-ci-deploy direct-main delivery contract

When all stage DoD, gates, browser evidence, and performance evidence required by this prompt pass, and `publish_after_success` is true, run `publish-ci-deploy` in direct-main mode. For this prompt pack, do not create a delivery branch, draft PR, or PR-based merge path. Work is published directly to `main` only after local gates pass.

A successful terminal state for this prompt means more than local green or a pushed commit. It must include, when the agent has authority and no external blocker remains:

- executor is on an up-to-date `main`, or has stopped with an exact blocker explaining why direct-main publish is unsafe;
- only intended scope is staged and committed; unrelated local changes are preserved and not staged;
- mandatory local gates for the stage pass before push;
- commit is pushed directly to `origin/main`;
- GitHub Actions and deploy workflow for `main` are monitored to green; failing checks are inspected and fixed if attributable to this diff, otherwise reported as blocker;
- local checkout is synchronized with `origin/main` after the push/deploy flow;
- Mac Studio repository checkout is synchronized with `origin/main` using `git pull --ff-only` from the actual repo checkout, normally `/Users/daniildegtyarev/Projects/roehub.com`;
- deployed runtime is updated through the repository deploy/runbook path, keeping the repo checkout and runtime bundle as separate surfaces when they differ;
- impacted services are restarted only when touched-path impact requires it; if impact is unclear, use the standard prod reload path from `publish-ci-deploy`;
- post-restart smoke verification is completed;
- final report names exact commands, host/paths used, commit SHA on `main`, CI/deploy status, restarted services, smoke result, or exact blocker.

Do not report successful publish/deploy while direct push to `origin/main`, main CI/deploy monitoring, Mac Studio git pull, required service restart/reload, or smoke verification remains pending.

# Final output: report format (strict)

Report in Russian with these exact sections:

- `Intent`: что реализовано и почему это нужно пользователю.
- `Scope`: bounded capability, routes, modules, files, and `owns`/`forbidden` compliance.
- `Design`: use cases, DTO, ports/adapters, migrations, JS modules, template fragments.
- `Contract impact`: classify public API, port, DTO, persisted schema, config, request hash/cache identity, browser-visible behavior, performance risk.
- `Tests`: exact commands, cwd, result, focused gates, lint/type gates, migration gates.
- `Docs`: docs changed, docs index result, or explicit reason docs were not changed.
- `Performance`: hot path impact, payload/latency/RSS/load checks, or explicit `none`.
- `Runtime evidence`: Playwright/browser evidence, automated test evidence, inference, assumptions.
- `Risks`: edge cases, migration/rollback risks, pre-existing/environmental/flaky failures.
- `Handoff`: stable exports, route includes, shared helpers, endpoint contracts for next agents.
- `Publish/deploy`: whether `publish-ci-deploy` ran, terminal state, or exact reason it was skipped.
