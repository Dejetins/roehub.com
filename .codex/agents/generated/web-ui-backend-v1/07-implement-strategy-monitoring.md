---
prompt_name: web_ui_backend_v1_07_backend_live_read_models
repo: roehub.com
branch: main
scope: "Этап 7: backend-only live/read-model layer for the already implemented /dashboard and /strategies UI."

language:
  implementation: python_fastapi_ports_adapters_tests
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "DDD, contracts, security, backend gates, scoped delivery"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 7 source of truth, adjusted by this prompt"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "current UI baseline and no-layout-change contract"
  task_entrypoints:
    - path: apps/api/routes/ui_dashboard.py
      why: "existing /ui/dashboard/summary backend route used by current /dashboard UI"
    - path: apps/api/dto/ui_dashboard.py
      why: "existing DashboardSummaryResponse contract used by current /dashboard UI"
    - path: apps/api/wiring/modules/ui_dashboard.py
      why: "current dashboard query service and degraded source inventory"
    - path: apps/api/routes/ui_strategies_dashboard.py
      why: "existing /ui/strategies/dashboard backend route used by current /strategies UI"
    - path: apps/api/dto/ui_strategies_dashboard.py
      why: "existing StrategyDashboardResponse contract used by current /strategies UI"
    - path: apps/api/wiring/modules/ui_strategies_dashboard.py
      why: "current strategies query service and degraded source inventory"
    - path: docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
      why: "existing strategy realtime substrate contract; read only when implementing stream-backed sources"
  read_only_ui_contract:
    - path: apps/web/templates/pages/dashboard.html
      why: "current /dashboard UI source of truth; do not edit"
    - path: apps/web/templates/pages/strategies.html
      why: "current /strategies UI source of truth; do not edit"
    - path: apps/web/dist/js/pages/dashboard.js
      why: "current polling/autorefresh consumer contract; do not edit"
    - path: apps/web/dist/js/pages/strategies.js
      why: "current polling/autorefresh consumer contract; do not edit"
    - path: apps/web/dist/js/core/poller.js
      why: "current no-overlap/hidden-tab/retry_after behavior; do not edit"
    - path: apps/web/dist/js/core/sse.js
      why: "existing generic SSE helper, but this stage must not wire browser SSE"
  conditional_bundles:
    redis_runtime:
      read_when: "only if wiring real Redis stream-backed source data or backend-only SSE route"
      paths:
        - apps/worker/strategy_live_runner
        - src/trading/contexts/strategy/adapters/outbound
        - docs/runbooks/strategy-live-worker.md
    tests:
      read_when: "when adding focused tests"
      paths:
        - tests/unit/apps/api/test_ui_dashboard_routes.py
        - tests/unit/apps/api/test_ui_strategy_dashboard_routes.py
        - tests/unit/apps/web/test_app_routes.py
  consult_if_needed:
    - path: docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
      read_when: "if run lifecycle, stream names, or event payload semantics are unclear"

current_ui_source_of_truth:
  dashboard:
    route: /dashboard
    template: apps/web/templates/pages/dashboard.html
    js: apps/web/dist/js/pages/dashboard.js
    browser_endpoint: /api/ui/dashboard/summary
    backend_endpoint: /ui/dashboard/summary
    response_contract: DashboardSummaryResponse
    current_behavior: "polling/autorefresh/no-overlap via createPoller; panel layout must not change"
  strategies:
    route: /strategies
    template: apps/web/templates/pages/strategies.html
    js: apps/web/dist/js/pages/strategies.js
    browser_endpoint: /api/ui/strategies/dashboard
    backend_endpoint: /ui/strategies/dashboard
    response_contract: StrategyDashboardResponse
    current_behavior: "selected-strategy workstation with polling/autorefresh/no-overlap; panel layout must not change"
  monitoring:
    route: /monitoring
    current_behavior: "compatibility placeholder only"

hard_requirements:
  backend_only: true
  current_ui_is_primary_source_of_truth: true
  no_ui_template_changes: true
  no_ui_css_changes: true
  no_ui_page_js_changes: true
  preserve_current_dashboard_endpoint: true
  preserve_current_strategies_endpoint: true
  preserve_existing_dto_field_names: true
  additive_backend_changes_only: true
  owner_scope_required: true
  bounded_payloads_required: true
  manual_refresh_rate_limit_required: true
  source_freshness_required: true
  exchange_rate_limits_respected: true
  no_primary_monitoring_page_without_reference: true

task_toggles:
  extend_existing_dashboard_summary_backend: true
  extend_existing_strategies_dashboard_backend: true
  implement_new_ui_layout: false
  implement_web_template_changes: false
  implement_web_css_changes: false
  implement_web_page_js_changes: false
  implement_polling_fallback: false
  implement_sse_backend_route: "optional; only if existing stream substrate is clear and route can be tested without browser UI wiring"
  publish_after_success: true

package_contract:
  depends_on:
    - "04-dashboard current baseline accepted"
    - "06-selected-strategy current baseline accepted"
  owns:
    - "apps/api/routes/ui_dashboard.py"
    - "apps/api/routes/ui_strategies_dashboard.py"
    - "apps/api/dto/ui_dashboard.py"
    - "apps/api/dto/ui_strategies_dashboard.py"
    - "apps/api/wiring/modules/ui_dashboard.py"
    - "apps/api/wiring/modules/ui_strategies_dashboard.py"
    - "apps/api/routes/streams.py only if backend-only SSE route is added"
    - "src/trading/contexts/strategy/application/** only for ports/use-cases needed by read-models"
    - "src/trading/contexts/strategy/adapters/** only for read-only adapters needed by read-models"
    - "tests/unit/apps/api/test_ui_dashboard_routes.py"
    - "tests/unit/apps/api/test_ui_strategy_dashboard_routes.py"
    - "tests/unit/apps/api/test_strategy_stream_routes.py only if backend-only SSE route is added"
    - "tests/unit/apps/web/test_app_routes.py only for no-regression route/asset assertions"
  forbidden:
    - "apps/web/templates/**"
    - "apps/web/dist/css/**"
    - "apps/web/dist/js/pages/**"
    - "apps/web/dist/js/components/**"
    - "apps/web/dist/js/core/**"
    - "new apps/web/templates/pages/monitoring.html"
    - "changing /dashboard or /strategies panel inventory/layout/classes"
    - "changing current theme/locale/header/login/register UI behavior"
    - "backtest package files"
    - "settings/account package files"
    - "trading/order execution semantics beyond existing run/stop APIs"
  integration_points:
    - "existing browser `/api/ui/dashboard/summary`; backend `/ui/dashboard/summary`"
    - "existing browser `/api/ui/strategies/dashboard`; backend `/ui/strategies/dashboard`"
    - "optional browser `/api/stream/strategies`; backend `/stream/strategies` if SSE route is added"
    - "existing StrategyRepository and StrategyRunRepository"
    - "existing Redis stream/read-only strategy realtime output, only if verified"
  handoff:
    - "current UI keeps polling existing aggregate endpoints; backend fills more real fields and preserves degraded states"

skill_routing:
  - skill: architecture-design
    use_when: "only if a new stream reader port/adapter boundary is required and cannot be expressed with existing repositories/adapters"
    timing: "before implementation only if needed"
    reason: "backend-only live sources cross strategy runtime, Redis, API DTOs, and browser contracts"
  - skill: contract-impact-analysis
    use_when: "extending DTOs, adding stream route, changing source states, refresh semantics, owner checks, or rate-limit behavior"
    timing: "before implementation and final report"
    reason: "current UI consumes stable backend DTOs"
  - skill: backend-quality-gates
    use_when: "running API route/service tests, ruff, pyright"
    timing: "during verification"
    reason: "this stage is backend-only"
  - skill: backend-performance-evidence
    use_when: "measuring Redis/DB fan-out, payload size, rate-limit behavior, or backend-only SSE route load smoke"
    timing: "during performance verification if stream/read-model fan-out is implemented"
    reason: "live read-model sources can stress current host"
  - skill: publish-ci-deploy
    use_when: "all backend tests, contract checks, scoped-diff guard, and any required performance smoke pass"
    timing: "after verification"
    reason: "full Roehub direct-main delivery chain after complete success"

target_envs:
  - local-dev
  - github-actions
  - mac-studio

required_literals:
  - "/dashboard"
  - "/strategies"
  - "/api/ui/dashboard/summary"
  - "/ui/dashboard/summary"
  - "/api/ui/strategies/dashboard"
  - "/ui/strategies/dashboard"
  - "DashboardSummaryResponse"
  - "StrategyDashboardResponse"
  - "retry_after_seconds"
  - "refresh_status"
  - "sources"

non_goals:
  - "Do not edit current `/dashboard` UI files."
  - "Do not edit current `/strategies` UI files."
  - "Do not wire browser SSE in this prompt."
  - "Do not create a primary `/monitoring` page."
  - "Do not split the current aggregate browser endpoints into required subroutes unless payload/performance evidence proves it is necessary."
  - "Do not fake unavailable PnL/positions/fills/equity/monthly/hourly data."
  - "Do not bypass backend cache/coalescing/rate limits from browser refresh/autorefresh."

final_report_format:
  - "Intent: что реализовано и почему это нужно текущему UI"
  - "Scope: backend-only capability, routes, modules, files, owns/forbidden compliance"
  - "Current UI contract: какие dashboard/strategies endpoints and DTOs сохранены без UI diff"
  - "Design: use cases, DTO, ports/adapters, migrations, source/degraded states"
  - "Contract impact: public API, port, DTO, persisted schema, config, cache/request identity, browser-visible behavior, performance risk"
  - "Tests: exact commands, cwd, results, focused/lint/type/migration gates"
  - "Docs: updated docs or explicit reason no docs changed"
  - "Performance: payload/latency/fan-out/RSS/load checks, or explicit none"
  - "Runtime evidence: API/test evidence, optional smoke, inference, assumptions clearly separated"
  - "Risks: edge cases, migration/rollback, pre-existing/environmental/flaky failures"
  - "Handoff: stable endpoints, DTO fields, source names, optional stream contract for next agents"
  - "Publish/deploy: direct-main publish-ci-deploy terminal state or exact reason skipped"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ui_dashboard_routes.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py"
    expect: "passes; update focused tests for any backend source or DTO change"
  - cmd: "uv run pytest -q tests/unit/apps/api/test_strategy_stream_routes.py"
    expect: "passes if backend-only `/stream/strategies` route is added; otherwise explicitly not applicable"
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py"
    expect: "passes; route/asset no-regression only, not UI redesign"
  - cmd: "uv run ruff check apps/api src/trading/contexts/strategy tests/unit/apps/api tests/unit/apps/web"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes or classify unrelated/pre-existing failures"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --name-only -- apps/web/templates apps/web/dist/css apps/web/dist/js"
    expect: "empty; any output is a blocker unless the user explicitly broadens scope"

expected_primary_touches:
  - "apps/api/routes/ui_dashboard.py"
  - "apps/api/routes/ui_strategies_dashboard.py"
  - "apps/api/dto/ui_dashboard.py"
  - "apps/api/dto/ui_strategies_dashboard.py"
  - "apps/api/wiring/modules/ui_dashboard.py"
  - "apps/api/wiring/modules/ui_strategies_dashboard.py"
  - "tests/unit/apps/api/test_ui_dashboard_routes.py"
  - "tests/unit/apps/api/test_ui_strategy_dashboard_routes.py"

possible_secondary_touches:
  - "apps/api/routes/streams.py"
  - "apps/api/main/app.py"
  - "apps/api/wiring/modules/__init__.py"
  - "src/trading/contexts/strategy/application/**"
  - "src/trading/contexts/strategy/adapters/**"
  - "tests/unit/apps/api/test_strategy_stream_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md"

safety_notes:
  - "Current UI is the source of truth. This prompt must not change templates, page CSS, page JS, shell controls, layout, panel inventory, or route visual behavior."
  - "Existing aggregate endpoints are the primary browser contract. Prefer filling existing DTO fields and source states over adding browser subroutes."
  - "If current migrations/stores cannot support a panel, keep typed unavailable/degraded state instead of synthetic production data."
  - "Browser path `/api/...` maps to backend routes without duplicate `/api` prefix."
  - "Manual refresh/autorefresh limits already exist in current UI; backend must preserve `retry_after_seconds` and `next_allowed_refresh_at` semantics."
---

# Task

Implement Stage 7 backend-only live/read-model support for the current `/dashboard` and `/strategies` UI.

Done means:

- current `/dashboard` UI continues to consume `GET /api/ui/dashboard/summary`;
- current `/strategies` UI continues to consume `GET /api/ui/strategies/dashboard`;
- existing templates, CSS and page JS remain untouched;
- backend fills more real bounded source data where current repositories/read-models support it;
- unsupported panels remain typed `unavailable/degraded/stale` through existing DTO source states;
- owner scope, payload bounds, refresh metadata and manual refresh rate limiting stay enforced;
- optional backend-only SSE route is added only if the existing realtime substrate is clear and testable without browser UI wiring;
- quality gates and scoped-diff guard pass.

## Context / Current State

Current `/dashboard` is already implemented as a terminal workstation. It renders [apps/web/templates/pages/dashboard.html](/Users/daniildegtyarev/Projects/roehub.com/apps/web/templates/pages/dashboard.html), loads [apps/web/dist/js/pages/dashboard.js](/Users/daniildegtyarev/Projects/roehub.com/apps/web/dist/js/pages/dashboard.js), and polls `GET /api/ui/dashboard/summary` with no-overlap autorefresh via `createPoller`.

Current `/strategies` is already implemented as a selected-strategy workstation. It renders [apps/web/templates/pages/strategies.html](/Users/daniildegtyarev/Projects/roehub.com/apps/web/templates/pages/strategies.html), loads [apps/web/dist/js/pages/strategies.js](/Users/daniildegtyarev/Projects/roehub.com/apps/web/dist/js/pages/strategies.js), and polls `GET /api/ui/strategies/dashboard` with no-overlap autorefresh via `createPoller`.

The current UI is the source of truth. This stage is not allowed to redesign, reshape, or rewire the browser surface.

## Requirements (Must)

- Preserve `DashboardSummaryResponse` field names consumed by current `dashboard.js`.
- Preserve `StrategyDashboardResponse` field names consumed by current `strategies.js`.
- Preserve browser-visible `/api/ui/dashboard/summary` and `/api/ui/strategies/dashboard` contracts.
- Keep backend routers registered without duplicate `/api` prefix.
- Add or improve backend read-model/source data only through DDD ports/adapters or existing repositories.
- Maintain per-user owner scope before any read.
- Keep list rows, fills, alerts, charts and series bounded.
- Keep `sources[]`, `generated_at`, `refresh_status`, `next_allowed_refresh_at`, `retry_after_seconds`, panel `state`, and `degradation_reason` semantics.
- Keep unsupported data as typed degraded/unavailable state.
- Add focused backend tests for every new real source or optional stream route.
- Prove no UI files changed.
- Use `publish-ci-deploy` only after full backend success and scoped diff safety.

## Requirements (Should)

- Prefer improving existing aggregate endpoints over adding browser-visible subroutes.
- If backend-only SSE is implemented, keep it read-only, owner-scoped, bounded, and covered by API tests.
- Add payload-size assertions for large aggregate responses when adding rows/series.
- Add source-level freshness/lag metadata for any real live source.

## Requirements (Nice-to-have)

- Add low-cardinality metrics for backend route latency/payload size or active backend-only stream connections if the existing observability layer supports it without scope creep.

# Context acquisition protocol

Read `.codex/AGENTS.md`, this prompt, the current dashboard/strategies API route/DTO/wiring modules, and the read-only UI contract files listed in front matter. Stop after the current backend DTO contracts and UI-consumed fields are clear.

Reading budget: keep pre-implementation reading to `<= 10 files`, `<= ~50k tokens`. Expand only for stream-substrate ambiguity, failing tests, or a new backend adapter boundary.

Do not preload unrelated `/backtests`, settings/account, or full worker code. Do not inspect visual references unless needed to confirm that this prompt must not change UI.

# Reading manifest

Use front matter `context_sources`. Treat `read_only_ui_contract` as read-only evidence, not write scope.

# Work plan (agent should follow)

1. Record current `/dashboard` and `/strategies` backend contracts: endpoint, DTO fields, sources, refresh semantics, payload bounds.
2. Identify which currently degraded/unavailable panels can be backed by existing repositories, Redis streams, or safe read-only projections.
3. Implement backend-only read-model/source improvements inside existing endpoints.
4. If and only if the stream substrate is clear, add a backend-only `/stream/strategies` route with tests; do not wire it into browser JS.
5. Add/update focused API tests and any route no-regression web tests.
6. Run quality gates, including the `git diff --name-only -- apps/web/templates apps/web/dist/css apps/web/dist/js` guard.
7. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- `GET /ui/dashboard/summary` remains compatible with current `DashboardSummaryResponse`.
- `GET /ui/strategies/dashboard` remains compatible with current `StrategyDashboardResponse`.
- Current `/dashboard` and `/strategies` templates/CSS/page JS have no diff.
- Any newly real source is owner-scoped, bounded, freshness-aware, and tested.
- Any unsupported source remains typed `unavailable/degraded` with explicit reason.
- Manual refresh still returns rate-limit metadata in DTO instead of failing page refresh.
- Optional `/stream/strategies` route, if added, is read-only, owner-scoped, bounded, and tested; current UI does not depend on it.
- `/monitoring` is not introduced as a primary page.
- No synthetic production PnL/positions/fills/equity data is invented.

# Implementation constraints

## Agent package boundaries

- Treat `package_contract.owns` as the write allow-list for this prompt.
- Treat `package_contract.forbidden` as hard forbidden. If the backend change appears to require a UI edit, stop and report the required integration point instead of broadening scope.
- In the final report, include the exact output of the scoped UI diff guard.

## API endpoint specification checklist

Before coding any new endpoint or DTO addition, write the local contract in tests or implementation notes with:

- `method/path`: browser-visible `/api/...` path and backend router path without duplicate `/api` prefix;
- `owner scope`: current user/account resolution and authorization check;
- `request DTO`: query/body fields, defaults, validation, idempotency or explicit none;
- `response DTO`: shape, nullable fields, enums, timestamps, pagination and caps;
- `status codes`: expected `200/400/401/403/404/409/422/429/500/503` semantics where applicable;
- `error payload`: `RoehubError` envelope, retryability/correlation id when available;
- `pagination`: cursor/keyset/page semantics, max limit, stable ordering, or explicit `none`;
- `cache identity`: request hash/cache key/persistence identity impact or explicit `none`;
- `compatibility`: `none`, `compatible-change`, `breaking-change`, or `unknown` with migration/deprecation notes.

## Backend runtime evidence checklist

For every backend source/route change, collect and report:

- focused API route tests;
- payload-size bound evidence for aggregate responses touched by this stage;
- auth/owner-scope evidence;
- refresh/rate-limit evidence when refresh semantics are touched;
- source degradation evidence for unavailable dependencies;
- performance/load smoke only when new Redis/DB fan-out or SSE route is added.

Browser screenshots are not required because UI changes are forbidden. If any browser-visible behavior changes unexpectedly, stop and report it as a scope conflict.

## Gate failure classification

- Classify every failing gate as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.
- Do not run `publish-ci-deploy` with unresolved `introduced` failures or non-empty forbidden UI diff.
- If a failure is pre-existing or environmental, include exact command, failure summary, and why it does or does not block this backend-only stage.

## API / contracts

- Public API contract: `compatible-change` only.
- DTO schema: `compatible-change`; existing consumed fields must not be removed/renamed.
- Port contract: `compatible-change` if read-only port/adapter is added.
- Persisted schema: expected `none`; if migration becomes necessary, stop and report the migration need unless it is already explicitly covered by the plan.
- Browser-visible behavior: `none` intended; current UI should render the same layout and consume the same endpoints.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- UI layout, template, CSS, page JS or visual changes.
- Browser SSE integration.
- Rebuilding the strategy live worker.
- Trading/order execution controls beyond existing run/stop API behavior.
- Backtests.
- New primary `/monitoring` page.
- Exchange upstream calls directly triggered by browser requests.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_ui_dashboard_routes.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py
uv run pytest -q tests/unit/apps/web/test_app_routes.py
uv run ruff check apps/api src/trading/contexts/strategy tests/unit/apps/api tests/unit/apps/web
uv run pyright
python -m tools.docs.generate_docs_index --check
git diff --name-only -- apps/web/templates apps/web/dist/css apps/web/dist/js
```

If adding a backend-only stream route:

```bash
uv run pytest -q tests/unit/apps/api/test_strategy_stream_routes.py
```

The final command must print no changed files. Any changed file under `apps/web/templates`, `apps/web/dist/css`, or `apps/web/dist/js` is a blocker for this prompt.

# i18n / language contract

This backend-only stage must not add user-visible UI copy. If an API error or source/degradation reason becomes browser-visible through existing UI fields, keep technical identifiers stable and add locale keys only if the current UI already localizes that surface. Do not edit templates or page JS to introduce copy.

# publish-ci-deploy direct-main delivery contract

When all stage DoD, gates, scoped UI diff guard, and performance evidence required by this prompt pass, and `publish_after_success` is true, run `publish-ci-deploy` in direct-main mode. For this prompt pack, do not create a delivery branch, draft PR, or PR-based merge path. Work is published directly to `main` only after local gates pass.

A successful terminal state for this prompt means more than local green or a pushed commit. It must include, when the agent has authority and no external blocker remains:

- executor is on an up-to-date `main`, or has stopped with an exact blocker explaining why direct-main publish is unsafe;
- only intended backend scope is staged and committed; unrelated local changes are preserved and not staged;
- mandatory local gates and scoped UI diff guard pass before push;
- commit is pushed directly to `origin/main`;
- GitHub Actions and deploy workflow for `main` are monitored to green;
- local checkout is synchronized with `origin/main` after the push/deploy flow;
- Mac Studio repository checkout is synchronized with `origin/main` using `git pull --ff-only` from the actual repo checkout, normally `/Users/daniildegtyarev/Projects/roehub.com`;
- deployed runtime is updated through the repository deploy/runbook path;
- impacted backend services are restarted/reloaded only when touched-path impact requires it;
- post-restart smoke verification covers `/api/ui/dashboard/summary` and `/api/ui/strategies/dashboard` auth behavior and, when possible, authenticated response shape;
- final report names exact commands, host/paths used, commit SHA on `main`, CI/deploy status, restarted services, smoke result, or exact blocker.

Do not report successful publish/deploy while direct push to `origin/main`, main CI/deploy monitoring, Mac Studio git pull, required service restart/reload, or smoke verification remains pending.

# Final output: report format (strict)

Report in Russian with these exact sections:

- `Intent`: что реализовано и почему это нужно текущему UI.
- `Scope`: backend-only capability, routes, modules, files, and `owns`/`forbidden` compliance.
- `Current UI contract`: какие `/dashboard` и `/strategies` endpoints/DTOs сохранены, и подтверждение отсутствия UI diff.
- `Design`: use cases, DTO, ports/adapters, migrations, source/degraded states.
- `Contract impact`: classify public API, port, DTO, persisted schema, config, request hash/cache identity, browser-visible behavior, performance risk.
- `Tests`: exact commands, cwd, result, focused gates, lint/type gates, migration gates.
- `Docs`: docs changed, docs index result, or explicit reason docs were not changed.
- `Performance`: hot path impact, payload/latency/RSS/load checks, or explicit `none`.
- `Runtime evidence`: API/backend evidence, automated test evidence, inference, assumptions.
- `Risks`: edge cases, migration/rollback risks, pre-existing/environmental/flaky failures.
- `Handoff`: stable endpoints, DTO fields, source names, optional stream contract for next agents.
- `Publish/deploy`: whether `publish-ci-deploy` ran, terminal state, or exact reason it was skipped.
