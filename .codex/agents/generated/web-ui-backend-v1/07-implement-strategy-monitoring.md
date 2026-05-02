---
prompt_name: web_ui_backend_v1_07_strategy_monitoring
repo: roehub.com
branch: current
scope: "Этап 7: strategy monitoring page, compact read-models и browser-facing SSE/polling bridge."

language:
  implementation: python_fastapi_jinja_css_js_redis
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "DDD, contracts, security, browser verification"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 7 source of truth"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "monitoring visual/layout/theme rules"
  task_entrypoints:
    - path: docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
      why: "existing realtime substrate contract"
    - path: apps/api/routes/strategies.py
      why: "existing strategy run/stop and list API"
    - path: apps/web/main/app.py
      why: "monitoring protected route"
    - path: apps/web/templates/pages/monitoring.html
      why: "monitoring page target"
  conditional_bundles:
    redis_runtime:
      read_when: "when implementing Redis stream reader/SSE bridge"
      paths:
        - apps/worker/strategy_live_runner
        - src/trading/contexts/strategy/adapters/outbound
        - docs/runbooks/strategy-live-worker.md
    tests:
      read_when: "when adding route/stream tests"
      paths:
        - tests/unit/apps/api
        - tests/unit/apps/web/test_app_routes.py
  consult_if_needed:
    - path: docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
      read_when: "if run lifecycle or stream payload semantics are unclear"

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

hard_requirements:
  use_existing_realtime_substrate: true
  sse_auth_owner_scope_required: true
  polling_fallback_required: true
  bounded_payloads_required: true
  no_overlap_polling_required: true
  browser_qa_required: true

task_toggles:
  implement_monitoring_read_models: true
  implement_sse_bridge: true
  implement_polling_fallback: true
  implement_monitoring_page: true
  publish_after_success: true

package_contract:
  depends_on:
    - "01-shell-auth-register accepted"
    - "02-design-system-js-core accepted"
    - "06-strategy-library-detail accepted or Strategy API contract stable"
  owns:
    - "apps/api/routes/ui_strategies_monitoring.py"
    - "apps/api/routes/streams.py strategy stream endpoints"
    - "apps/api/dto/ui_strategies_monitoring.py"
    - "apps/api/wiring/modules/ui_strategies_monitoring.py"
    - "apps/web/templates/pages/monitoring.html"
    - "apps/web/templates/fragments/monitoring/**"
    - "apps/web/dist/js/pages/monitoring.js"
    - "apps/web/dist/css/pages/monitoring.css"
    - "tests/unit/apps/api/test_ui_strategy_monitoring_routes.py"
  forbidden:
    - "strategy library create/detail templates"
    - "backtest package files"
    - "settings/account package files"
    - "trading live-control semantics beyond documented run/stop calls"
  integration_points:
    - "apps/api/main.py route include"
    - "SSE event contract"
    - "JS core sse.js/poller.js"
    - "strategy run/stop existing API"
  handoff:
    - "owner-scoped monitoring DTOs and stream/polling fallback"

skill_routing:
  - skill: architecture-design
    use_when: "if stream reader port/adapter boundary or ACL mapping is not clear"
    timing: "before implementation only if needed"
    reason: "SSE bridge crosses strategy runtime, Redis, API delivery, and browser contracts"
  - skill: contract-impact-analysis
    use_when: "adding `/ui/strategies/*`, `/stream/strategies`, DTOs, stream event semantics, auth/owner checks, or polling defaults"
    timing: "before implementation and final report"
    reason: "monitoring adds public API/SSE contracts"
  - skill: backend-quality-gates
    use_when: "running route/stream/service tests, ruff, pyright"
    timing: "during verification"
    reason: "backend route and stream code must be verified"
  - skill: browser-qa-evidence
    use_when: "verifying monitoring page, SSE reconnect/polling fallback, 401 handling, mobile layout"
    timing: "after backend tests"
    reason: "monitoring is live browser behavior"
  - skill: playwright
    use_when: "capturing screenshots/snapshots"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI"
  - skill: backend-performance-evidence
    use_when: "measuring Redis/DB fan-out, SSE connection count, reconnect behavior, or load smoke"
    timing: "during performance verification"
    reason: "live monitoring can stress current host"
  - skill: publish-ci-deploy
    use_when: "all tests, browser QA, stream auth, and performance smoke pass"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "/monitoring"
  - "/api/ui/strategies/monitor"
  - "/api/stream/strategies"
  - "/stream/strategies"
  - "last_event_id"
  - "SSE"
  - "polling fallback"

non_goals:
  - "Do not rewrite strategy live worker."
  - "Do not implement trading/order routing."
  - "Do not use HTMX as high-frequency live transport."
  - "Do not leak another user's stream events."

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
  - "Publish/deploy: terminal state publish-ci-deploy or exact reason it was skipped"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ui_strategy_monitoring_routes.py tests/unit/apps/api/test_strategy_stream_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes; create focused tests if missing"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/strategy tests/unit/apps/api tests/unit/apps/web"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/api/routes/ui_strategies_monitoring.py"
  - "apps/api/routes/streams.py"
  - "apps/api/dto/ui_strategies_monitoring.py"
  - "apps/api/wiring/modules/ui_strategies_monitoring.py"
  - "apps/api/main/app.py"
  - "src/trading/contexts/strategy/application/**"
  - "apps/web/templates/pages/monitoring.html"
  - "apps/web/templates/fragments/monitoring/*"
  - "apps/web/dist/js/pages/monitoring.js"
  - "apps/web/dist/css/pages/monitoring.css"
  - "tests/unit/apps/api/test_ui_strategy_monitoring_routes.py"
  - "tests/unit/apps/api/test_strategy_stream_routes.py"

possible_secondary_touches:
  - "apps/api/wiring/modules/__init__.py"
  - "docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md"

safety_notes:
  - "Browser path `/api/stream/strategies` maps to backend `/stream/strategies`."
  - "SSE is read-only and must authorize before stream read."
  - "Keep rows/fills/chart points bounded."
---

# Task

Implement Stage 7 strategy monitoring.

Done means:

- protected `/monitoring` page exists;
- backend has compact monitor/snapshot/positions/fills/equity endpoints;
- backend has authorized SSE bridge over existing Redis Streams or a documented polling-only fallback if stream substrate is unavailable;
- page reconnects/downgrades safely;
- Playwright evidence exists.

## Context / Current State

- Strategy API supports list/get/run/stop.
- Strategy runtime already publishes realtime primitives to Redis Streams.
- UI lacks browser-facing monitoring read models and SSE bridge.

## Requirements (Must)

- Reuse existing strategy runtime and Redis stream substrate.
- Add owner-scoped authorization before any stream read.
- Limit list rows, fills, alerts and chart points.
- Implement 401 stream stop/redirect behavior.
- Implement hidden-tab pause/no-overlap polling.
- Add focused backend/web tests.
- Use `publish-ci-deploy` only after full success.

## Requirements (Should)

- Use SSE for state/events and polling fallback for snapshots.
- Keep at most one idle SSE connection per monitoring page.
- Make mobile list/detail collapse into tabs.

## Requirements (Nice-to-have)

- Add minimal active SSE connection metric if observability layer supports it without cardinality risk.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 7, realtime stream doc, then task entrypoints. Expand into worker/runtime only when implementing the stream reader.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Define monitoring DTOs and stream event contract.
2. Add application query/stream reader ports and adapters.
3. Add routes/wiring.
4. Implement page, JS SSE wrapper integration and CSS.
5. Add tests for auth/owner scope, fallback, 401, bounded payloads.
6. Run browser QA and gates.
7. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- Strategy list and selected snapshot render from backend DTO.
- Start/stop actions reflect state within one refresh cycle.
- SSE reconnects or degrades to polling.
- 401 stops stream and sends user to login.
- Hidden tab pauses polling.
- Mobile layout folds list/detail into tabs.
- Financial colors remain invariant.

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

- Public API contract: `compatible-change`.
- DTO schema: `compatible-change`.
- Port contract: `compatible-change` if stream reader port is added.
- Persisted schema: expected `none`.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Rebuilding strategy live worker.
- Trading execution controls beyond existing run/stop.
- Backtests.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_ui_strategy_monitoring_routes.py tests/unit/apps/api/test_strategy_stream_routes.py tests/unit/apps/web/test_app_routes.py
uv run ruff check apps/api apps/web src/trading/contexts/strategy tests/unit/apps/api tests/unit/apps/web
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/monitoring
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/monitoring-desktop.png
```

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
