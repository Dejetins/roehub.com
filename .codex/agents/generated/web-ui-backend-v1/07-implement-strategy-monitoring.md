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
    - path: apps/api/wiring/modules/strategy.py
      why: "strategy composition root"
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
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

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

Report in Russian: `Intent`, `Scope`, `Design`, `Contract impact`, `Tests`, `Runtime evidence`, `Performance`, `Risks`, `Handoff`, `Publish/deploy`.
