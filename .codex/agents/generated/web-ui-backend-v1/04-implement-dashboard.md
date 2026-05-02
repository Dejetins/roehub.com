---
prompt_name: web_ui_backend_v1_04_dashboard
repo: roehub.com
branch: current
scope: "Этап 4: защищенный dashboard overview и компактный backend read-model."

language:
  implementation: python_fastapi_jinja_css_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "DDD, contracts, gates, browser verification"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 4 source of truth"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "dashboard visual/components/theme rules"
  task_entrypoints:
    - path: apps/api/main/app.py
      why: "backend router include composition root"
    - path: apps/web/main/app.py
      why: "dashboard route and protected gate"
    - path: apps/web/templates/pages/dashboard.html
      why: "dashboard page target"
    - path: apps/web/dist/js/core/api.js
      why: "shared API client/error handling"
  conditional_bundles:
    backend_sources:
      read_when: "building dashboard summary from existing data"
      paths:
        - apps/api/routes/backtests.py
        - apps/api/routes/strategies.py
        - apps/api/routes/identity.py
    tests:
      read_when: "adding route/read-model tests"
      paths:
        - tests/unit/apps/api/test_backtests_routes.py
        - tests/unit/apps/web/test_app_routes.py
  consult_if_needed:
    - path: docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
      read_when: "only if dashboard health needs strategy realtime details"

hard_requirements:
  protected_dashboard_required: true
  one_summary_request_target: true
  compact_payload_target_kb_compressed: 50
  degraded_panel_behavior_required: true
  no_large_payloads: true
  no_overlap_polling: true
  browser_qa_required: true

task_toggles:
  implement_backend_read_model: true
  implement_dashboard_page: true
  implement_polling: true
  implement_persistence: false
  publish_after_success: true

skill_routing:
  - skill: architecture-design
    use_when: "if a new cross-context read-model/ACL boundary is not obvious from existing contexts"
    timing: "before implementation only if needed"
    reason: "dashboard aggregates multiple bounded contexts"
  - skill: contract-impact-analysis
    use_when: "adding `/ui/dashboard/*` endpoints, DTOs, degraded source semantics, or browser polling defaults"
    timing: "before implementation and final report"
    reason: "dashboard adds public API and browser-visible behavior"
  - skill: backend-quality-gates
    use_when: "running route/DTO/service tests, ruff, pyright"
    timing: "during verification"
    reason: "backend read-model and web routes are Python surfaces"
  - skill: browser-qa-evidence
    use_when: "verifying dashboard render, hidden-tab/no-overlap behavior, console/network, theme colors"
    timing: "after backend tests"
    reason: "dashboard acceptance is browser-visible"
  - skill: playwright
    use_when: "capturing browser snapshots/screenshots"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI"
  - skill: backend-performance-evidence
    use_when: "making payload/latency/fan-out claims or adding dashboard load smoke"
    timing: "during performance verification"
    reason: "dashboard summary can create backend fan-out"
  - skill: publish-ci-deploy
    use_when: "all DoD, tests, browser QA, and payload checks pass"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "/dashboard"
  - "/api/ui/dashboard/summary"
  - "/ui/dashboard/summary"
  - "degraded"
  - "terminal-orange"

non_goals:
  - "Do not implement monitoring SSE in dashboard."
  - "Do not materialize large strategy/backtest details."
  - "Do not add persistence unless explicitly required by read-model design."

final_report_format:
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ui_dashboard_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes; create focused tests if files do not exist"
  - cmd: "uv run ruff check apps/api apps/web src tests/unit/apps/api tests/unit/apps/web"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/api/routes/ui_dashboard.py"
  - "apps/api/dto/ui_dashboard.py"
  - "apps/api/wiring/modules/ui_dashboard.py"
  - "apps/api/main/app.py"
  - "apps/web/templates/pages/dashboard.html"
  - "apps/web/templates/fragments/dashboard/*"
  - "apps/web/dist/js/pages/dashboard.js"
  - "apps/web/dist/css/pages/dashboard.css"
  - "tests/unit/apps/api/test_ui_dashboard_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"

possible_secondary_touches:
  - "src/trading/contexts/*/application/**"
  - "apps/api/wiring/modules/__init__.py"

safety_notes:
  - "`/api/ui/dashboard/summary` is browser path; backend router path is `/ui/dashboard/summary`."
  - "Dashboard must not import private domain internals without mapper/ACL."
  - "One failed source should degrade a panel, not the whole page, unless auth fails."
---

# Task

Implement Stage 4 dashboard overview.

Done means:

- protected `/dashboard` renders;
- backend exposes compact dashboard summary read-model;
- page uses one summary request where practical;
- degraded panel behavior exists;
- polling is 10-15s, no-overlap, hidden-tab aware;
- Playwright evidence exists.

## Context / Current State

- Existing backend has auth, strategies and backtest jobs APIs.
- There is no current dashboard read-model.
- Stage 2 JS core should provide `api.js` and `poller.js`.

## Requirements (Must)

- Add `/ui/dashboard/summary` backend route and DTO.
- Keep browser path `/api/ui/dashboard/summary`.
- Keep payload bounded; target < 50 KB compressed.
- Use ports/query services/ACL for cross-context assembly.
- Add focused API and web tests.
- Verify browser behavior and theme financial colors.
- Use `publish-ci-deploy` only after full success.

## Requirements (Should)

- Prefer partial DTO with typed degraded source state over throwing whole-page errors.
- Avoid many browser calls on first render.
- Keep polling interval configurable or clearly centralized.

## Requirements (Nice-to-have)

- Optional cursor endpoints for alerts/recent jobs only if summary would become too large.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 4, design manifest dashboard rules, then entrypoints. Expand only for the contexts used by summary data.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Define dashboard summary DTO and failure/degraded semantics.
2. Implement use case/query service and route/wiring.
3. Implement protected dashboard page and JS polling.
4. Add tests for auth, DTO, degraded source, payload bounds where practical.
5. Run browser QA and quality gates.
6. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- One summary request can render dashboard.
- Auth-required behavior matches other protected routes.
- Source failure degrades one panel, not the whole page.
- Polling pauses on hidden tab and does not overlap.
- Financial deltas keep fixed semantic colors in all themes.
- Browser screenshot/snapshot and console/network evidence exist.

# Implementation constraints

## API / contracts

- Public API contract: `compatible-change`.
- DTO schema: `compatible-change`.
- Persisted schema: `none`.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Strategy live monitoring details.
- Backtest result details.
- Account settings persistence.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_ui_dashboard_routes.py tests/unit/apps/web/test_app_routes.py
uv run ruff check apps/api apps/web src tests/unit/apps/api tests/unit/apps/web
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/dashboard
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/dashboard-desktop.png
```

# Final output: report format (strict)

Report in Russian: `Intent`, `Scope`, `Design`, `Contract impact`, `Tests`, `Runtime evidence`, `Performance`, `Risks`, `Handoff`, `Publish/deploy`.
