---
prompt_name: web_ui_backend_v1_04_dashboard
repo: roehub.com
branch: main
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
  protected_dashboard_required: true
  one_summary_request_target: true
  compact_payload_target_kb_compressed: 50
  degraded_panel_behavior_required: true
  no_fake_dashboard_kpis: true
  no_large_payloads: true
  no_overlap_polling: true
  browser_qa_required: true

task_toggles:
  implement_backend_read_model: true
  implement_dashboard_page: true
  implement_polling: true
  implement_persistence: false
  publish_after_success: true

package_contract:
  depends_on:
    - "01-shell-auth-register accepted"
    - "02-design-system-js-core accepted"
  owns:
    - "apps/api/routes/ui_dashboard.py"
    - "apps/api/dto/ui_dashboard.py"
    - "apps/api/wiring/modules/ui_dashboard.py"
    - "apps/web/templates/pages/dashboard.html"
    - "apps/web/templates/fragments/dashboard/**"
    - "apps/web/dist/js/pages/dashboard.js"
    - "apps/web/dist/css/pages/dashboard.css"
    - "tests/unit/apps/api/test_ui_dashboard_routes.py"
    - "tests/unit/apps/web/test_app_routes.py dashboard assertions"
  forbidden:
    - "settings/account package files"
    - "monitoring package files"
    - "strategy library package files"
    - "backtests package files"
    - "identity secret storage internals"
  integration_points:
    - "apps/api/main.py route include"
    - "apps/api/wiring/modules/__init__.py export"
    - "JS core api.js/poller.js"
    - "dashboard DTO contract"
  handoff:
    - "bounded dashboard summary endpoint and page evidence"

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
  - "Do not invent dashboard KPIs/metrics that are not backed by accepted strategy/backtest/account read models."

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
  - "Exact dashboard KPIs depend on read models accepted after Stages 5-9; render unavailable panels as degraded/empty rather than fake values."
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
- Exact dashboard KPIs are intentionally not frozen yet; use only accepted read models and explicit degraded/unavailable states.

## Requirements (Must)

- Add `/ui/dashboard/summary` backend route and DTO.
- Keep browser path `/api/ui/dashboard/summary`.
- Keep payload bounded; target < 50 KB compressed.
- Use ports/query services/ACL for cross-context assembly.
- Use only real accepted read-model fields; never fabricate KPI values to fill the UI.
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
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

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
- Missing/unaccepted KPI sources render as degraded/unavailable, not as fake zero/default business values.
- Polling pauses on hidden tab and does not overlap.
- Financial deltas keep fixed semantic colors in all themes.
- Browser screenshot/snapshot and console/network evidence exist.

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
