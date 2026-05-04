---
prompt_name: web_ui_backend_v1_06_strategy_dashboard
repo: roehub.com
branch: main
scope: "Этап 6: /strategies как selected-strategy analytics workstation по strategy_statistic.png."

language:
  implementation: jinja_css_plain_js_python_tests
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and browser verification rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 6 source of truth"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "strategy dashboard visual/reference rules"
  task_entrypoints:
    - path: apps/web/main/app.py
      why: "strategies routes and /strategies/new"
    - path: apps/web/templates
      why: "current strategy list/create/detail routes replacement targets"
    - path: apps/web/dist/strategy_ui.js
      why: "old strategy JS replacement target and API usage reference"
    - path: apps/api/routes/strategies.py
      why: "existing Strategy API contract"
  conditional_bundles:
    canonical_reference:
      read_when: "always before implementation; this stage is reference-fidelity gated"
      paths:
        - /Users/daniildegtyarev/Projects/roehub_web_ui/strategy_statistic.png
    strategy_contract:
      read_when: "if create/clone/delete/run API semantics are unclear"
      paths:
        - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
        - tests/unit/apps/api/test_strategies_routes.py
    market_reference:
      read_when: "if builder needs instruments/indicators dropdowns"
      paths:
        - apps/api/routes/market_data_reference.py
        - apps/api/routes/indicators.py
  consult_if_needed:
    - path: docs/architecture/apps/web/web-strategy-ui-crud-builder-delete-v1.md
      read_when: "if current Web Strategy UI behavior must be preserved"

style_references:
  design_manifest:
    path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
    purpose: "визуальный source of truth для токенов, тем, layouts, density и accessibility"
  external_reference_root:
    path: /Users/daniildegtyarev/Projects/roehub_web_ui
    purpose: "reference screenshots/assets; canonical for this stage is strategy_statistic.png"
  canonical_reference:
    route: /strategies
    path: /Users/daniildegtyarev/Projects/roehub_web_ui/strategy_statistic.png
    fidelity: "hard reference-shaped contract for selected-strategy analytics workstation"
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
  preserve_strategy_api: true
  preserve_immutable_strategy_model: true
  support_strategies_new: true
  backend_read_model_required: true
  data_source_inventory_required: true
  manual_refresh_required: true
  autorefresh_controls_required: true
  branded_dropdowns_required: true
  visible_native_select_forbidden: true
  remove_old_strategy_ui_dependency: true
  reference_fidelity_required: true
  reject_generic_library_cards: true
  browser_qa_required: true

task_toggles:
  implement_strategy_list: true
  implement_strategy_create: true
  implement_strategy_deeplink_state: true
  implement_new_backend_read_model: true
  publish_after_success: true

package_contract:
  depends_on:
    - "01-shell-auth-register accepted"
    - "02-design-system-js-core accepted"
  owns:
    - "apps/web/templates/pages/strategies.html"
    - "apps/web/templates/fragments/strategies/**"
    - "apps/web/dist/js/pages/strategies.js"
    - "apps/web/dist/css/pages/strategies.css"
    - "apps/api/routes/ui_strategies_dashboard.py"
    - "apps/api/dto/ui_strategies_dashboard.py"
    - "apps/api/wiring/modules/ui_strategies_dashboard.py"
    - "tests/unit/apps/api/test_ui_strategy_dashboard_routes.py"
    - "tests/unit/apps/web/test_app_routes.py strategy assertions"
  forbidden:
    - "separate /monitoring primary page files"
    - "backtest package files"
    - "settings/account package files"
    - "mutable Strategy API semantics"
  integration_points:
    - "existing /api/strategies* routes"
    - "browser `/api/ui/strategies/dashboard*`; backend `/ui/strategies/dashboard*`"
    - "strategy create/clone payload shape"
    - "shared JS api.js"
    - "refresh/autorefresh helper"
    - "branded dropdown/selector component"
  handoff:
    - "selected-strategy dashboard route/read-model without old strategy_ui.js dependency"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing strategy route behavior, browser defaults, Strategy API assumptions, create/clone payload shape, or adding read-model endpoint"
    timing: "before implementation and final report"
    reason: "strategy API shape and browser create flow are contract surfaces"
  - skill: backend-quality-gates
    use_when: "running web/API route tests, ruff, pyright"
    timing: "during verification"
    reason: "stage touches Python web and possibly API calls/tests"
  - skill: browser-qa-evidence
    use_when: "verifying selected strategy dashboard, `/strategies/new` compatibility, reference fidelity, console/network, responsive layout"
    timing: "after local tests"
    reason: "selected-strategy dashboard is browser-visible"
  - skill: playwright
    use_when: "capturing Playwright evidence"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI"
  - skill: publish-ci-deploy
    use_when: "all strategy UI workflows and gates pass with scoped diff"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "/strategies"
  - "/strategies/new"
  - "/strategies/{strategy_id}"
  - "strategy_statistic.png"
  - "/api/ui/strategies/dashboard"
  - "/ui/strategies/dashboard"
  - "/api/strategies"
  - "/api/strategies/clone"
  - "strategy_ui.js"
  - "refresh_status"
  - "retry_after_seconds"
  - "branded dropdown"

non_goals:
  - "Do not build a generic strategy library/card grid."
  - "Do not create a separate primary `/monitoring` page for this reference."
  - "Do not make strategy mutable update endpoint."
  - "Do not fake metrics that backend does not provide."
  - "Do not use visible native select/system dropdowns for strategy selectors, filters, sort or autorefresh."
  - "Do not run unbounded aggregation over strategy events for first paint."

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
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py"
    expect: "passes; add strategy UI asset tests if needed"
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ui_strategy_dashboard_routes.py"
    expect: "passes if UI strategy dashboard endpoint is added"
  - cmd: "uv run pytest -q tests/unit/apps/api/test_strategies_routes.py"
    expect: "passes if API assumptions/touched tests exist"
  - cmd: "uv run ruff check apps/web apps/api tests/unit/apps/web tests/unit/apps/api"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes if Python touched"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/web/templates/pages/strategies.html"
  - "apps/web/templates/fragments/strategies/*"
  - "apps/web/dist/js/pages/strategies.js"
  - "apps/web/dist/css/pages/strategies.css"
  - "apps/api/routes/ui_strategies_dashboard.py"
  - "apps/api/dto/ui_strategies_dashboard.py"
  - "apps/api/wiring/modules/ui_strategies_dashboard.py"
  - "tests/unit/apps/api/test_ui_strategy_dashboard_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"

possible_secondary_touches:
  - "apps/web/main/app.py"
  - "apps/api/main/app.py"
  - "apps/api/wiring/modules/__init__.py"

safety_notes:
  - "Current `/strategies/new` must remain a compatibility entrypoint, but not a separate visual page."
  - "Create/clone must preserve canonical indicator payload shape."
  - "Do not use old `strategy_ui.js` as long-term dependency."
  - "The page body after the global header must be reference-shaped against strategy_statistic.png."
  - "Current storage may not cover all reference stats; add bounded read-models/projections or typed degraded panels with explicit data-source inventory."
  - "Manual refresh/autorefresh must use shared no-overlap helpers and backend `retry_after_seconds`; browser must not call exchanges directly."
---

# Task

Implement Stage 6 `/strategies` selected-strategy dashboard.

Done means:

- `/strategies` shows a selected-strategy analytics workstation shaped by `strategy_statistic.png`;
- `/strategies/new` works as compatibility redirect/alias to create mode inside `/strategies`;
- `/strategies/{strategy_id}` is a deep-link/redirect to the same `/strategies` workstation state, not an equivalent separate page body;
- backend exposes owner-scoped bounded UI read-model where required;
- data-source inventory/freshness/degraded states are explicit for every live/stat panel;
- manual refresh/autorefresh controls exist where live data is shown;
- create/list/clone/delete use existing Strategy API;
- old `strategy_ui.js` dependency is removed from these pages;
- Playwright evidence exists.

## Context / Current State

- Current Strategy API already supports immutable create/list/get/clone/delete/run/stop.
- Strategy model is immutable: edit means clone/create, not update.
- `/strategies` owns the concrete strategy dashboard reference. `/monitoring` is compatibility-only in the v1 map.

## Requirements (Must)

- Open `/Users/daniildegtyarev/Projects/roehub_web_ui/strategy_statistic.png` before coding.
- List the reference panel inventory before implementation notes/final report.
- Implement `/api/ui/strategies/dashboard?strategy_id=&state=active|all&cursor=` or a compatible bounded read-model required by the reference.
- Include `sources[]`, `generated_at`, `refresh_status`, `next_allowed_refresh_at`/`retry_after_seconds` or equivalent in live/stat DTOs.
- Use branded selector/filter/autorefresh dropdowns; visible native select is not acceptable.
- Preserve expected panels from the plan: command/status bars, top summary/strategy info, chart, metric grid, monthly stats, drawdown/equity, best/worst days, hourly results, trades/events table, symbol results/breakdowns.
- Add persistent read-model/projection requirements when current migrations cannot support a reference panel; otherwise keep the panel as typed `degraded/unavailable/stale`.
- Preserve public Strategy API semantics.
- Preserve create workflow.
- Do not fake unavailable strategy statistics.
- Keep financial color semantics if financial values appear.
- Add focused web tests.
- Use `publish-ci-deploy` only after all gates pass.

## Requirements (Should)

- Use shared design system components.
- Keep page JS modular and scoped.
- Prefer existing API helpers from JS core.

## Requirements (Nice-to-have)

- Optional compact read-model endpoint only if existing API causes obvious UX/performance issue.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 6, design manifest strategy sections, `strategy_statistic.png`, then task entrypoints. Expand only for API semantics or failing tests.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Open `strategy_statistic.png` and record panel inventory.
2. Record panel inventory and route/deep-link decisions; keep visual pages limited to `/strategies`.
3. Define/implement bounded UI read-model route/wiring if needed by the reference panels.
4. Implement `/strategies` page/fragments/JS/CSS as selected-strategy workstation.
5. Preserve existing API calls and payload shape for create/clone/delete/run/stop.
6. Add/adjust tests.
7. Run browser QA and quality gates.
8. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- Implemented panel inventory matches `strategy_statistic.png`; any deviation is named and justified.
- List, clone, run/stop and soft-delete continue to call existing `/api/strategies*` routes or documented UI wrappers.
- Create/clone preserve canonical indicator payload shape.
- `/strategies/new` is route-tested and browser-checked as compatibility redirect/alias to create mode inside `/strategies`.
- `/strategies/{strategy_id}` deep link behavior is route-tested as selected state inside `/strategies`.
- No dependency on old `strategy_ui.js`.
- Manual refresh/autorefresh works without overlapping requests and respects backend retry windows.
- Open branded selector/filter dropdown is captured in browser evidence.
- Generic strategy card-grid/library layout is not acceptable.

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

- Public API contract: `compatible-change` if `/api/ui/strategies/dashboard*` is added.
- Browser-visible behavior: intentional `breaking-change`.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Monitoring/SSE.
- Separate primary `/monitoring` page.
- New mutable strategy update API.
- Backtest integration.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/web/test_app_routes.py
uv run pytest -q tests/unit/apps/api/test_ui_strategy_dashboard_routes.py
uv run pytest -q tests/unit/apps/api/test_strategies_routes.py
uv run ruff check apps/web apps/api tests/unit/apps/web tests/unit/apps/api
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/strategies
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/strategies-desktop.png
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
