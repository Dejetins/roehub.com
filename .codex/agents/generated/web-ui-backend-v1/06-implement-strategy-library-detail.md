---
prompt_name: web_ui_backend_v1_06_strategy_library_detail
repo: roehub.com
branch: current
scope: "Этап 6: новая библиотека стратегий, create flow и детали стратегии поверх существующего Strategy API."

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
      why: "strategy library visual rules"
  task_entrypoints:
    - path: apps/web/main/app.py
      why: "strategies routes and /strategies/new"
    - path: apps/web/templates
      why: "current strategy list/create/detail templates replacement targets"
    - path: apps/web/dist/strategy_ui.js
      why: "old strategy JS replacement target and API usage reference"
    - path: apps/api/routes/strategies.py
      why: "existing Strategy API contract"
  conditional_bundles:
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
    purpose: "reference screenshots/assets; inspect only stage-relevant pages"
  default_palette: terminal-orange
  theme_variants:
    - terminal-orange
    - graphite
    - matrix-green
    - high-contrast
  invariant_financial_colors: true

hard_requirements:
  preserve_strategy_api: true
  preserve_immutable_strategy_model: true
  support_strategies_new: true
  no_backend_api_required_initially: true
  remove_old_strategy_ui_dependency: true
  browser_qa_required: true

task_toggles:
  implement_strategy_list: true
  implement_strategy_create: true
  implement_strategy_detail: true
  implement_new_backend_read_model: false
  publish_after_success: true

package_contract:
  depends_on:
    - "01-shell-auth-register accepted"
    - "02-design-system-js-core accepted"
  owns:
    - "apps/web/templates/pages/strategies.html"
    - "apps/web/templates/pages/strategy_create.html"
    - "apps/web/templates/pages/strategy_detail.html"
    - "apps/web/templates/fragments/strategies/**"
    - "apps/web/dist/js/pages/strategies.js"
    - "apps/web/dist/css/pages/strategies.css"
    - "tests/unit/apps/web/test_app_routes.py strategy assertions"
    - "optional apps/api/routes/ui_strategies.py only if read-model is justified"
  forbidden:
    - "monitoring/SSE package files"
    - "backtest package files"
    - "settings/account package files"
    - "mutable Strategy API semantics"
  integration_points:
    - "existing /api/strategies* routes"
    - "strategy create/clone payload shape"
    - "shared JS api.js"
  handoff:
    - "strategy list/create/detail routes without old strategy_ui.js dependency"

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
    use_when: "verifying list/create/detail, route `/strategies/new`, console/network, responsive layout"
    timing: "after local tests"
    reason: "strategy library is browser-visible"
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
  - "/api/strategies"
  - "/api/strategies/clone"
  - "strategy_ui.js"

non_goals:
  - "Do not implement live monitoring on strategy detail; monitoring belongs to `/monitoring`."
  - "Do not make strategy mutable update endpoint."
  - "Do not fake metrics that backend does not provide."

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
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py"
    expect: "passes; add strategy UI asset tests if needed"
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
  - "apps/web/templates/pages/strategy_create.html"
  - "apps/web/templates/pages/strategy_detail.html"
  - "apps/web/templates/fragments/strategies/*"
  - "apps/web/dist/js/pages/strategies.js"
  - "apps/web/dist/css/pages/strategies.css"
  - "tests/unit/apps/web/test_app_routes.py"

possible_secondary_touches:
  - "apps/web/main/app.py"
  - "apps/api/routes/ui_strategies.py"
  - "apps/api/dto/ui_strategies.py"

safety_notes:
  - "Current `/strategies/new` must not disappear."
  - "Create/clone must preserve canonical indicator payload shape."
  - "Do not use old `strategy_ui.js` as long-term dependency."
---

# Task

Implement Stage 6 strategy library/create/detail UI.

Done means:

- `/strategies` shows new strategy library;
- `/strategies/new` works as create entrypoint or controlled redirect to create modal;
- `/strategies/{strategy_id}` shows new detail page;
- create/list/clone/delete use existing Strategy API;
- old `strategy_ui.js` dependency is removed from these pages;
- Playwright evidence exists.

## Context / Current State

- Current Strategy API already supports immutable create/list/get/clone/delete/run/stop.
- Strategy model is immutable: edit means clone/create, not update.
- Live monitoring belongs to Stage 7 `/monitoring`.

## Requirements (Must)

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

Read `.codex/AGENTS.md`, plan Stage 6, design manifest strategy sections, then task entrypoints. Expand only for API semantics or failing tests.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Inspect current strategy routes/templates/JS.
2. Implement new list/create/detail templates and page JS/CSS.
3. Preserve existing API calls and payload shape.
4. Add/adjust tests.
5. Run browser QA and quality gates.
6. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- List, clone and soft-delete continue to call existing `/api/strategies*` routes.
- Create/clone preserve canonical indicator payload shape.
- `/strategies/new` is route-tested and browser-checked.
- Strategy detail does not imply live monitoring.
- No dependency on old `strategy_ui.js`.

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

- Public API contract: `none` for initial replacement.
- Browser-visible behavior: intentional `breaking-change`.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Monitoring/SSE.
- New mutable strategy update API.
- Backtest integration.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/web/test_app_routes.py
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
