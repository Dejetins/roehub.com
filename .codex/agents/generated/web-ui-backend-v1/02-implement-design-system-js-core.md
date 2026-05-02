---
prompt_name: web_ui_backend_v1_02_design_system_js_core
repo: roehub.com
branch: main
scope: "Этап 2: design tokens, themes, shared Jinja components/macros и JS core."

language:
  implementation: jinja_css_plain_js_python_tests
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and browser verification requirements"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 2 source of truth"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "CSS tokens, default palette, theme invariants, components"
  task_entrypoints:
    - path: apps/web/templates/base.html
      why: "theme switcher hook and shared components integration"
    - path: apps/web/dist
      why: "current assets to replace/split"
    - path: tests/unit/apps/web/test_app_routes.py
      why: "asset and route smoke"
  conditional_bundles:
    shell_stage:
      read_when: "if Stage 1 implementation is present and must be integrated"
      paths:
        - apps/web/main/app.py
        - apps/web/templates/pages/dashboard.html
    browser_qa:
      read_when: "when collecting Playwright evidence"
      paths:
        - docs/architecture/apps/web/web-ui-design-manifest-v1.md
  consult_if_needed:
    - path: apps/web/dist/site.css
      read_when: "only to ensure old CSS dependency is removed, not as visual basis"

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
  default_theme_terminal_orange: true
  theme_switching_required: true
  financial_colors_invariant: true
  no_page_specific_color_literals: true
  js_core_required: true
  no_overlap_poller_required: true
  api_client_error_mapping_required: true
  locale_js_required: true
  i18n_component_contract_required: true
  browser_qa_required: true

task_toggles:
  implement_css_tokens: true
  implement_theme_switcher: true
  implement_shared_components: true
  implement_js_core: true
  implement_backend_preferences_api: false
  publish_after_success: true

package_contract:
  depends_on:
    - "00-contract-freeze-and-cleanup-boundary accepted"
    - "01-shell-auth-register accepted or current shell bounded"
  owns:
    - "apps/web/templates/macros/**"
    - "apps/web/templates/components/**"
    - "apps/web/dist/css/tokens.css"
    - "apps/web/dist/css/themes.css"
    - "apps/web/dist/css/base.css"
    - "apps/web/dist/css/layout.css"
    - "apps/web/dist/css/components.css"
    - "apps/web/dist/js/core/**"
    - "apps/web/dist/js/components/**"
    - "tests/unit/apps/web/** asset/theme smoke tests"
  forbidden:
    - "page-specific product workflows"
    - "apps/api/**"
    - "src/trading/contexts/**"
  integration_points:
    - "CSS token names"
    - "theme.js data-theme API"
    - "api.js/poller.js/sse.js/dom.js public JS helpers"
    - "locale.js language switcher/helper API"
    - "shared macro/component names"
  handoff:
    - "stable UI kit, theme contract, and JS core for page packages"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing browser-visible defaults, theme persistence semantics, asset paths, or JS API client behavior"
    timing: "before implementation and final report"
    reason: "themes and JS core define browser-visible defaults and error behavior"
  - skill: browser-qa-evidence
    use_when: "verifying theme switcher, no financial color drift, console/network cleanliness, desktop/mobile layout"
    timing: "after local tests"
    reason: "design system is browser-visible"
  - skill: playwright
    use_when: "capturing snapshots/screenshots"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI evidence"
  - skill: backend-quality-gates
    use_when: "running web tests/lint/type checks"
    timing: "during verification"
    reason: "asset references and Python web tests must stay green"
  - skill: publish-ci-deploy
    use_when: "all tests, Playwright evidence, and theme acceptance pass with scoped diff"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "terminal-orange"
  - "graphite"
  - "matrix-green"
  - "high-contrast"
  - "data-theme"
  - "--rh-financial-positive"
  - "--rh-financial-negative"
  - "api.js"
  - "poller.js"
  - "sse.js"
  - "dom.js"
  - "locale.js"
  - "en"
  - "ru"
  - "data-locale"

non_goals:
  - "Do not implement page-specific business workflows."
  - "Do not implement backend account preferences yet."
  - "Do not add Node/Vite/React toolchain."
  - "Do not recolor financial semantics by theme."
  - "Do not hardcode user-visible component labels in only one language."

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
  - cmd: "uv run pytest -q tests/unit/apps/web"
    expect: "passes"
  - cmd: "uv run ruff check apps/web tests/unit/apps/web"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes if Python touched"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/web/templates/macros/ui.html"
  - "apps/web/templates/components/*.html"
  - "apps/web/dist/css/tokens.css"
  - "apps/web/dist/css/themes.css"
  - "apps/web/dist/css/base.css"
  - "apps/web/dist/css/layout.css"
  - "apps/web/dist/css/components.css"
  - "apps/web/dist/js/core/*.js"
  - "apps/web/dist/js/components/*.js"
  - "tests/unit/apps/web"

possible_secondary_touches:
  - "apps/web/templates/base.html"
  - "apps/web/templates/pages/dashboard.html"
  - "docs/architecture/apps/web/web-ui-design-manifest-v1.md"

safety_notes:
  - "Use CSS variables for UI colors."
  - "Financial colors are semantic invariants and must not be theme-overridden."
  - "Mutation requests need one CSRF extension point in JS core, not ad hoc page code."
---

# Task

Implement Stage 2 design system, theme switching, shared components, and JS core.

Done means:

- CSS tokens and themes exist with `terminal-orange` default;
- shared Jinja macros/components exist and can render placeholders;
- `theme.js`, `api.js`, `poller.js`, `sse.js`, `dom.js`, `notifications.js`, `formatters.js`, `validators.js` exist as scoped JS core;
- `locale.js` exists as scoped JS core for language switcher state, cookie/localStorage sync, fallback `en`, and dynamic strings;
- theme switch updates `data-theme` without reload and preserves financial colors;
- shared components expose i18n hooks/keys for labels, empty states, errors and button text;
- `poller.js` prevents overlapping requests and hidden-tab repeated polling;
- browser evidence exists.

## Context / Current State

- Stage 1 shell should be in place or this task must integrate with the current base shell.
- Current `site.css`, `strategy_ui.js`, `backtest_ui.js` are replacement targets, not visual sources.
- Design manifest fixes `terminal-orange` as default and requires theme switching.
- Design manifest fixes `en` as default locale, `ru` as secondary locale, and requires language switching.

## Requirements (Must)

- Implement tokens and themes through CSS variables.
- Never hardcode page color literals outside token/theme layer except narrowly justified non-color values.
- Keep `financial` tokens invariant across themes.
- Implement JS core with deterministic 401/403/409/422/timeout handling.
- Implement `locale.js` with `en`/`ru` support, safe fallback, and no localized route/API paths.
- Provide CSRF extension point for state-changing calls.
- Add tests/smokes for asset references and theme hooks.
- Run Playwright evidence.
- If and only if all checks pass and DoD is complete, run `publish-ci-deploy`.

## Requirements (Should)

- Prefer small plain ES modules with JSDoc where helpful.
- Keep components compact and reusable.
- Use icon delivery only if it is self-hosted and scoped; otherwise use text controls.

## Requirements (Nice-to-have)

- Add a visual QA fixture route only if it stays clearly internal/dev and tested.

# Context acquisition protocol

Read `.codex/AGENTS.md`, the plan Stage 2, design manifest token/component sections, then task entrypoints. Stop when files and contracts are bounded.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`; avoid broad repo reads.

# Work plan (agent should follow)

1. Inspect current shell asset hooks.
2. Add CSS token/theme/component files.
3. Add shared Jinja macros/components.
4. Add JS core modules, including the shared `dom.js` and `locale.js` helpers from the target asset structure.
5. Wire shell placeholders to use the new assets.
6. Add focused tests.
7. Run browser QA and quality gates.
8. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- Shared components render without page-specific CSS.
- New templates do not depend on old `site.css`.
- Default theme is `terminal-orange`.
- Theme switch updates `data-theme` immediately.
- Default locale is `en`; language switcher can select `ru`; root `data-locale` and `<html lang>` stay aligned.
- Shared components render `en` and `ru` labels without missing keys.
- Financial values keep fixed semantic colors across all themes.
- `api.js` handles 401, 403, 409, 422, timeout deterministically.
- `poller.js` has no-overlap behavior and hidden-tab pause.
- Components have accessible labels/focus states.

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

- Public API contract: `none`.
- Browser-visible behavior: intentional `breaking-change`.
- Config schema: `compatible-change` only if theme defaults become server config.

## Browser-visible behavior

- Provide desktop and mobile evidence if layout/theme changes.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Backend preferences persistence.
- Account settings page.
- Full page implementations.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/web
uv run ruff check apps/web tests/unit/apps/web
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/dashboard
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/ui-kit-dashboard-placeholder.png
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
