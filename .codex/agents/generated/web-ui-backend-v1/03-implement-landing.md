---
prompt_name: web_ui_backend_v1_03_landing
repo: roehub.com
branch: current
scope: "Этап 3: публичный лендинг Roehub по design manifest и general_page.png."

language:
  implementation: jinja_css_plain_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and browser verification rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 3 source of truth"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "landing visual rules and default palette"
  task_entrypoints:
    - path: apps/web/main/app.py
      why: "landing route and template context"
    - path: apps/web/templates/pages/landing.html
      why: "target landing page if Stage 1 moved templates"
    - path: apps/web/templates/landing.html
      why: "legacy landing location if still present"
    - path: apps/web/dist/css/pages
      why: "page CSS target"
  conditional_bundles:
    reference_assets:
      read_when: "when comparing with supplied visual reference"
      paths:
        - /Users/daniildegtyarev/Projects/roehub_web_ui/general_page.png
    auth_header:
      read_when: "if auth/register CTA behavior is ambiguous"
      paths:
        - docs/architecture/identity/identity-keycloak-auth-model-v1.md
  consult_if_needed:
    - path: docs/web-ui+backend-plan-deep-research.md
      read_when: "if landing scope or visual intent is unclear"

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
  anonymous_landing_required: true
  no_api_dependency_for_first_render: true
  product_diagram_required: true
  no_stock_orb_gradient_backgrounds: true
  mobile_no_horizontal_overflow: true
  browser_qa_required: true

task_toggles:
  implement_landing_template: true
  implement_landing_css: true
  implement_landing_js_optional: true
  implement_backend_api: false
  publish_after_success: true

package_contract:
  depends_on:
    - "01-shell-auth-register accepted"
    - "02-design-system-js-core accepted"
  owns:
    - "apps/web/templates/pages/landing.html"
    - "apps/web/dist/css/pages/landing.css"
    - "apps/web/dist/js/pages/landing.js"
    - "tests/unit/apps/web/test_app_routes.py landing assertions"
  forbidden:
    - "protected page feature implementations"
    - "apps/api/**"
    - "src/trading/contexts/**"
  integration_points:
    - "public / route"
    - "header CTA routes"
    - "theme switcher visibility"
  handoff:
    - "public landing route and CTA behavior for final QA"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing public route behavior, auth/register CTA, browser-visible defaults, or asset paths"
    timing: "before final report"
    reason: "landing is browser-visible and public"
  - skill: browser-qa-evidence
    use_when: "verifying desktop/mobile layout, console/network, overflow, CTA routing, theme behavior"
    timing: "after implementation"
    reason: "landing acceptance is visual/runtime"
  - skill: playwright
    use_when: "capturing Playwright snapshot/screenshots"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI evidence"
  - skill: backend-quality-gates
    use_when: "running web tests/lint"
    timing: "during verification"
    reason: "template route and assets must remain green"
  - skill: publish-ci-deploy
    use_when: "all landing acceptance, tests, and Playwright evidence pass"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "/"
  - "ROEHUB"
  - "terminal-orange"

non_goals:
  - "Do not add backend API dependency for anonymous landing render."
  - "Do not build a SPA or marketing-only generic page."
  - "Do not use decorative orbs/gradients as primary visual."

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
  - "Publish/deploy: publish-ci-deploy terminal state; if successful, include main merge, local sync, Mac Studio git pull, impacted service restart/reload, and smoke verification evidence; otherwise exact reason it was skipped"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/web tests/unit/apps/web"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/web/templates/pages/landing.html"
  - "apps/web/dist/css/pages/landing.css"
  - "apps/web/dist/js/pages/landing.js"
  - "tests/unit/apps/web/test_app_routes.py"

possible_secondary_touches:
  - "apps/web/templates/base.html"
  - "apps/web/main/app.py"

safety_notes:
  - "Landing must work anonymously and without backend API availability."
  - "Hero H1 should carry Roehub/product identity, not generic marketing copy."
---

# Task

Implement Stage 3 public landing page.

Done means:

- `/` renders anonymously;
- first viewport clearly signals Roehub/product and CTA;
- visual style follows design manifest and `general_page.png`;
- no backend API is required for anonymous first render;
- mobile has no horizontal overflow;
- Playwright evidence exists.

## Context / Current State

- Stage 1/2 should provide shell, tokens, themes, shared components.
- The old light landing is not a visual baseline.
- Product brand is `Roehub`; `QUANT CLI` in references is style only.

## Requirements (Must)

- Implement `pages/landing.html` and scoped CSS.
- Use a product/platform diagram or UI-native visual, not stock imagery.
- Keep header auth/register actions correct.
- Ensure theme switcher works if visible in shell.
- Run web route tests and browser QA.
- Use `publish-ci-deploy` only after all gates pass and task is complete.

## Requirements (Should)

- Keep JS optional and non-blocking.
- Show a hint of next section content in first viewport.
- Keep landing usable without JS.

## Requirements (Nice-to-have)

- Add a small status/product capability band if it fits the reference style.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 3, design manifest landing sections, then task entrypoints. Stop once route/template/CSS scope is clear.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Inspect existing landing route/template.
2. Implement landing template and scoped CSS.
3. Wire asset references.
4. Add/adjust route smoke tests.
5. Run local server and Playwright desktop/mobile QA.
6. Run quality gates.
7. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- `/` loads anonymously without API dependency.
- Auth/register CTAs route correctly.
- Desktop and mobile screenshots show no overflow or broken text.
- No failed same-origin network requests except expected optional auth checks.
- No console errors.

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
- Browser-visible behavior: intentional replacement.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Dashboard or account functionality.
- Backend endpoints.
- New frontend build toolchain.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/web/test_app_routes.py
uv run ruff check apps/web tests/unit/apps/web
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/landing-desktop.png
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

# publish-ci-deploy terminal delivery contract

When all stage DoD, gates, browser evidence, and performance evidence required by this prompt pass, and `publish_after_success` is true, run `publish-ci-deploy` to the natural terminal state. A successful terminal state for this prompt means more than PR creation, green CI, or deploy workflow completion. It must include, when the agent has authority and no external blocker remains:

- branch/PR merged into `main`, or exact blocker why merge is outside current authority;
- local checkout synchronized with `origin/main`;
- Mac Studio repository checkout synchronized with `origin/main` using `git pull --ff-only` from the actual repo checkout, normally `/Users/daniildegtyarev/Projects/roehub.com`;
- deployed runtime updated through the repository deploy/runbook path, keeping the repo checkout and runtime bundle as separate surfaces when they differ;
- impacted services restarted only when touched-path impact requires it; if impact is unclear, use the standard prod reload path from `publish-ci-deploy`;
- post-restart smoke verification completed;
- final report names exact commands, host/paths used, commit SHA on `main`, restarted services, smoke result, or exact blocker.

Do not report successful publish/deploy while merge to `main`, Mac Studio git pull, required service restart/reload, or smoke verification remains pending.

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
