---
prompt_name: web_ui_backend_v1_02_design_system_js_core
repo: roehub.com
branch: current
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

hard_requirements:
  default_theme_terminal_orange: true
  theme_switching_required: true
  financial_colors_invariant: true
  no_page_specific_color_literals: true
  js_core_required: true
  no_overlap_poller_required: true
  api_client_error_mapping_required: true
  browser_qa_required: true

task_toggles:
  implement_css_tokens: true
  implement_theme_switcher: true
  implement_shared_components: true
  implement_js_core: true
  implement_backend_preferences_api: false
  publish_after_success: true

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

non_goals:
  - "Do not implement page-specific business workflows."
  - "Do not implement backend account preferences yet."
  - "Do not add Node/Vite/React toolchain."
  - "Do not recolor financial semantics by theme."

final_report_format:
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

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
- `theme.js`, `api.js`, `poller.js`, `sse.js`, `notifications.js`, `formatters.js`, `validators.js` exist as scoped JS core;
- theme switch updates `data-theme` without reload and preserves financial colors;
- `poller.js` prevents overlapping requests and hidden-tab repeated polling;
- browser evidence exists.

## Context / Current State

- Stage 1 shell should be in place or this task must integrate with the current base shell.
- Current `site.css`, `strategy_ui.js`, `backtest_ui.js` are replacement targets, not visual sources.
- Design manifest fixes `terminal-orange` as default and requires theme switching.

## Requirements (Must)

- Implement tokens and themes through CSS variables.
- Never hardcode page color literals outside token/theme layer except narrowly justified non-color values.
- Keep `financial` tokens invariant across themes.
- Implement JS core with deterministic 401/403/409/422/timeout handling.
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

# Reading manifest

Use front matter `context_sources`; avoid broad repo reads.

# Work plan (agent should follow)

1. Inspect current shell asset hooks.
2. Add CSS token/theme/component files.
3. Add shared Jinja macros/components.
4. Add JS core modules.
5. Wire shell placeholders to use the new assets.
6. Add focused tests.
7. Run browser QA and quality gates.
8. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- Shared components render without page-specific CSS.
- New templates do not depend on old `site.css`.
- Default theme is `terminal-orange`.
- Theme switch updates `data-theme` immediately.
- Financial values keep fixed semantic colors across all themes.
- `api.js` handles 401, 403, 409, 422, timeout deterministically.
- `poller.js` has no-overlap behavior and hidden-tab pause.
- Components have accessible labels/focus states.

# Implementation constraints

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

# Final output: report format (strict)

Report in Russian:

- `Intent`
- `Scope`
- `Design`
- `Contract impact`
- `Tests`
- `Runtime evidence`
- `Risks`
- `Handoff`
- `Publish/deploy`
