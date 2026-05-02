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
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

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

# Final output: report format (strict)

Report in Russian: `Intent`, `Scope`, `Design`, `Contract impact`, `Tests`, `Runtime evidence`, `Risks`, `Handoff`, `Publish/deploy`.
