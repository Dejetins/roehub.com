---
prompt_name: web_ui_backend_v1_01_shell_auth_register
repo: roehub.com
branch: current
scope: "Этап 1: новый app shell, header tabs, login/logout/register entrypoints и protected route gate."

language:
  implementation: python_jinja_css_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract and browser verification rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 1 source of truth"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "shell/header/theme style contract"
  task_entrypoints:
    - path: apps/web/main/app.py
      why: "web app routes, protected page helper, /api proxy"
      inspect_symbols:
        - create_app
        - _register_routes
        - _render_protected_page
    - path: apps/web/templates/base.html
      why: "shared layout to replace"
    - path: apps/web/templates/login.html
      why: "current login entrypoint and inline script removal target"
    - path: apps/web/templates/logout.html
      why: "current logout flow and inline script removal target"
    - path: tests/unit/apps/web/test_app_routes.py
      why: "route smoke and protected redirect tests"
  conditional_bundles:
    auth_contract:
      read_when: "when register/login/logout behavior is ambiguous"
      paths:
        - docs/architecture/identity/identity-keycloak-auth-model-v1.md
        - src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py
    edge_contract:
      read_when: "when changing /api proxy or same-origin behavior"
      paths:
        - docs/runbooks/web-ui-gateway-same-origin.md
        - infra/caddy/Caddyfile.vps
  consult_if_needed:
    - path: tests/unit/apps/web/test_security.py
      read_when: "if adding or modifying security tests"

hard_requirements:
  replace_shell: true
  protected_routes_login_gate: true
  register_entrypoint_keycloak_backed: true
  no_local_username_password_registration: true
  no_external_cdn_script: true
  no_inline_auth_scripts: true
  support_strategies_new_entrypoint: true
  browser_qa_required: true

task_toggles:
  implement_routes: true
  implement_base_template: true
  implement_placeholder_pages: true
  implement_auth_js_or_server_redirects: true
  implement_backend_auth_changes: false
  publish_after_success: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing auth/register route behavior, /api proxy behavior, next redirect, asset/CSP defaults, or browser-visible defaults"
    timing: "before implementation and before final report"
    reason: "auth and browser-visible routing are contract surfaces"
  - skill: backend-quality-gates
    use_when: "running web route/security tests, ruff, pyright"
    timing: "during verification"
    reason: "stage touches Python FastAPI web routes"
  - skill: browser-qa-evidence
    use_when: "verifying landing, login redirect, protected redirects, header tabs, register CTA, console/network"
    timing: "after local tests pass"
    reason: "shell/auth behavior is browser-visible"
  - skill: playwright
    use_when: "capturing screenshots/snapshots through Playwright CLI"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI evidence"
  - skill: publish-ci-deploy
    use_when: "all DoD criteria pass, Playwright evidence exists, and diff is limited to this stage"
    timing: "after verification"
    reason: "user requires full Roehub delivery chain after 100% completion"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "/"
  - "/login"
  - "/logout"
  - "/register"
  - "/dashboard"
  - "/settings"
  - "/strategies"
  - "/strategies/new"
  - "/monitoring"
  - "/backtests"
  - "/backtests/new"
  - "/backtests/{job_id}"
  - "/api/auth/current-user"
  - "terminal-orange"

non_goals:
  - "Do not implement real dashboard/settings/monitoring/backtest page functionality beyond placeholders."
  - "Do not create local username/password registration."
  - "Do not migrate to React/Next/SPA."
  - "Do not change backend auth semantics unless Keycloak registration entrypoint is explicitly required."

final_report_format:
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/web/test_security.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/web tests/unit/apps/web"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes if Python route/settings/API client types changed"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/web/main/app.py"
  - "apps/web/templates/base.html"
  - "apps/web/templates/login.html"
  - "apps/web/templates/logout.html"
  - "apps/web/templates/pages/*.html"
  - "apps/web/templates/partials/user_badge.html"
  - "apps/web/dist/css/tokens.css"
  - "apps/web/dist/css/themes.css"
  - "apps/web/dist/css/base.css"
  - "apps/web/dist/css/layout.css"
  - "apps/web/dist/js/pages/auth.js"
  - "tests/unit/apps/web/test_app_routes.py"
  - "tests/unit/apps/web/test_security.py"

possible_secondary_touches:
  - "apps/api/routes/identity.py"
  - "apps/api/wiring/modules/identity.py"
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"

safety_notes:
  - "Server-side auth state comes only from `/api/auth/current-user`."
  - "Sanitize external `next=https://...` to safe local path."
  - "Do not add second `/api` prefix inside backend routes."
  - "If registration requires a Keycloak realm/client choice that is not discoverable, stop and ask."
---

# Task

Implement Stage 1 shell/auth/register foundation for Roehub Web UI v1.

Done means:

- new terminal-style base shell and header tabs are in place;
- public `/`, `/login`, `/logout`, `/register` routes exist;
- protected placeholder routes exist for dashboard/settings/strategies/monitoring/backtests;
- protected routes redirect anonymous users to `/login?next=<safe-local-path>`;
- auth/logout no longer require inline scripts;
- HTMX is self-hosted or no external CDN remains in the shell;
- `/strategies/new` remains a supported create entrypoint or controlled redirect;
- Playwright evidence exists.

## Context / Current State

- Current `apps/web` is FastAPI SSR/Jinja2 with `/assets` and local `/api/*` proxy.
- Current `base.html` uses HTMX from CDN.
- Current login/logout templates use inline JavaScript.
- New design uses terminal-orange default palette and dark shell.

## Requirements (Must)

- Implement only shell/auth/register and placeholders.
- Preserve production same-origin split: browser `/api/*`, backend routes without `/api` prefix.
- Keep web stateless: no domain use-case imports in `apps/web`.
- Add/adjust tests for route map, protected redirects, `next` sanitization, asset references.
- Run browser QA through Playwright CLI.
- If all checks pass and the task is 100% complete, use `publish-ci-deploy`. If any gate fails, do not publish.

## Requirements (Should)

- Prefer server-side redirects over JS where practical.
- Keep templates organized under `pages/`, `components/`, `macros/` as the plan defines.
- Keep text in Russian where appropriate, technical route/API identifiers unchanged.

## Requirements (Nice-to-have)

- Add minimal active nav state tests.
- Add a simple user badge placeholder compatible with later account work.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. implementation plan Stage 1 and design manifest shell/theme sections
3. task entrypoints
4. conditional auth/edge docs only when needed

Pre-implementation reading target: `<= 8 files`, `<= ~45k tokens`.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.

# Reading manifest

Use front matter `context_sources`. Do not preload all conditional bundles.

# Work plan (agent should follow)

1. Inspect current route/template/test shape.
2. Implement route map and protected placeholder pages.
3. Replace shared shell/header and auth/register entrypoints.
4. Remove external CDN/inline auth script dependency.
5. Add focused tests.
6. Run local web server and Playwright CLI evidence.
7. Run quality gates.
8. If and only if all DoD criteria pass, run `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- Anonymous `/` returns 200.
- Protected pages redirect to `/login?next=<safe-local-path>`.
- External `next` target is sanitized.
- Header tabs render and active state is deterministic.
- Login/logout do not require inline scripts.
- Register CTA leads to selected Keycloak-backed entrypoint or a documented compatible auth extension.
- No external CDN script remains in base shell.
- Playwright snapshot and desktop screenshot exist for `/` and protected redirect behavior.

# Implementation constraints

## API / contracts

- Public API contract is `compatible-change` only if auth registration endpoint is added.
- Browser-visible behavior is intentionally `breaking-change`.
- Config schema changes are `compatible-change` and must be documented.

## Browser-visible behavior

- Browser claims require Playwright evidence.
- Check desktop and at least one protected route redirect.

# Files to indicate (expected touched areas)

Use front matter `expected_primary_touches` and `possible_secondary_touches`.

# Non-goals

- Real page data.
- Settings/account persistence.
- Monitoring/SSE.
- Backtest create/results.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/web/test_security.py
uv run ruff check apps/web tests/unit/apps/web
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/shell-landing-desktop.png
"$PWCLI" open http://127.0.0.1:8010/settings
"$PWCLI" snapshot
```

# Final output: report format (strict)

Report in Russian with:

- `Intent`
- `Scope`
- `Design`
- `Contract impact`
- `Tests`
- `Runtime evidence`
- `Risks`
- `Handoff`
- `Publish/deploy`: state whether `publish-ci-deploy` ran and terminal state.
