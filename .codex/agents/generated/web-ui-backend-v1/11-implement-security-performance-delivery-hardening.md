---
prompt_name: web_ui_backend_v1_11_security_performance_delivery_hardening
repo: roehub.com
branch: current
scope: "Этап 11: финальный security/performance/browser/delivery hardening Web UI v1."

language:
  implementation: python_fastapi_jinja_css_js_config_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "pre-ship, security, browser, gates, delivery invariants"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 11 source of truth"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "final visual/browser acceptance"
  task_entrypoints:
    - path: apps/web/main/app.py
      why: "web routes, proxy, cache/security headers"
    - path: apps/api/main/app.py
      why: "API middleware/routes"
    - path: infra/caddy/Caddyfile.vps
      why: "production same-origin edge split and SSE buffering concerns"
    - path: docs/runbooks/web-ui-gateway-same-origin.md
      why: "local/prod same-origin contract"
  conditional_bundles:
    csrf_csp:
      read_when: "when implementing CSRF/CSP/cache headers"
      paths:
        - docs/architecture/identity/identity-keycloak-auth-model-v1.md
        - apps/web/main/security.py
        - src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py
    page_packages:
      read_when: "when final sweep finds page-specific failures"
      paths:
        - apps/web/templates/pages
        - apps/web/dist/js/pages
        - apps/web/dist/css/pages
    deploy_workflows:
      read_when: "when validating delivery/deploy routes"
      paths:
        - .github/workflows/deploy-web.yml
        - infra/docker/docker-compose.web.prod.yml
  consult_if_needed:
    - path: docs/runbooks/mac-studio-native-backend-operations.md
      read_when: "if backend/API service impact is ambiguous"

hard_requirements:
  csrf_strategy_required: true
  csp_no_external_cdn_inline_scripts: true
  protected_html_no_store: true
  versioned_assets_cache_headers: true
  edge_split_verified: true
  sse_buffering_checked_if_sse_exists: true
  full_browser_sweep_required: true
  full_backend_gates_required: true

task_toggles:
  implement_csrf: true
  implement_cache_headers: true
  implement_csp: true
  implement_asset_versioning: true
  implement_final_browser_sweep: true
  publish_after_success: true

skill_routing:
  - skill: production-risk-review
    use_when: "before final publish to inspect diff for security, contracts, data safety, deployment risk"
    timing: "before publish"
    reason: "hardening stage is release-risk sensitive"
  - skill: contract-impact-analysis
    use_when: "changing CSRF/CSP/cache/config/asset versioning, browser defaults, edge routing, or API behavior"
    timing: "before implementation and final report"
    reason: "security/config/browser defaults are contracts"
  - skill: backend-quality-gates
    use_when: "running full ruff/pyright/pytest/docs gates"
    timing: "during verification"
    reason: "final hardening must be fully gated"
  - skill: browser-qa-evidence
    use_when: "final desktop/mobile/console/network/theme sweep"
    timing: "after local gates"
    reason: "browser-visible app release must be runtime verified"
  - skill: playwright
    use_when: "capturing final Playwright screenshots/snapshots"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI sweep"
  - skill: backend-performance-evidence
    use_when: "running performance smoke or claiming 1 vCPU/2GB suitability"
    timing: "during performance verification"
    reason: "performance claims need evidence"
  - skill: publish-ci-deploy
    use_when: "all full gates, browser sweep, security checks, and docs index pass"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions
  - macstudio

required_literals:
  - "CSRF"
  - "CSP"
  - "Cache-Control: no-store"
  - "script-src 'self'"
  - "connect-src 'self'"
  - "/api/*"
  - "terminal-orange"

non_goals:
  - "Do not implement unfinished product page features."
  - "Do not relax security because local flow is easier."
  - "Do not deploy if any full gate or browser sweep fails."

final_report_format:
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright/performance evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

quality_gates:
  - cmd: "uv run ruff check ."
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - "apps/web/main/app.py"
  - "apps/web/main/security.py"
  - "apps/web/templates/base.html"
  - "apps/web/dist/**"
  - "apps/api/**"
  - "infra/caddy/Caddyfile.vps"
  - "tests/unit/apps/web/**"
  - "tests/unit/apps/api/**"

possible_secondary_touches:
  - ".github/workflows/deploy-web.yml"
  - "docs/runbooks/web-ui-gateway-same-origin.md"
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"

safety_notes:
  - "Do not create CSP that silently breaks auth/logout/register; verify in browser."
  - "SSE routes need buffering disabled at edge when used."
  - "Full gates are mandatory before publish."
---

# Task

Implement Stage 11 final security/performance/delivery hardening.

Done means:

- state-changing browser requests have CSRF/Origin protection;
- protected HTML is `no-store`;
- versioned assets have appropriate cache headers;
- CSP is tightened after CDN/inline scripts are gone;
- edge split HTML/assets vs `/api/*` is verified;
- SSE buffering is checked if SSE routes exist;
- full browser sweep passes;
- full backend gates pass.

## Context / Current State

- Prior page packages should already be implemented.
- This stage is not for new product features; it is for release hardening.

## Requirements (Must)

- Add CSRF strategy and tests.
- Add cache headers and asset versioning.
- Tighten CSP without breaking auth/UI.
- Verify production edge route assumptions.
- Run full gates.
- Run final Playwright sweep.
- Use `publish-ci-deploy` only after all checks pass.

## Requirements (Should)

- Keep security/config behavior centralized.
- Document any hardening decision in directly affected docs.
- Avoid broad unrelated refactors.

## Requirements (Nice-to-have)

- Add a small security header smoke test matrix.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 11, relevant web/api/security entrypoints, then conditional bundles for the hardening surface touched.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Inspect current final app surface and route/page packages.
2. Implement CSRF/cache/CSP/asset-versioning narrowly.
3. Add focused tests.
4. Run final local gates.
5. Run final Playwright sweep.
6. Review production risk.
7. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- Core auth flow does not need inline scripts.
- No external script CDN remains in shell.
- State-changing requests carry CSRF/Origin protection.
- Protected HTML has no-store.
- Assets are versioned.
- Browser QA has desktop/mobile screenshots and no console errors for main pages.
- Theme switcher works and financial colors are invariant.
- Backend gates and docs index pass.

# Implementation constraints

## API / contracts

- Config schema changes are `compatible-change`.
- Browser-visible behavior must be verified.
- Do not silently break public API or same-origin edge contract.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- New page features.
- New backend domain capabilities.
- Capacity threshold setting beyond smoke; Stage 12 owns capacity report.

# Quality gates (must run and pass)

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
python -m tools.docs.generate_docs_index --check
```

Final Playwright CLI sweep:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/final-landing.png
"$PWCLI" open http://127.0.0.1:8010/dashboard
"$PWCLI" snapshot
"$PWCLI" open http://127.0.0.1:8010/settings
"$PWCLI" snapshot
"$PWCLI" open http://127.0.0.1:8010/monitoring
"$PWCLI" snapshot
"$PWCLI" open http://127.0.0.1:8010/backtests
"$PWCLI" snapshot
"$PWCLI" open http://127.0.0.1:8010/backtests/new
"$PWCLI" snapshot
```

# Final output: report format (strict)

Report in Russian: `Intent`, `Scope`, `Security`, `Contract impact`, `Tests`, `Browser QA`, `Performance`, `Risks`, `Handoff`, `Publish/deploy`.
