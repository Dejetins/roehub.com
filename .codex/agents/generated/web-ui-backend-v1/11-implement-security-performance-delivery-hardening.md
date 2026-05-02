---
prompt_name: web_ui_backend_v1_11_security_performance_delivery_hardening
repo: roehub.com
branch: main
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

package_contract:
  depends_on:
    - "all implemented page packages accepted or explicitly skipped"
    - "12-capacity-load-validation is accepted or explicitly scheduled as the immediately-following capacity gate"
  owns:
    - "apps/web/main/app.py security/cache/asset hooks only"
    - "apps/web/main/security.py"
    - "apps/web/templates/base.html security/header integration only"
    - "apps/web/dist/** asset version references only"
    - "apps/api/main.py middleware/security integration only"
    - "infra/caddy/Caddyfile.vps"
    - "docs/runbooks/web-ui-gateway-same-origin.md"
    - "tests/unit/apps/web/** security/header assertions"
    - "tests/unit/apps/api/** security/middleware assertions"
  forbidden:
    - "new product page features"
    - "domain context internals except security boundary integration"
    - "migration chains"
    - "AI/backtest runtime feature work"
  integration_points:
    - "CSRF strategy"
    - "CSP policy"
    - "Cache-Control/asset versioning"
    - "edge split and SSE buffering"
    - "final Playwright sweep"
  handoff:
    - "release hardening evidence and deploy-ready gate status"

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
  - "Capacity validation belongs after main page packages; if it is not yet accepted, leave an explicit handoff to Stage 12 instead of treating capacity as complete."
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
- Stage 12 capacity/load validation should run after main page packages and be coordinated with this final hardening window.

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
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

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
