---
prompt_name: web_ui_backend_v1_05_settings_account
repo: roehub.com
branch: current
scope: "Этап 5: settings/account page, preferences, integrations, sessions, audit, exchange keys UI."

language:
  implementation: python_fastapi_jinja_css_js_sql
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "security, contracts, DDD, gates"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 5 source of truth"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "settings visual/theme contract"
  task_entrypoints:
    - path: apps/api/routes/identity.py
      why: "identity router facade for auth/exchange keys"
    - path: src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
      why: "current exchange-key route contract"
    - path: docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
      why: "active exchange secret policy"
    - path: apps/web/templates/pages/settings.html
      why: "settings page target"
  conditional_bundles:
    identity_auth:
      read_when: "when sessions/current-user/profile behavior is implemented"
      paths:
        - docs/architecture/identity/identity-keycloak-auth-model-v1.md
        - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
        - src/trading/contexts/identity/adapters/outbound/persistence/postgres/session_repository.py
    migrations:
      read_when: "when adding identity preferences/integrations/audit/profile tables"
      paths:
        - migrations/postgres/0005_identity_keycloak_cutover_v1.sql
        - apps/migrations/bootstrap.py
        - tests/unit/apps/migrations/test_bootstrap_apply_flow.py
    tests:
      read_when: "when adding account routes/use cases"
      paths:
        - tests/unit/apps/api/test_identity_exchange_keys_routes.py
        - tests/unit/apps/web/test_app_routes.py
  consult_if_needed:
    - path: docs/runbooks/keycloak-local-setup-and-ops.md
      read_when: "auth/session local setup is ambiguous"

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
  secrets_write_only: true
  no_secret_leak_in_response_dom_logs_screenshots: true
  duplicate_exchange_key_code: "exchange_key_already_exists"
  cursor_pagination_sessions_audit: true
  identity_migrations_channel_required: true
  csrf_strategy_gate_required_for_mutations: true
  theme_preference_preserves_financial_colors: true
  browser_qa_required: true

task_toggles:
  implement_ui_account_routes: true
  implement_identity_preferences: true
  implement_audit_events: true
  implement_settings_page: true
  implement_exchange_key_ui: true
  publish_after_success: true

package_contract:
  depends_on:
    - "01-shell-auth-register accepted"
    - "02-design-system-js-core accepted"
    - "CSRF strategy chosen or mutation gate explicitly documented"
  owns:
    - "apps/api/routes/ui_account.py"
    - "apps/api/dto/ui_account.py"
    - "apps/api/wiring/modules/ui_account.py"
    - "src/trading/contexts/identity/** account preferences/audit additions only"
    - "migrations/postgres/0006_*.sql"
    - "apps/web/templates/pages/settings.html"
    - "apps/web/templates/fragments/account/**"
    - "apps/web/dist/js/pages/settings.js"
    - "apps/web/dist/css/pages/settings.css"
    - "tests/unit/apps/api/test_ui_account_routes.py"
  forbidden:
    - "backtest context/files"
    - "strategy context/files"
    - "monitoring package files"
    - "exchange secret policy rewrites beyond additive UI needs"
  integration_points:
    - "identity migration chain"
    - "apps/api/main.py route include"
    - "exchange-key route contract"
    - "theme preference default resolution"
  handoff:
    - "owner-scoped account/preferences/audit endpoints and settings UI"

skill_routing:
  - skill: architecture-design
    use_when: "if profile/preferences/integrations/audit table ownership or identity boundary is not clear"
    timing: "before implementation only if needed"
    reason: "identity/account tables are persisted contracts"
  - skill: contract-impact-analysis
    use_when: "adding account endpoints, DTOs, persisted schema, audit event schema, config/defaults, or theme preferences"
    timing: "before implementation and final report"
    reason: "settings crosses public API, secrets, persistence, config, browser defaults"
  - skill: backend-quality-gates
    use_when: "running identity/API/web/migration tests, ruff, pyright"
    timing: "during verification"
    reason: "stage has backend and migration surfaces"
  - skill: browser-qa-evidence
    use_when: "verifying settings workflows, secret masking, duplicate 409, theme persistence, mobile layout"
    timing: "after backend tests"
    reason: "settings is browser-visible and security-sensitive"
  - skill: playwright
    use_when: "capturing screenshots/snapshots"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI"
  - skill: publish-ci-deploy
    use_when: "all backend/browser/migration gates pass and secret-leak checks are clean"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "/settings"
  - "/api/exchange-keys"
  - "/api/ui/account/preferences"
  - "/ui/account/preferences"
  - "exchange_key_already_exists"
  - "identity_user_preferences"
  - "identity_audit_events"
  - "terminal-orange"

non_goals:
  - "Do not change exchange secret storage policy except additive UI/account needs."
  - "Do not expose api_secret/passphrase/plain api_key."
  - "Do not implement local 2FA."
  - "Do not add arbitrary webhook integrations without validation."

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
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes; create focused tests if missing"
  - cmd: "uv run pytest -q tests/unit/apps/migrations"
    expect: "passes if migrations/bootstrap changed"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/api/routes/ui_account.py"
  - "apps/api/dto/ui_account.py"
  - "apps/api/wiring/modules/ui_account.py"
  - "apps/api/main/app.py"
  - "src/trading/contexts/identity/**"
  - "migrations/postgres/0006_*.sql"
  - "apps/web/templates/pages/settings.html"
  - "apps/web/templates/fragments/account/*"
  - "apps/web/dist/js/pages/settings.js"
  - "apps/web/dist/css/pages/settings.css"
  - "tests/unit/apps/api/test_ui_account_routes.py"

possible_secondary_touches:
  - "apps/migrations/bootstrap.py"
  - "tests/unit/apps/migrations/*"
  - "docs/architecture/identity/*.md"

safety_notes:
  - "Actual backend route paths are `/ui/account/...`; browser sees `/api/ui/account/...`."
  - "If CSRF strategy is not implemented, do not broaden mutation surface without a documented gate."
  - "Exchange secrets must not appear in DOM, JSON, logs, screenshots, or Playwright artifacts."
---

# Task

Implement Stage 5 settings/account.

Done means:

- protected `/settings` page exists;
- account/profile/limits/integrations/notifications/preferences/sessions/audit endpoints exist as scoped UI API where required;
- exchange keys UI uses existing secret-safe exchange-key endpoints;
- theme preference persists and never changes financial color semantics;
- destructive/settings mutations write audit events;
- sessions/audit are cursor-paginated;
- browser/security evidence exists.

## Context / Current State

- Current backend already has auth/current-user and exchange keys.
- Missing account preferences, integrations, audit, sessions UI read-models.
- Identity SQL migrations live under `migrations/postgres`.

## Requirements (Must)

- Preserve exchange-key v2 secret policy.
- Duplicate active exchange key remains deterministic `409` with code `exchange_key_already_exists`.
- Add owner-scoped persisted tables only through correct identity migration channel.
- Add authorization/owner-scope tests.
- Add secret leak checks in tests or browser evidence.
- Use `publish-ci-deploy` only after complete success.

## Requirements (Should)

- Use HTMX/forms for low-frequency settings fragments where it fits.
- Keep JS small and page-scoped.
- Use cursor pagination for sessions/audit.

## Requirements (Nice-to-have)

- Add density preference if it is cheap and does not broaden scope.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 5, design manifest settings sections, exchange-key policy, then task entrypoints. Expand into migrations only if adding tables.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Confirm CSRF/state-changing gate and identity DB ownership.
2. Define UI account DTOs/errors/pagination.
3. Implement identity use cases/ports/adapters and migrations.
4. Add API routes/wiring.
5. Implement settings page/fragments/JS/CSS.
6. Add tests for authz, validation, duplicates, audit, defaults, secret leak.
7. Run browser QA and quality gates.
8. Use `publish-ci-deploy` only after all gates pass.

# Acceptance criteria (Definition of Done)

- `/settings` opens behind auth gate.
- Add/list/delete exchange key flow works without secret leakage.
- Duplicate exchange key shows deterministic 409.
- Delete is confirmed and UX-idempotent.
- Toggles/preferences save without full reload where implemented.
- Theme preference survives reload and preserves financial colors.
- Sessions/audit paginate.
- Mobile layout has no horizontal overflow.

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
- Persisted schema: `compatible-change` through additive identity tables.
- Config schema: `none` unless new integration credentials are introduced.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Local password registration.
- Local 2FA.
- Arbitrary external integrations.
- Strategy/backtest workflows.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/web/test_app_routes.py
uv run pytest -q tests/unit/apps/migrations
uv run ruff check apps/api apps/web src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/settings
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/settings-desktop.png
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
