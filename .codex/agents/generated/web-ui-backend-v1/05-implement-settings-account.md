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
    - path: apps/api/wiring/modules/identity.py
      why: "identity composition root"
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
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

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

Report in Russian: `Intent`, `Scope`, `Design`, `Contract impact`, `Tests`, `Runtime evidence`, `Security`, `Risks`, `Handoff`, `Publish/deploy`.
