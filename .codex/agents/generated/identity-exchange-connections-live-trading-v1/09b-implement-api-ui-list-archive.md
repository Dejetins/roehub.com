---
prompt_name: identity_exchange_connections_v1_09b_api_ui_list_archive
repo: roehub.com
branch: main
scope: "Stage 09B: expose active-only default list, disabled/archived history filters, archive action, and active-only limits after Stage 09A lifecycle foundation."

language:
  implementation: python_js_templates_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and browser verification rules"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 09 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "must confirm 09A accepted and update 09B"
    - path: docs/architecture/identity/exchange-connections-stage-reports/09a-lifecycle-domain-persistence.md
      why: "lifecycle foundation accepted evidence"
  task_entrypoints:
    - path: apps/api/routes/ui_account.py
      why: "public account facade routes and limits"
      inspect_symbols: ["get_exchange_connections", "post_exchange_connection_disable", "get_limits"]
    - path: apps/api/dto/ui_account.py
      why: "exchange connection response DTO status literals"
      inspect_symbols: ["ExchangeConnectionResponse", "ExchangeConnectionsResponse"]
    - path: apps/api/exchange_control_client.py
      why: "apps/api -> exchange-control client methods"
      inspect_symbols: ["ExchangeControlClient", "HttpExchangeControlClient"]
    - path: apps/web/templates/fragments/account/exchange_keys.html
      why: "settings exchange connections table and actions"
      inspect_symbols: ["settings.exchange"]
    - path: apps/web/dist/js/pages/settings.js
      why: "browser list rendering, filtering, actions"
      inspect_symbols: ["renderExchangeKeys", "statusClass"]
  conditional_bundles:
    exchange_control_internal_api:
      read_when: "archive internal command needs API/client wiring"
      paths:
        - src/trading/contexts/exchange_control/adapters/inbound/http/app.py
        - tests/unit/contexts/exchange_control/test_exchange_control_runtime.py
    browser_tests:
      read_when: "UI behavior or filters are changed"
      paths:
        - tests/unit/apps/web/test_app_routes.py
        - tests/unit/apps/api/test_ui_account_routes.py
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/09a-lifecycle-domain-persistence.md
      read_when: "09A handoff is unclear"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/09b-api-ui-list-archive.md"
  canonical_shape: "stage report with API/UI/browser evidence tables"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "09B"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_09a_must_be_accepted: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  default_list_active_only: true
  archive_endpoint_post_only: true
  delete_endpoint_forbidden: true
  limits_count_active_only: true
  archive_mutation_csrf_fail_closed: true
  archive_mutation_recent_auth_required: true
  browser_visible_evidence_required: true

task_toggles:
  implementation_changes_allowed: true
  browser_ui_changes_allowed: true
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing public API filters, DTO status literals, UI defaults, limits, or action availability"
    timing: "during implementation and final report"
    reason: "public account/UI behavior changes"
  - skill: backend-quality-gates
    use_when: "running focused route/web tests and static checks"
    timing: "during verification"
    reason: "API/UI stage"
  - skill: browser-qa-evidence
    use_when: "verifying settings table filters and archive behavior"
    timing: "during verification"
    reason: "browser-visible behavior changes"
  - skill: publish-ci-deploy
    use_when: "local/browser validation and docs are complete"
    timing: "after validation"
    reason: "direct-main delivery is required"

target_envs: ["local-dev", "browser", "mac-studio"]

required_literals:
  - "status=active"
  - "status=disabled"
  - "status=archived"
  - "/archive"
  - "Connected Exchange APIs"
  - "exchange_connections_used"

non_goals:
  - "Do not implement permission mismatch semantics in Stage 09B."
  - "Do not cleanup old stage08 records in Stage 09B."
  - "Do not add DELETE endpoint."
  - "Do not place orders."

final_report_format:
  language: ru
  sections: ["Вердикт", "API/UI", "Browser evidence", "Проверки", "Direct-main delivery", "Что дальше"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "uv run pyright apps/api src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - "apps/api/routes/ui_account.py"
  - "apps/api/dto/ui_account.py"
  - "apps/api/exchange_control_client.py"
  - "apps/web/templates/fragments/account/exchange_keys.html"
  - "apps/web/dist/js/pages/settings.js"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/09b-api-ui-list-archive.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - "src/trading/contexts/exchange_control/adapters/inbound/http/app.py"
  - "tests/unit/contexts/exchange_control/test_exchange_control_runtime.py"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Default list must not show disabled/archived records."
  - "Archive action must not imply physical deletion."
---

# Task

Implement Stage 09B API/UI list and archive semantics.

Done means:

- `GET /api/ui/account/exchange-connections` returns only active rows by default;
- explicit filters can show `disabled` and `archived`;
- `/settings` main table shows active rows only and has an explicit history/filter for disabled/archived;
- disabled rows can be archived through `POST .../archive`;
- active rows cannot be archived directly;
- account limits count only active rows;
- Stage 09B report and ledger are updated and delivered on `main`.

## Context / Current State

- Stage 09A must already have introduced `archived` lifecycle support.
- Current Stage 08 UI showed disabled test rows in `Connected Exchange APIs`; Stage 09B fixes that.
- Stage 09 plan forbids `DELETE` in v1 and requires explicit `POST .../archive`.

## Requirements (Must)

- Stop if 09A is not accepted in the ledger.
- Add `status=active|disabled|archived|all` query behavior; no status means active only.
- Update DTOs to include `archived` and `archived_at` where required.
- Add/route archive action only for owned disabled connections.
- Keep archive mutation protected by same-origin/CSRF and recent-auth gates; missing or cross-origin mutation context must fail closed before the command runs.
- Keep rotate/validate unavailable for disabled/archived.
- Update limits calculation to count `status == "active"` only.
- Update UI labels/actions so the main section does not imply disabled/archived are connected.
- Add browser-visible tests for default active-only list and explicit history/filter.
- Create Stage 09B report and update ledger.

## Requirements (Should)

- Prefer tabs or segmented controls for active/history filters.
- Keep table dimensions stable and avoid layout shifts.
- Preserve secret input clearing and password-manager hardening from Stage 08.

## Requirements (Nice-to-have)

- Include accessible labels for active/history controls.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 09, ledger, 09A report, then task entrypoints. Pre-implementation target: `<= 10 files`.

# Reading manifest

Use front matter as canonical. Read conditional bundles only if route/client/internal API changes are needed.

# Work plan (agent should follow)

1. Confirm `main` and fast-forward pull.
2. Verify 09A accepted.
3. Implement API/client/filter/archive behavior.
4. Update UI active/history rendering and actions.
5. Update focused tests.
6. Create Stage 09B report, update ledger.
7. Run gates.
8. Direct-main commit/push and CI/deploy follow-through.

# Acceptance criteria (Definition of Done)

- Default API list excludes disabled/archived.
- Explicit status filters work.
- Archive active returns deterministic rejection.
- Archive disabled returns archived state.
- Limits count only active.
- UI main table hides disabled/archived; history/filter exposes them explicitly.
- No `DELETE` route is added for Stage 09B.
- Docs and ledger updated.

# Implementation constraints

## Documentation

Create `09b-api-ui-list-archive.md`; update ledger after validation.

## Tests

Use focused API/web tests and browser QA evidence where available.

# Files to indicate (expected touched areas)

Use front matter expected touches.

# Non-goals

Permission semantics, controlled cleanup, production readiness.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/exchange_control`
- `uv run ruff check apps/api apps/web src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/exchange_control`
- `uv run pyright apps/api src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/contexts/exchange_control`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **API/UI**
3) **Browser evidence**
4) **Проверки**
5) **Direct-main delivery**
6) **Что дальше**
