---
prompt_name: identity_exchange_connections_v1_10c_settings_trading_cjm_ui
repo: roehub.com
branch: main
scope: "Stage 10C: simplify /settings exchange connection CJM for trading-only accounts: no read/trade selector, mainnet default with advanced testnet, Active/History only, Disconnect and Re-check semantics."

language:
  implementation: python_js_templates_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and browser verification"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 10 UI source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "must confirm 10B accepted and update 10C"
    - path: docs/architecture/identity/exchange-connections-stage-reports/10b-auto-validation-connect-rotate.md
      why: "accepted auto-validation behavior"
  task_entrypoints:
    - path: apps/web/templates/fragments/account/exchange_keys.html
      why: "settings exchange connection markup"
      inspect_symbols: ["settings.exchange"]
    - path: apps/web/dist/js/pages/settings.js
      why: "browser rendering/actions/filter behavior"
      inspect_symbols: ["renderExchangeKeys", "connect", "validate", "disable"]
    - path: apps/api/routes/ui_account.py
      why: "browser account endpoints and request shape"
      inspect_symbols: ["exchange_connections"]
    - path: apps/api/dto/ui_account.py
      why: "DTO fields rendered by UI"
      inspect_symbols: ["ExchangeConnectionResponse"]
  conditional_bundles:
    tests:
      read_when: "updating web/API route tests"
      paths:
        - tests/unit/apps/web/test_app_routes.py
        - tests/unit/apps/api/test_ui_account_routes.py
    browser_assets:
      read_when: "CSS/layout changes are needed"
      paths:
        - apps/web/dist/css
        - apps/web/templates
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/06-settings-ui.md
      read_when: "previous UI constraints or password-manager hardening details are unclear"
    - path: docs/architecture/identity/exchange-connections-stage-reports/08-settings-production-repair.md
      read_when: "production browser origin or password-manager artifact requirements are unclear"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/10c-settings-trading-cjm-ui.md"
  canonical_shape: "stage report with UI/browser evidence tables"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "10C"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_10b_must_be_accepted: true
  remove_permissions_selector: true
  mainnet_default: true
  testnet_advanced_control: true
  active_history_only: true
  no_disabled_tab: true
  disconnect_label_required: true
  recheck_not_happy_path_validate: true
  password_manager_hardening_preserved: true
  browser_visible_evidence_required: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  concrete_runtime_calls_required: true
  tests_are_not_acceptance: true

task_toggles:
  implementation_changes_allowed: true
  browser_ui_changes_allowed: true
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing browser-visible defaults, request payloads, action labels, or status filters"
    timing: "before final report"
    reason: "Stage 10C changes user workflow"
  - skill: browser-qa-evidence
    use_when: "verifying /settings user flow and UI state"
    timing: "during verification"
    reason: "browser-visible CJM stage"
  - skill: playwright
    use_when: "running local or production-like browser proof"
    timing: "during verification"
    reason: "user requires e2e evidence"
  - skill: backend-quality-gates
    use_when: "running web/API tests, lint, type checks, docs index"
    timing: "during verification"
    reason: "UI still depends on API contracts"
  - skill: publish-ci-deploy
    use_when: "browser evidence and docs are complete"
    timing: "after validation"
    reason: "direct-main delivery required"

target_envs: ["local-dev", "browser", "mac-studio"]

required_literals:
  - "Connect and validate"
  - "Disconnect"
  - "Re-check"
  - "Active"
  - "History"
  - "Ready for trading"
  - "read_only_not_supported"

non_goals:
  - "Do not implement backend auto-validation; Stage 10B owns it."
  - "Do not run production data reclassification; Stage 10D owns it."
  - "Do not add order placement."

final_report_format:
  language: ru
  sections: ["Вердикт", "CJM/UI", "Browser evidence", "Проверки", "Direct-main delivery", "Что дальше"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "uv run ruff check apps/web apps/api src/trading/contexts/exchange_control tests/unit/apps/web tests/unit/apps/api tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "uv run pyright apps/api src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"

runtime_acceptance:
  required: true
  commands:
    - cmd: "curl -fsS \"$ROEHUB_BASE_URL/settings\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" | rg 'Connect and validate|Active|History|Disconnect|Re-check'"
      expect: "settings HTML contains the new CJM labels"
    - cmd: "! curl -fsS \"$ROEHUB_BASE_URL/settings\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" | rg 'name=\"permissions\"|data-permissions|>Disabled<'"
      expect: "settings HTML does not expose read/trade selector or separate Disabled tab"
    - cmd: "authenticated Playwright /settings: open connect dialog, verify no read/trade selector, default mainnet, testnet advanced, Active/History only, Disconnect/Re-check labels, and capture network create payload"
      expect: "browser proof and network payload show no user-selected permissions field"
    - cmd: "rg -n 'stage10|api_secret|apiKey|password' output/playwright .playwright-cli || true"
      expect: "no real secret values or password-manager artifacts are present; any field-name-only matches are explained"
  acceptance_rule: "If authenticated browser/network evidence cannot be collected, Stage 10C is blocked or partial. Route tests alone are not acceptance."

expected_primary_touches:
  - "apps/web/templates/fragments/account/exchange_keys.html"
  - "apps/web/dist/js/pages/settings.js"
  - "apps/api/routes/ui_account.py"
  - "apps/api/dto/ui_account.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/10c-settings-trading-cjm-ui.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - "apps/web/dist/css"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Do not reintroduce fields that trigger browser password managers to save API keys as Roehub login credentials."
  - "Disabled remains backend state but must not be a separate user-facing tab."
---

# Task

Implement Stage 10C `/settings` trading-only CJM.

Done means:

- add form has no `read`/`trade` permissions selector;
- create action is framed as `Connect and validate`;
- `Mainnet` is default; `Testnet` is visible only as advanced/dev/test control;
- main list has `Active` and `History`, not separate `Disabled`;
- active rows show `Ready for trading` and actions `Re-check`, `Rotate`, `Disconnect`;
- `Disconnect` calls the existing backend disable semantics;
- History contains disabled/archived/rejected records as applicable;
- password-manager hardening remains intact;
- Stage 10C report and ledger are updated and delivered on `main`.

## Context / Current State

- Stage 10B must already prevent new non-trading-ready connections from becoming active.
- Stage 09B introduced Active/Disabled/Archived filters; Stage 10C simplifies this to Active/History for users.

## Requirements (Must)

- Stop if Stage 10B is not accepted in ledger.
- Remove the user-facing permissions selector from `/settings`.
- Ensure submitted create payload no longer depends on user-selected `read`/`trade`.
- Default environment to mainnet.
- Keep testnet available only as advanced/dev-style control with clear wording.
- Replace visible `Disabled` tab with `History`.
- Rename user-facing disable action to `Disconnect`.
- Rename manual validate action to `Re-check` and make it secondary.
- Make readonly/not-usable errors clear: not partially successful, not active.
- Preserve secret input clearing and password-manager suppression.
- Add route/web tests and browser evidence.
- Run concrete HTML/API/browser/network checks from `runtime_acceptance`; tests alone are not acceptance.
- Create Stage 10C report and update ledger.

## Requirements (Should)

- Keep layout dense and operational, not a marketing/landing page.
- Avoid in-app explanatory paragraphs; use concise labels/statuses.
- Keep table dimensions stable.

## Requirements (Nice-to-have)

- Add accessible labels/tooltips for advanced testnet and re-check.

# Context acquisition protocol

Read `.codex/AGENTS.md`, Stage 10 plan, ledger, 10B report, then task entrypoints. Pre-implementation target: `<= 12 files`.

# Reading manifest

Use front matter as canonical. Read browser assets only if layout/CSS changes are required.

# Work plan (agent should follow)

1. Confirm `main` and fast-forward pull.
2. Verify 10B accepted.
3. Inspect current `/settings` exchange panel and tests.
4. Implement UI/CJM changes.
5. Add/update tests.
6. Run browser QA evidence.
7. Create Stage 10C report and update ledger.
8. Run gates.
9. Direct-main delivery.

# Acceptance criteria (Definition of Done)

- No visible read/trade selector.
- Create payload does not rely on user-selected permissions.
- Active tab contains only trading-ready rows.
- History replaces Disabled/Archived user tabs.
- Disconnect/Re-check labels render correctly and actions work.
- Password manager does not offer to save API key/secret as Roehub login.
- Browser evidence, network payload evidence and secret artifact grep are recorded.
- Docs and ledger updated.

# Implementation constraints

## Documentation

- Create `docs/architecture/identity/exchange-connections-stage-reports/10c-settings-trading-cjm-ui.md`.
- Update ledger before final output.

# Files to indicate (expected touched areas)

Use front matter expected touches.

# Non-goals

Backend auto-validation, data cleanup, trading execution.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/contexts/exchange_control`
- `uv run ruff check apps/web apps/api src/trading/contexts/exchange_control tests/unit/apps/web tests/unit/apps/api tests/unit/contexts/exchange_control`
- `uv run pyright apps/api src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/contexts/exchange_control`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **CJM/UI**
3) **Browser evidence**
4) **Проверки**
5) **Direct-main delivery**
6) **Что дальше**
