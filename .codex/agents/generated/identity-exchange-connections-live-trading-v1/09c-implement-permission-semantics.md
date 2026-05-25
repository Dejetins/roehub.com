---
prompt_name: identity_exchange_connections_v1_09c_permission_semantics
repo: roehub.com
branch: main
scope: "Stage 09C: introduce requested/exchange/effective permissions and permission_mismatch semantics inside exchange-control, preserving API compatibility."

language:
  implementation: python_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 09 permission source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "must confirm 09B accepted and update 09C"
    - path: docs/architecture/identity/exchange-connections-stage-reports/09b-api-ui-list-archive.md
      why: "API/UI lifecycle handoff"
  task_entrypoints:
    - path: src/trading/contexts/exchange_control/application/validation.py
      why: "validation status/result contract"
      inspect_symbols: ["VALIDATION_STATUSES", "ExchangeCredentialValidationResult"]
    - path: src/trading/contexts/exchange_control/adapters/outbound/exchange_validation.py
      why: "Binance/Bybit permission normalization"
      inspect_symbols: ["normalize_binance_api_restrictions", "normalize_bybit_api_key_info"]
    - path: src/trading/contexts/exchange_control/application/connections.py
      why: "connection view and validation recording"
      inspect_symbols: ["ExchangeConnectionView", "validate_connection"]
    - path: apps/api/dto/ui_account.py
      why: "public DTO compatibility"
      inspect_symbols: ["ExchangeConnectionResponse"]
    - path: apps/web/dist/js/pages/settings.js
      why: "UI display of permission fields and mismatch"
      inspect_symbols: ["renderExchangeKeys", "statusClass"]
  conditional_bundles:
    persistence:
      read_when: "permission fields are persisted outside permission_summary_json"
      paths:
        - migrations/postgres/0008_exchange_connections_v1.sql
        - src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py
        - tests/unit/apps/migrations
    tests:
      read_when: "updating validation/API/UI tests"
      paths:
        - tests/unit/contexts/exchange_control/test_exchange_validation.py
        - tests/unit/apps/api/test_ui_account_routes.py
        - tests/unit/apps/web/test_app_routes.py
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/05-binance-bybit-validation.md
      read_when: "existing validation acceptance contract is unclear"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/09c-permission-semantics.md"
  canonical_shape: "stage report with permission truth table and evidence"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "09C"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_09b_must_be_accepted: true
  effective_permissions_exchange_control_owned: true
  permission_mismatch_canonical_status: true
  requested_trade_readonly_reason_required: true
  compatibility_permissions_alias_required: true
  no_execution_required: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true

task_toggles:
  implementation_changes_allowed: true
  ui_display_changes_allowed: true
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing validation status enum, DTO fields, persisted permission metadata, or UI defaults"
    timing: "before final report"
    reason: "permissions are externally visible and future execution-critical"
  - skill: backend-quality-gates
    use_when: "running validator/API/UI tests and static checks"
    timing: "during verification"
    reason: "permission semantics are backend-owned"
  - skill: browser-qa-evidence
    use_when: "UI permission/mismatch display changes"
    timing: "during verification"
    reason: "status must not mislead the user"
  - skill: publish-ci-deploy
    use_when: "validation and docs are complete"
    timing: "after validation"
    reason: "direct-main delivery required"

target_envs: ["local-dev", "browser", "mac-studio"]

required_literals:
  - "requested_permissions"
  - "exchange_permissions"
  - "effective_permissions"
  - "permission_mismatch"
  - "requested_trade_but_exchange_readonly"
  - "exchange_permissions_exceed_requested"

non_goals:
  - "Do not add trading execution."
  - "Do not change exchange SDK endpoints except permission normalization."
  - "Do not cleanup old stage08 rows."

final_report_format:
  language: ru
  sections: ["Вердикт", "Permission semantics", "Контракты", "Проверки", "Direct-main delivery", "Что дальше"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/exchange_control/application/validation.py"
  - "src/trading/contexts/exchange_control/adapters/outbound/exchange_validation.py"
  - "src/trading/contexts/exchange_control/application/connections.py"
  - "apps/api/dto/ui_account.py"
  - "apps/api/routes/ui_account.py"
  - "apps/web/dist/js/pages/settings.js"
  - "tests/unit/contexts/exchange_control"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/09c-permission-semantics.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - "migrations/postgres"
  - "src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py"
  - "docs/architecture/README.md"

safety_notes:
  - "Do not let apps/api or UI compute effective permissions independently."
  - "Do not treat requested trade + readonly exchange key as trade-ready."
---

# Task

Implement Stage 09C permission semantics.

Done means:

- `requested_permissions`, `exchange_permissions`, and `effective_permissions` are explicit in the domain/API/UI contract;
- `effective_permissions` is computed inside `exchange-control`;
- `permission_mismatch` is a canonical validation status;
- `requested_trade_but_exchange_readonly` is a validation reason;
- `permissions` remains a compatibility alias for `requested_permissions`;
- UI no longer displays requested trade + readonly exchange validation as normal successful trade state;
- Stage 09C report and ledger are updated and delivered on `main`.

## Context / Current State

- Current implementation stores `permissions` as a user-requested value and validation status separately.
- This produced confusing UI states like requested `trade` alongside `valid_readonly`.
- Stage 09 plan requires `effective_permissions` as the future execution-facing value.

## Requirements (Must)

- Stop if 09B is not accepted in ledger.
- Add canonical `permission_mismatch` status and tests.
- Keep `skipped_external_validation` in the status contract.
- Compute:
  - requested `read` + exchange readonly -> effective `read`, `valid_readonly`;
  - requested `read` + exchange trade -> effective `read`, warning `exchange_permissions_exceed_requested`;
  - requested `trade` + exchange trade -> effective `trade`, `valid_trade_enabled`;
  - requested `trade` + exchange readonly -> effective `read`, `permission_mismatch`, reason `requested_trade_but_exchange_readonly`;
  - withdrawal/transfer -> effective `none`, `invalid_permissions`;
  - invalid credentials -> effective `none`, `invalid_credentials`.
- Preserve old `permissions` response field as alias to `requested_permissions`.
- Make UI display mismatch as warning/negative, not normal success.
- Add tests for Binance and Bybit normalization.
- Create Stage 09C report and update ledger.

## Requirements (Should)

- Prefer keeping permission detail in `permission_summary_json` unless dedicated columns are clearly lower-risk.
- Keep labels bounded for mismatch metrics.

## Requirements (Nice-to-have)

- Include a truth table in the stage report.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 09, ledger, 09B report, then task entrypoints. Pre-implementation target: `<= 12 files`.

# Reading manifest

Use front matter as canonical. Do not inspect execution/order code unless a grep shows accidental imports.

# Work plan (agent should follow)

1. Confirm `main` and fast-forward pull.
2. Verify 09B accepted.
3. Implement permission status/result model in `exchange-control`.
4. Update API DTO mappings and compatibility alias.
5. Update UI display and tests.
6. Update report/ledger.
7. Run gates.
8. Direct-main delivery.

# Acceptance criteria (Definition of Done)

- All truth-table scenarios have tests.
- Public DTO includes explicit fields plus `permissions` alias.
- `effective_permissions` is exchange-control-owned.
- UI distinguishes mismatch.
- No order/execution path is added.

# Implementation constraints

## Security

No secrets/raw exchange responses in logs, metrics, docs, or tests.

## Documentation

Create `09c-permission-semantics.md` and update ledger.

# Files to indicate (expected touched areas)

Use front matter expected touches.

# Non-goals

Cleanup/backfill, production readiness, execution.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py`
- `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web`
- `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **Permission semantics**
3) **Контракты**
4) **Проверки**
5) **Direct-main delivery**
6) **Что дальше**
