---
prompt_name: identity_exchange_connections_v1_10b_auto_validation_connect_rotate
repo: roehub.com
branch: main
scope: "Stage 10B: make connect/rotate automatically validate credentials and only keep trading-ready connections in Active."

language:
  implementation: python_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and direct-main delivery"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 10 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "must confirm 10A accepted and update 10B"
    - path: docs/architecture/identity/exchange-connections-stage-reports/10a-trading-capability-readiness-model.md
      why: "accepted readiness contract"
  task_entrypoints:
    - path: src/trading/contexts/exchange_control/application/connections.py
      why: "create/rotate/validate use cases"
      inspect_symbols: ["ExchangeConnectionService"]
    - path: src/trading/contexts/exchange_control/adapters/inbound/http/app.py
      why: "internal command API for create/rotate/validate"
      inspect_symbols: ["exchange_connections"]
    - path: apps/api/routes/ui_account.py
      why: "account facade create/rotate endpoints"
      inspect_symbols: ["exchange_connections"]
    - path: apps/api/exchange_control_client.py
      why: "apps/api -> exchange-control command client"
      inspect_symbols: ["ExchangeControlClient"]
  conditional_bundles:
    validation_adapters:
      read_when: "auto-validation requires normalizing Binance/Bybit responses or skip policy"
      paths:
        - src/trading/contexts/exchange_control/adapters/outbound/exchange_validation.py
        - tests/unit/contexts/exchange_control/test_exchange_validation.py
    persistence:
      read_when: "non-ready attempted connections need a durable non-active record"
      paths:
        - src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py
        - migrations/postgres
    api_tests:
      read_when: "account facade request/response behavior changes"
      paths:
        - tests/unit/apps/api/test_ui_account_routes.py
        - tests/unit/contexts/exchange_control
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/05-binance-bybit-validation.md
      read_when: "live validation skip policy or native adapter contract is unclear"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/10b-auto-validation-connect-rotate.md"
  canonical_shape: "stage report with auto-validation scenarios and evidence"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "10B"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_10a_must_be_accepted: true
  auto_validate_on_create: true
  auto_validate_on_rotate: true
  active_only_if_trading_ready: true
  readonly_not_active: true
  unsafe_or_invalid_not_active: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  no_trading_execution: true
  concrete_runtime_calls_required: true
  tests_are_not_acceptance: true
  env_backed_readonly_credentials_required_for_acceptance: true

task_toggles:
  implementation_changes_allowed: true
  public_ui_changes_allowed: false
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: root-cause-debugging
    use_when: "current create/rotate/validate flow behaves differently than Stage 10A contract"
    timing: "during investigation"
    reason: "avoid patching symptoms around command sequencing"
  - skill: contract-impact-analysis
    use_when: "changing create/rotate API behavior, errors, persistence state, or validation timing"
    timing: "before final report"
    reason: "auto-validation changes client-visible behavior"
  - skill: backend-quality-gates
    use_when: "running focused backend/API gates"
    timing: "during verification"
    reason: "backend/API stage"
  - skill: publish-ci-deploy
    use_when: "local validation and docs are complete"
    timing: "after validation"
    reason: "direct-main delivery required"

target_envs: ["local-dev", "mac-studio"]

required_literals:
  - "auto_validation"
  - "ready_for_trading"
  - "read_only_not_supported"
  - "ip_restriction_required"
  - "validation_unavailable"
  - "exchange_connection_auto_validation_total"

non_goals:
  - "Do not remove the UI permissions selector yet; Stage 10C owns browser CJM."
  - "Do not place or simulate orders."
  - "Do not physically delete failed connection attempts."

final_report_format:
  language: ru
  sections: ["Вердикт", "Auto-validation", "Runtime evidence", "Контракты", "Проверки", "Direct-main delivery", "Что дальше"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"

runtime_acceptance:
  required: true
  required_env:
    - "ROEHUB_E2E_BYBIT_MAINNET_READONLY_API_KEY"
    - "ROEHUB_E2E_BYBIT_MAINNET_READONLY_API_SECRET"
  commands:
    - cmd: "curl -i -X POST \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections\" -H \"Origin: $ROEHUB_BASE_URL\" -H \"Cookie: $ROEHUB_RECENT_AUTH_SESSION_COOKIE\" -H \"X-CSRF-Token: $ROEHUB_CSRF_TOKEN\" --data \"{\\\"exchange_name\\\":\\\"bybit\\\",\\\"market_type\\\":\\\"spot\\\",\\\"environment\\\":\\\"mainnet\\\",\\\"label\\\":\\\"stage10b_readonly_reject\\\",\\\"api_key\\\":\\\"$ROEHUB_E2E_BYBIT_MAINNET_READONLY_API_KEY\\\",\\\"api_secret\\\":\\\"$ROEHUB_E2E_BYBIT_MAINNET_READONLY_API_SECRET\\\"}\""
      expect: "deterministic not-active result with read_only_not_supported or equivalent Stage 10 reason; response/logs do not expose secrets"
    - cmd: "curl -fsS \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections?status=active\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" | jq -e 'all(.items[]; .label != \"stage10b_readonly_reject\")'"
      expect: "readonly attempt is absent from Active"
    - cmd: "curl -i -X POST \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections\" -H \"Origin: $ROEHUB_BASE_URL\" -H \"Cookie: $ROEHUB_RECENT_AUTH_SESSION_COOKIE\" -H \"X-CSRF-Token: $ROEHUB_CSRF_TOKEN\" --data '{\"exchange_name\":\"bybit\",\"market_type\":\"spot\",\"environment\":\"mainnet\",\"label\":\"stage10b_invalid_reject\",\"api_key\":\"INVALID\",\"api_secret\":\"INVALID\"}'"
      expect: "invalid credentials are rejected/not active with deterministic reason"
    - cmd: "curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_connection_auto_validation_total|exchange_connection_trading_readiness_total'"
      expect: "auto-validation/readiness counters are present after calls"
    - cmd: "psql \"$ROEHUB_PG_DSN\" -c \"SELECT label, status, status_reason, permission_summary_json FROM exchange_connections WHERE label IN ('stage10b_readonly_reject','stage10b_invalid_reject') ORDER BY created_at DESC;\""
      expect: "no selected row is active/trading-ready; no secret-bearing columns selected"
  acceptance_rule: "Readonly env-backed validation is mandatory. If readonly env vars are absent, Stage 10B is blocked, not accepted. Fake/in-memory validation is not production evidence."

expected_primary_touches:
  - "src/trading/contexts/exchange_control/application/connections.py"
  - "src/trading/contexts/exchange_control/adapters/inbound/http/app.py"
  - "apps/api/routes/ui_account.py"
  - "apps/api/exchange_control_client.py"
  - "tests/unit/contexts/exchange_control"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/10b-auto-validation-connect-rotate.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - "src/trading/contexts/exchange_control/adapters/outbound/exchange_validation.py"
  - "src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py"
  - "migrations/postgres"
  - "docs/architecture/README.md"

safety_notes:
  - "Auto-validation must not log plaintext credentials, ciphertext, HMAC, tokens, cookies, or raw exchange bodies."
  - "Validation unavailable is not success; do not mark connection active-ready from local skip evidence."
---

# Task

Implement Stage 10B auto-validation for connect and rotate.

Done means:

- create/connect automatically validates credentials before a connection can be active/trading-ready;
- rotate automatically validates the new credential version before it becomes active;
- readonly, unsafe permissions, invalid credentials, missing required IP restriction, and validation unavailable do not leave an active/limit-consuming connection;
- manual validation remains available only as re-check behavior for later UI work;
- Stage 10B report and ledger are updated and delivered on `main`.

## Context / Current State

- Stage 10A defines trading-only readiness.
- Current Stage 09 flow can save active rows with `permission_mismatch`; Stage 10B must stop producing new active non-trading-ready rows.

## Requirements (Must)

- Stop if Stage 10A is not accepted in ledger.
- Auto-run validation on create and rotate.
- Treat validation skip/unavailable as non-ready, not active success.
- Preserve CSRF/same-origin and recent-auth gates before mutation.
- Use `exchange-control` for secret-bearing validation; do not move plaintext/decrypt paths into `apps/api`.
- Add metrics/audit for auto-validation outcomes with bounded labels.
- Add tests for:
  - trading-ready -> active;
  - readonly -> not active;
  - invalid credentials -> not active;
  - unsafe withdrawal/transfer -> not active;
  - missing mainnet IP restriction -> not active;
  - validation unavailable -> not active;
  - rotate failure does not replace working credential version.
- Run the concrete create/list/metrics/DB calls in `runtime_acceptance`; tests alone are not acceptance.
- Create Stage 10B report and update ledger.

## Requirements (Should)

- Prefer deterministic error codes suitable for UI:
  - `read_only_not_supported`;
  - `unsafe_permissions`;
  - `ip_restriction_required`;
  - `invalid_credentials`;
  - `validation_unavailable`.
- Keep non-ready attempted connections out of account limits.

## Requirements (Nice-to-have)

- Include a create/rotate decision table in the report.

# Context acquisition protocol

Read `.codex/AGENTS.md`, Stage 10 plan, ledger, 10A report, then task entrypoints. Pre-implementation target: `<= 12 files`.

# Reading manifest

Use front matter as canonical. Do not inspect UI files unless tests require it.

# Work plan (agent should follow)

1. Confirm `main` and fast-forward pull.
2. Verify 10A accepted.
3. Trace current create/rotate/validate command flow.
4. Implement auto-validation sequencing and non-ready handling.
5. Add focused tests and metrics/audit assertions.
6. Create Stage 10B report and update ledger.
7. Run gates.
8. Direct-main delivery and CI/deploy follow-through.

# Acceptance criteria (Definition of Done)

- New create/rotate cannot produce active non-trading-ready records.
- Failed rotate preserves the previous active credential version.
- Readonly key is rejected/not active with deterministic reason.
- Validation unavailable is not success.
- Concrete readonly and invalid credential API calls prove not-active behavior, or the stage is blocked due to missing env-backed readonly credentials.
- Secret boundary remains in exchange-control.
- Docs and ledger updated.

# Implementation constraints

## Documentation

- Create `docs/architecture/identity/exchange-connections-stage-reports/10b-auto-validation-connect-rotate.md`.
- Update ledger before final output.

# Files to indicate (expected touched areas)

Use front matter expected touches.

# Non-goals

UI CJM, production Playwright readiness, data repair of existing rows, trading execution.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py`
- `uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **Auto-validation**
3) **Runtime evidence**
4) **Контракты**
5) **Проверки**
6) **Direct-main delivery**
7) **Что дальше**
