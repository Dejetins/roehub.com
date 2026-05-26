---
prompt_name: identity_exchange_connections_v1_10a_trading_capability_readiness_model
repo: roehub.com
branch: main
scope: "Stage 10A: introduce trading-only product capability/readiness semantics for exchange connections while keeping legacy permission fields as deprecated compatibility surface."

language:
  implementation: python_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, safety invariants, direct-main delivery"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 10 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "handoff ledger; confirm Stage 09E accepted and update Stage 10A"
    - path: docs/architecture/identity/exchange-connections-stage-reports/09e-lifecycle-production-readiness.md
      why: "latest accepted production lifecycle evidence"
  task_entrypoints:
    - path: src/trading/contexts/exchange_control/application/connections.py
      why: "connection domain/read-model and validation recording"
      inspect_symbols: ["ExchangeConnectionView", "ExchangeConnectionService"]
    - path: src/trading/contexts/exchange_control/application/validation.py
      why: "validation status/result contract"
      inspect_symbols: ["ExchangeCredentialValidationResult"]
    - path: apps/api/dto/ui_account.py
      why: "public account facade DTO compatibility"
      inspect_symbols: ["ExchangeConnectionResponse"]
    - path: apps/api/routes/ui_account.py
      why: "account facade create/list mappings"
      inspect_symbols: ["exchange_connections"]
  conditional_bundles:
    persistence:
      read_when: "capability/readiness cannot safely fit existing permission_summary_json"
      paths:
        - migrations/postgres
        - src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py
        - tests/unit/apps/migrations
    ui_tests:
      read_when: "DTO changes require web route updates in this stage"
      paths:
        - tests/unit/apps/api/test_ui_account_routes.py
        - tests/unit/apps/web/test_app_routes.py
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/09c-permission-semantics.md
      read_when: "legacy requested/exchange/effective permission semantics are unclear"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/10a-trading-capability-readiness-model.md"
  canonical_shape: "stage report with contract/readiness truth table and evidence"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "10A"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_09e_must_be_accepted: true
  trading_only_product_capability: true
  read_only_not_successful_connection: true
  permissions_fields_deprecated_not_authoritative: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  no_trading_execution: true
  no_secret_leakage: true
  concrete_runtime_calls_required: true
  tests_are_not_acceptance: true

task_toggles:
  implementation_changes_allowed: true
  ui_selector_changes_allowed: false
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing DTO fields, readiness semantics, legacy permissions compatibility, persistence metadata, or API behavior"
    timing: "during design and before final report"
    reason: "Stage 10 intentionally changes product semantics"
  - skill: backend-quality-gates
    use_when: "running focused Python tests, ruff, pyright, docs index"
    timing: "during verification"
    reason: "backend/domain/DTO stage"
  - skill: publish-ci-deploy
    use_when: "local gates, report and ledger are complete"
    timing: "after validation"
    reason: "direct-main delivery is required"

target_envs: ["local-dev", "mac-studio"]

required_literals:
  - "requested_capability"
  - "effective_capability"
  - "connection_readiness"
  - "ready_for_trading"
  - "read_only_not_supported"
  - "permissions_deprecated"

non_goals:
  - "Do not remove the UI permissions selector in Stage 10A; that belongs to Stage 10C."
  - "Do not implement auto-validation create/rotate in Stage 10A; that belongs to Stage 10B unless required for tests."
  - "Do not place orders or add exchange-execution behavior."
  - "Do not physically delete existing records."

final_report_format:
  language: ru
  sections: ["Вердикт", "Capability/readiness", "Runtime evidence", "Контракты", "Проверки", "Direct-main delivery", "Что дальше"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"

runtime_acceptance:
  required: true
  commands:
    - cmd: "curl -fsS \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections?status=active\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" | jq '.items[] | {connection_id,status,requested_capability,effective_capability,connection_readiness,permissions,requested_permissions}'"
      expect: "response shape exposes Stage 10 readiness/capability fields; legacy permissions fields are present only as compatibility/non-authoritative fields"
    - cmd: "psql \"$ROEHUB_PG_DSN\" -c \"SELECT connection_id, status, permission_summary_json FROM exchange_connections ORDER BY created_at DESC LIMIT 5;\""
      expect: "DB/read-model evidence shows where readiness/capability is sourced without selecting secret-bearing columns"
    - cmd: "curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_connection_trading_readiness_total|exchange_control_active'"
      expect: "runtime metrics surface is alive and Stage 10 readiness metric exists if introduced in this stage"
  acceptance_rule: "If these runtime calls cannot be executed, mark Stage 10A blocked or partial; unit tests alone are not acceptance."

expected_primary_touches:
  - "src/trading/contexts/exchange_control/application/connections.py"
  - "src/trading/contexts/exchange_control/application/validation.py"
  - "apps/api/dto/ui_account.py"
  - "apps/api/routes/ui_account.py"
  - "tests/unit/contexts/exchange_control"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/10a-trading-capability-readiness-model.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - "migrations/postgres"
  - "src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Read-only credentials must not be represented as successful Roehub trading connections."
  - "Legacy permissions fields may remain for compatibility but must not be the source of product readiness."
---

# Task

Implement Stage 10A trading capability/readiness model.

Done means:

- Roehub account exchange connections have explicit product readiness semantics for trading;
- `requested_capability` is `trading` for the `/settings` product flow;
- `effective_capability` is `trading` or `none`;
- `connection_readiness` exposes user-facing readiness such as `ready_for_trading`, `needs_action`, `rejected`, `disconnected`, `archived`;
- legacy `permissions` / `requested_permissions` remain compatibility fields but are not authoritative for product readiness;
- Stage 10A report and ledger are updated;
- changes are delivered directly to `main` after validation.

## Context / Current State

- Stage 09 is accepted and provides lifecycle, archive, permission mismatch, and Active/History foundations.
- Stage 10 changes the product model: user connects an exchange account for strategy trading, not a read-only monitoring key.
- This stage defines backend/domain/API readiness semantics before UI and auto-validation changes.

## Requirements (Must)

- Stop if Stage 09E is not accepted in the ledger.
- Add explicit readiness/capability model without removing accepted Stage 09 fields abruptly.
- Map exchange validation outcomes to product readiness:
  - trade-enabled + safe + required IP policy OK -> `effective_capability=trading`, `connection_readiness=ready_for_trading`;
  - readonly -> `effective_capability=none`, reason `read_only_not_supported`;
  - withdrawal/transfer -> `effective_capability=none`, unsafe permissions reason;
  - invalid credentials -> `effective_capability=none`;
  - missing mainnet IP restriction -> `effective_capability=none`;
  - validation unavailable -> `effective_capability=none` or explicit `needs_action`/retry state, never active-ready.
- Preserve DTO compatibility for old consumers while marking old permission fields deprecated/non-authoritative.
- Add focused tests for the readiness truth table.
- Run concrete runtime/API/DB/metrics calls listed in `runtime_acceptance`; tests alone are not acceptance.
- Create Stage 10A report and update the ledger after validation.

## Requirements (Should)

- Prefer storing readiness/capability in existing `permission_summary_json` unless additive columns are clearly safer.
- Keep labels bounded and secret-free.
- Keep implementation scoped to model/API semantics; avoid UI churn.

## Requirements (Nice-to-have)

- Include a readiness truth table in the stage report.

# Context acquisition protocol

Read only in this order:

1. `.codex/AGENTS.md`
2. Stage 10 section in the plan
3. iteration ledger
4. Stage 09E report
5. task entrypoints
6. conditional bundles only when needed

Pre-implementation reading target: `<= 12 files`.

# Reading manifest

Use front matter as the canonical reading map. Do not inspect execution/order code unless a no-order grep fails.

# Work plan (agent should follow)

1. Confirm branch is `main` and pull `origin/main` fast-forward.
2. Verify Stage 09E accepted in the ledger.
3. Inspect current permission/readiness domain and DTO mappings.
4. Implement trading capability/readiness semantics.
5. Add tests for readiness mapping and compatibility fields.
6. Create Stage 10A report and update ledger.
7. Run quality gates.
8. Direct-main commit/push and CI/deploy follow-through.

# Acceptance criteria (Definition of Done)

- Readiness truth table is test-covered and proven through concrete API/DB runtime calls.
- Read-only key cannot be represented as successful trading-ready connection.
- Public DTO has explicit readiness/capability fields.
- Deprecated permission fields remain readable but non-authoritative.
- Stage report and ledger include exact runtime command evidence or a blocked/partial reason.
- No order placement/execution code is added.
- Docs index passes.
- Ledger records evidence, contract impact, blocker status, and 10B handoff.

# Implementation constraints

## Documentation

- Create `docs/architecture/identity/exchange-connections-stage-reports/10a-trading-capability-readiness-model.md`.
- Update the shared iteration ledger before final response.

# Files to indicate (expected touched areas)

Use front matter expected touches.

# Non-goals

- UI selector removal.
- Auto-validation create/rotate.
- Data reclassification.
- Trading execution.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/migrations`
- `uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/migrations`
- `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **Capability/readiness**
3) **Runtime evidence**
4) **Контракты**
5) **Проверки**
6) **Direct-main delivery**
7) **Что дальше**
