---
prompt_name: identity_exchange_connections_v1_10d_non_trading_active_reclassification
repo: roehub.com
branch: main
scope: "Stage 10D: controlled reclassification/backfill of existing active exchange connections that are not trading-ready under Stage 10 semantics."

language:
  implementation: python_sql_docs_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and data safety"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 10 reclassification source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "must confirm 10C accepted and update 10D"
    - path: docs/architecture/identity/exchange-connections-stage-reports/10c-settings-trading-cjm-ui.md
      why: "accepted UI/CJM contract"
  task_entrypoints:
    - path: src/trading/contexts/exchange_control/application/connections.py
      why: "lifecycle command and readiness state"
      inspect_symbols: ["ExchangeConnectionService"]
    - path: src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py
      why: "query/update predicates for reclassification"
      inspect_symbols: ["PostgresExchangeConnectionRepository"]
    - path: tools
      why: "operator cleanup/reclassification command location"
    - path: tests/unit/contexts/exchange_control
      why: "reclassification safety tests"
  conditional_bundles:
    ops_workflow:
      read_when: "dispatching or updating Mac Studio cleanup/reclassification workflow"
      paths:
        - .github/workflows
        - infra/macos
        - scripts
    api_visibility:
      read_when: "asserting Active/History API state"
      paths:
        - apps/api/routes/ui_account.py
        - tests/unit/apps/api/test_ui_account_routes.py
        - tests/unit/apps/web/test_app_routes.py
    migrations:
      read_when: "reclassification needs additive audit event or metric persistence support"
      paths:
        - migrations/postgres
        - tests/unit/apps/migrations

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/10d-non-trading-active-reclassification.md"
  canonical_shape: "stage report with dry-run/execution/data-safety evidence"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "10D"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_10c_must_be_accepted: true
  dry_run_before_mutation_required: true
  active_non_trading_ready_only: true
  use_supported_lifecycle_path: true
  physical_delete_forbidden: true
  audit_and_metrics_required: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  no_secret_leakage: true
  concrete_runtime_calls_required: true
  tests_are_not_acceptance: true

task_toggles:
  implementation_changes_allowed: true
  data_reclassification_allowed_after_dry_run: true
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing persisted lifecycle/readiness state or account limits visibility"
    timing: "before data mutation and final report"
    reason: "Stage 10D changes production data classification"
  - skill: backend-quality-gates
    use_when: "running repository/tool/API tests and static checks"
    timing: "during verification"
    reason: "data repair must be deterministic"
  - skill: publish-ci-deploy
    use_when: "local gates, dry-run/execution evidence, docs and ledger are complete"
    timing: "after validation"
    reason: "direct-main delivery and runtime evidence required"

target_envs: ["local-dev", "mac-studio", "postgres", "prometheus"]

required_literals:
  - "dry-run"
  - "reclassified"
  - "read_only_not_supported"
  - "ready_for_trading"
  - "exchange_connection_reclassification_total"
  - "physical hard delete запрещен"

non_goals:
  - "Do not revalidate or place orders as part of reclassification."
  - "Do not archive active trading-ready connections."
  - "Do not physically delete rows."
  - "Do not design reactivate."

final_report_format:
  language: ru
  sections: ["Вердикт", "Reclassification", "Runtime evidence", "Data safety", "Проверки", "Direct-main delivery", "Что дальше"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tools tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"

runtime_acceptance:
  required: true
  commands:
    - cmd: "uv run python -m tools.exchange_connections.reclassify_non_trading_active --dry-run --json"
      expect: "dry-run prints candidate count, labels/ids redacted as needed, and reasons; no data mutation"
    - cmd: "uv run python -m tools.exchange_connections.reclassify_non_trading_active --execute --json"
      expect: "execute runs only after accepted dry-run and reclassifies exactly approved candidates through supported lifecycle/repair path"
    - cmd: "curl -fsS \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections?status=active\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" | jq -e 'all(.items[]; .connection_readiness == \"ready_for_trading\" and .effective_capability == \"trading\")'"
      expect: "Active API contains only trading-ready rows"
    - cmd: "psql \"$ROEHUB_PG_DSN\" -c \"SELECT label, status, status_reason, permission_summary_json FROM exchange_connections WHERE status='active' ORDER BY created_at DESC LIMIT 20;\""
      expect: "DB evidence excludes active non-trading-ready rows without selecting secret-bearing columns"
    - cmd: "psql \"$ROEHUB_PG_DSN\" -c \"SELECT event_type, target_id, metadata_json, created_at FROM identity_audit_events WHERE event_type IN ('exchange_connection_reclassified','exchange_connection_disabled') ORDER BY created_at DESC LIMIT 20;\""
      expect: "audit evidence exists and metadata is redacted"
    - cmd: "curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_connection_reclassification_total|exchange_connection_trading_readiness_total'"
      expect: "reclassification/readiness metrics are present"
  acceptance_rule: "Dry-run -> execute -> API/DB/audit/metrics proof is mandatory. Tests alone are not acceptance."

expected_primary_touches:
  - "tools"
  - "src/trading/contexts/exchange_control/application/connections.py"
  - "src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py"
  - "tests/unit/contexts/exchange_control"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/10d-non-trading-active-reclassification.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - ".github/workflows"
  - "infra/macos"
  - "scripts"
  - "migrations/postgres"
  - "docs/architecture/README.md"

safety_notes:
  - "Reclassification is a production data operation; dry-run evidence must precede mutation."
  - "Do not include API secrets, ciphertext, HMAC, tokens, cookies, or raw exchange responses in dry-run/execution output."
---

# Task

Implement Stage 10D controlled reclassification/backfill for existing active non-trading-ready exchange connections.

Done means:

- there is a dry-run-first tool/command/workflow that finds active rows not trading-ready under Stage 10 semantics;
- execution moves only eligible rows out of Active through supported lifecycle/repair code;
- no physical delete occurs;
- audit and metrics evidence are recorded;
- default Active API/UI contains only trading-ready rows after execution;
- Stage 10D report and ledger are updated and delivered on `main`.

## Context / Current State

- Stage 10A-10C change future semantics and UI.
- Existing production rows may still be active even though they are readonly, mismatch, invalid, or otherwise not trading-ready.
- This stage repairs existing state safely.

## Requirements (Must)

- Stop if 10C is not accepted in ledger.
- Dry-run before mutation.
- Select only active records that are not trading-ready by Stage 10 readiness fields or Stage 09 fallback evidence:
  - `permission_mismatch`;
  - `effective_permissions=read`;
  - `exchange_permissions=read`;
  - `effective_capability!=trading`;
  - `connection_readiness!=ready_for_trading`.
- Do not touch archived rows.
- Do not touch active rows already trading-ready.
- Use supported lifecycle path where possible; if a repair path is required, document why.
- Emit redacted audit/metrics.
- Prove Active list contains only trading-ready rows after reclassification.
- Run concrete dry-run, execute, API, DB, audit and metrics calls from `runtime_acceptance`; tests alone are not acceptance.
- Create Stage 10D report and update ledger.

## Requirements (Should)

- Include operator instructions for Mac Studio execution.
- Include exact rollback/recovery semantics: forward-only unless future reactivation is designed.
- Keep output redacted and count-oriented.

## Requirements (Nice-to-have)

- Include SQL evidence snippets in the report with no secret-bearing columns.

# Context acquisition protocol

Read `.codex/AGENTS.md`, Stage 10 plan, ledger, 10C report, then task entrypoints. Pre-implementation target: `<= 12 files`.

# Reading manifest

Use front matter as canonical. Do not broaden into trading execution.

# Work plan (agent should follow)

1. Confirm `main` and fast-forward pull.
2. Verify 10C accepted.
3. Implement/verify dry-run predicate.
4. Add tests for selected and non-selected rows.
5. Run dry-run and inspect redacted evidence.
6. Execute only after dry-run is correct.
7. Prove Active-only trading-ready state through API/DB.
8. Create Stage 10D report and update ledger.
9. Run gates.
10. Direct-main delivery and runtime evidence.

# Acceptance criteria (Definition of Done)

- Dry-run count and selected reasons are documented.
- Execution count matches approved candidates.
- No active trading-ready rows are changed.
- No physical delete path is added.
- Audit and metrics are present.
- Active list excludes non-trading-ready rows.
- Stage report and ledger include dry-run, execute, API, DB, audit and metrics evidence.
- Docs/ledger updated.

# Implementation constraints

## Documentation

- Create `docs/architecture/identity/exchange-connections-stage-reports/10d-non-trading-active-reclassification.md`.
- Update ledger before final output.

# Files to indicate (expected touched areas)

Use front matter expected touches.

# Non-goals

Reactivate, order execution, broad cleanup, physical delete.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations`
- `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tools tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations`
- `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **Reclassification**
3) **Runtime evidence**
4) **Data safety**
5) **Проверки**
6) **Direct-main delivery**
7) **Что дальше**
