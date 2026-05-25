---
prompt_name: identity_exchange_connections_v1_09d_e2e_cleanup_controlled_backfill
repo: roehub.com
branch: main
scope: "Stage 09D: make e2e cleanup mandatory and archive old disabled test/development exchange connections through controlled lifecycle commands, without physical deletion."

language:
  implementation: python_sql_docs_browser_optional
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, data safety, direct-main delivery"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 09 cleanup/backfill source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "must confirm 09C accepted and update 09D"
    - path: docs/architecture/identity/exchange-connections-stage-reports/09c-permission-semantics.md
      why: "permission/lifecycle handoff before cleanup"
  task_entrypoints:
    - path: src/trading/contexts/exchange_control/application/connections.py
      why: "archive lifecycle command and repository port"
      inspect_symbols: ["ExchangeConnectionService", "ExchangeConnectionRepository"]
    - path: src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py
      why: "Postgres lifecycle repository and query predicates"
      inspect_symbols: ["PostgresExchangeConnectionRepository"]
    - path: apps/api/exchange_control_client.py
      why: "facade/internal client path for cleanup through supported commands"
      inspect_symbols: ["ExchangeControlClient", "HttpExchangeControlClient"]
    - path: tests/unit/contexts/exchange_control
      why: "existing lifecycle tests to extend for cleanup/backfill safety"
  conditional_bundles:
    cleanup_tooling:
      read_when: "adding or updating a cleanup script/management command"
      paths:
        - tools
        - scripts
        - infra/macos
    api_browser_visibility:
      read_when: "asserting hidden/default list behavior through API or browser"
      paths:
        - apps/api/routes/ui_account.py
        - apps/web/dist/js/pages/settings.js
        - tests/unit/apps/api/test_ui_account_routes.py
        - tests/unit/apps/web/test_app_routes.py
    migration_or_sql:
      read_when: "adding a data migration or SQL-driven dry-run query"
      paths:
        - migrations/postgres
        - tests/unit/apps/migrations
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/08-settings-production-repair.md
      read_when: "Stage 8 record prefixes or production cleanup evidence are unclear"
    - path: docs/architecture/identity/exchange-connections-stage-reports/09b-api-ui-list-archive.md
      read_when: "default hidden/list semantics are unclear"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/09d-e2e-cleanup-controlled-backfill.md"
  canonical_shape: "stage report with dry-run/execution/evidence tables"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "09D"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_09c_must_be_accepted: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  physical_delete_forbidden: true
  cleanup_uses_supported_archive_path: true
  dry_run_before_mutation_required: true
  active_user_records_must_not_be_touched: true
  e2e_cleanup_create_validate_disable_archive_assert_hidden_required: true
  no_secret_leakage: true

task_toggles:
  implementation_changes_allowed: true
  data_cleanup_allowed_after_dry_run: true
  browser_or_api_visibility_checks_required: true
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: contract-impact-analysis
    use_when: "cleanup/backfill touches persisted rows, lifecycle semantics, or user-visible list defaults"
    timing: "before mutation and final report"
    reason: "cleanup is data-changing even when it is soft archive"
  - skill: backend-quality-gates
    use_when: "running repository/use-case tests, migration tests, lint, pyright, docs index"
    timing: "during verification"
    reason: "cleanup must be deterministic and test-covered"
  - skill: browser-qa-evidence
    use_when: "proving archived cleanup records are hidden from default /settings"
    timing: "during verification"
    reason: "the original symptom is browser-visible clutter"
  - skill: playwright
    use_when: "authenticated browser proof is feasible in the target environment"
    timing: "during verification"
    reason: "e2e cleanup must be proven with real user flow when available"
  - skill: publish-ci-deploy
    use_when: "local validation, cleanup evidence, docs and ledger are complete"
    timing: "after validation"
    reason: "direct-main delivery and post-deploy verification are required"

target_envs: ["local-dev", "browser", "mac-studio"]

required_literals:
  - "stage08_"
  - "e2e_"
  - "smoke_"
  - "dry-run"
  - "archived"
  - "exchange_connection_cleanup_total"
  - "exchange_connection_archived"

non_goals:
  - "Do not physically delete exchange connections, credential versions, or audit events."
  - "Do not archive active records."
  - "Do not hard-code a specific real user connection label into cleanup logic."
  - "Do not implement new permission semantics; that belongs to Stage 09C."
  - "Do not place orders or add execution behavior."

final_report_format:
  language: ru
  sections: ["Вердикт", "Cleanup/backfill", "Data safety", "Проверки", "Direct-main delivery", "Что дальше"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tools tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/exchange_control/application/connections.py"
  - "src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py"
  - "tests/unit/contexts/exchange_control"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/09d-e2e-cleanup-controlled-backfill.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - "tools"
  - "scripts"
  - "infra/macos"
  - "apps/api/routes/ui_account.py"
  - "apps/api/exchange_control_client.py"
  - "apps/web/dist/js/pages/settings.js"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Cleanup must be soft archive only and must leave audit/history intact."
  - "Dry-run output must show counts and redacted identifiers only; no secrets, ciphertext, HMAC, tokens, cookies, or raw exchange bodies."
  - "Manually created active records must remain untouched unless the user explicitly approves a separate operation."
---

# Task

Implement Stage 09D e2e cleanup and controlled backfill/archive for exchange connections.

Done means:

- e2e/test cleanup flow is mandatory: create -> validate or deterministic skip/failure -> disable -> archive -> assert hidden;
- old disabled development/e2e records matching the approved cleanup predicate can be archived through the supported lifecycle path;
- cleanup has dry-run and execution modes with redacted evidence;
- active records and manually created user records are not touched;
- no physical delete path is introduced;
- Stage 09D report and iteration ledger are updated;
- changes are delivered directly to `main` after validation.

## Context / Current State

- Stage 09A must have introduced the `archived` lifecycle state.
- Stage 09B must have hidden disabled/archived records from the default account list and `/settings` main table.
- Stage 09C must have clarified permission semantics so cleanup rows do not appear trade-ready by accident.
- The current production issue is stale disabled records created by development/e2e flows remaining visible or confusing unless explicitly archived.

## Requirements (Must)

- Read context using the protocol below and stop once sufficient.
- Stop if Stage 09C is not accepted in the ledger.
- Implement cleanup/backfill using repository/use-case/internal command archive behavior, not ad hoc SQL `DELETE`.
- Add dry-run mode before any data-changing cleanup.
- Cleanup predicate must be conservative:
  - label prefix `stage08_%`, `e2e_%`, or `smoke_%`;
  - lifecycle `status='disabled'`;
  - optional owner/test account and created-at window filters when available;
  - never active records;
  - never manually created active user records.
- Add tests proving non-matching records, active records, and manually created records are not archived.
- Record `exchange_connection_archived` audit events and `exchange_connection_cleanup_total` metric evidence.
- Verify default API/UI list hides archived cleanup rows.
- Create Stage 09D report and update ledger after validation and before final output.

## Requirements (Should)

- Prefer a reusable operator command/script over one-off manual SQL.
- Include `--dry-run` as the default or safest documented invocation.
- Include rollback/recovery notes: archived rows are forward-only history; recovery is explicit history view or a future unarchive design, not silent unarchive.

## Requirements (Nice-to-have)

- Include a compact cleanup predicate table in the Stage 09D report.

# Context acquisition protocol

Read only in this order:

1. `.codex/AGENTS.md`
2. Stage 09 section in the plan
3. iteration ledger
4. Stage 09C report
5. task entrypoints
6. conditional bundles only when needed

Pre-implementation reading target: `<= 12 files`.

# Reading manifest

Use front matter as the canonical reading map. Do not inspect order execution or trading code unless a grep indicates accidental coupling.

# Work plan (agent should follow)

1. Confirm current branch is `main` and pull `origin/main` fast-forward.
2. Verify Stage 09C is accepted in the ledger.
3. Inspect lifecycle archive command and current stale test/e2e record patterns.
4. Add deterministic cleanup/backfill path with dry-run first.
5. Add tests for conservative predicates, active-record safety, archive audit, metrics, and hidden default list.
6. Run dry-run and record redacted evidence.
7. If dry-run is correct, run execution only through the supported archive path and record redacted evidence.
8. Create Stage 09D report and update ledger.
9. Run quality gates.
10. Direct-main commit/push and CI/deploy follow-through.

# Acceptance criteria (Definition of Done)

- Dry-run shows only eligible disabled test/e2e/smoke records and no active records.
- Execution archives eligible rows through supported lifecycle code.
- Default API/UI list excludes archived cleanup rows.
- Audit events and cleanup metrics are present and secret-free.
- No physical delete route, SQL delete, or hard-delete behavior is added.
- Tests cover positive and negative cleanup predicates.
- Docs index passes.
- Ledger records cleanup evidence, data-safety decision, contract impact, direct-main delivery, and Stage 09E handoff.

# Implementation constraints

## Data safety

- Treat cleanup as production data mutation.
- Do not print secret-bearing fields or raw exchange responses.
- Do not assume all disabled records are test records.

## Documentation

- Create `docs/architecture/identity/exchange-connections-stage-reports/09d-e2e-cleanup-controlled-backfill.md`.
- Update the shared iteration ledger after validation and before final response.
- Keep the plan current if implementation reveals a narrower or safer cleanup predicate.

# Files to indicate (expected touched areas)

Use front matter expected touches.

# Non-goals

- Physical delete.
- Permission model changes.
- New UI design beyond required hidden/default proof.
- Trading execution.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations`
- `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tools tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations`
- `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **Cleanup/backfill**
3) **Data safety**
4) **Проверки**
5) **Direct-main delivery**
6) **Что дальше**
