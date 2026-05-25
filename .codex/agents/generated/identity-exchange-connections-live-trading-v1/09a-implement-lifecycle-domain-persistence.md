---
prompt_name: identity_exchange_connections_v1_09a_lifecycle_domain_persistence
repo: roehub.com
branch: main
scope: "Stage 09A: add exchange connection archived lifecycle state in persistence/domain, archive command, audit event, and metrics without changing public UI behavior yet."

language:
  implementation: python_sql_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, direct-main delivery, safety rules"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 09 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "handoff ledger; must be updated after validation"
    - path: docs/architecture/identity/exchange-connections-stage-reports/08-settings-production-repair.md
      why: "latest accepted production-browser repair and cleanup gap"
  task_entrypoints:
    - path: migrations/postgres/0008_exchange_connections_v1.sql
      why: "current exchange_connections schema and constraints"
      inspect_symbols: ["exchange_connections", "exchange_credential_versions"]
    - path: src/trading/contexts/exchange_control/application/connections.py
      why: "domain lifecycle and repository port"
      inspect_symbols: ["ExchangeConnectionRecord", "ExchangeConnectionRepository", "ExchangeConnectionService"]
    - path: src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py
      why: "Postgres repository lifecycle implementation"
      inspect_symbols: ["PostgresExchangeConnectionRepository"]
    - path: src/trading/contexts/exchange_control/adapters/inbound/http/app.py
      why: "internal API capabilities and metrics registry"
      inspect_symbols: ["EXCHANGE_CONTROL_INTERNAL_CAPABILITIES", "_exchange_connection_response"]
  conditional_bundles:
    audit_schema:
      read_when: "adding or verifying exchange_connection_archived audit enum"
      paths:
        - migrations/postgres/0007_identity_exchange_audit_events_v1.sql
        - tests/unit/apps/migrations/test_identity_exchange_audit_events_sql.py
    runtime_tests:
      read_when: "archive command is exposed through exchange-control internal API in this stage"
      paths:
        - tests/unit/contexts/exchange_control/test_exchange_control_runtime.py
        - apps/api/exchange_control_client.py
  consult_if_needed:
    - path: docs/runbooks/exchange-secret-management.md
      read_when: "secret custody or OpenBao behavior becomes relevant"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/09a-lifecycle-domain-persistence.md"
  canonical_shape: "stage report with evidence tables: contract, implementation, validation, residual risk"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "09A"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_08_must_be_accepted: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  physical_delete_forbidden: true
  archive_only_disabled_connections: true
  add_exchange_connection_archived_audit_event: true
  no_secret_leakage: true

task_toggles:
  implementation_changes_allowed: true
  public_ui_behavior_change_allowed: false
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: contract-impact-analysis
    use_when: "touching schema, DTO, internal API capabilities, audit enum, or metrics"
    timing: "during design and before final report"
    reason: "lifecycle state changes affect persisted contracts"
  - skill: backend-quality-gates
    use_when: "running pytest, ruff, pyright, docs-index"
    timing: "during verification"
    reason: "backend/schema stage"
  - skill: publish-ci-deploy
    use_when: "local validation, docs, and ledger are complete"
    timing: "after validation"
    reason: "stage rollout requires direct-main delivery and CI/deploy follow-through"

target_envs: ["local-dev", "mac-studio"]

required_literals:
  - "archived"
  - "archived_at"
  - "exchange_connection_archived"
  - "exchange_connection_archive_total"
  - "physical hard delete запрещен"

non_goals:
  - "Do not change default UI list behavior in Stage 09A."
  - "Do not add public DELETE endpoint."
  - "Do not physically delete exchange connections or credential versions."
  - "Do not implement trading execution or order placement."

final_report_format:
  language: ru
  sections: ["Вердикт", "Изменения", "Контракты", "Проверки", "Direct-main delivery", "Что дальше"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - "migrations/postgres"
  - "src/trading/contexts/exchange_control/application/connections.py"
  - "src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py"
  - "src/trading/contexts/exchange_control/adapters/inbound/http/app.py"
  - "tests/unit/contexts/exchange_control"
  - "tests/unit/apps/migrations"
  - "docs/architecture/identity/exchange-connections-stage-reports/09a-lifecycle-domain-persistence.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Archive is a soft lifecycle state, not physical deletion."
  - "Do not expose secrets, ciphertext, HMAC, raw exchange responses, or user tokens in reports or logs."
---

# Task

Implement Stage 09A lifecycle foundation for exchange connections.

Done means:

- `exchange_connections` can represent `active`, `disabled`, and `archived`;
- `archived_at` and lifecycle timestamp constraints are represented and tested;
- domain/repository support archive only for owned disabled connections;
- `exchange_connection_archived` audit enum/event path and archive metrics contract exist;
- no public UI/default-list behavior is changed yet;
- Stage 09A report and iteration ledger are updated;
- changes are delivered directly to `main` after validation.

## Context / Current State

- Stage 08 accepted the production browser add-key flow but only disabled its dummy connection.
- Current implementation supports `active` and `disabled`; disabled rows still exist in the operational list.
- Stage 09 plan chooses `POST .../archive` as the v1 archive command and explicitly forbids physical hard delete.
- `exchange_connection_archived` is the Stage 09 audit event; do not reuse `exchange_connection_deleted` for archive.

## Requirements (Must)

- Read context using the protocol below and stop once sufficient.
- Verify Stage 08 is accepted in the ledger before implementation.
- Add persistence/domain support for `archived` without changing the public default list yet.
- Enforce state timestamps:
  - `active`: `disabled_at IS NULL`, `archived_at IS NULL`;
  - `disabled`: `disabled_at IS NOT NULL`, `archived_at IS NULL`;
  - `archived`: `disabled_at IS NOT NULL`, `archived_at IS NOT NULL`.
- Add archive command semantics:
  - archive disabled owned connection -> archived;
  - archive active -> deterministic rejection;
  - archive archived -> idempotent success or deterministic already-archived response;
  - rotate/validate archived -> not found/rejected.
- Add/verify `exchange_connection_archived` audit event type with redacted metadata only.
- Add metrics contract for archive/cleanup with bounded labels and no user/connection/credential labels.
- Create Stage 09A report and update the iteration ledger after validation.

## Requirements (Should)

- Keep the migration additive and idempotent for existing production DBs.
- Preserve `connection_id` and active credential version references for audit/history.
- Keep old `permissions` behavior unchanged until Stage 09C.

## Requirements (Nice-to-have)

- Include a small transition table in the Stage 09A report.

# Context acquisition protocol

Read only in this order:

1. `.codex/AGENTS.md`
2. Stage 09 section in the plan
3. iteration ledger
4. Stage 08 report
5. task entrypoints
6. conditional bundles only when needed

Pre-implementation reading target: `<= 10 files`.

# Reading manifest

Use front matter as the canonical reading map. Do not broaden into UI or Playwright implementation unless a gate fails.

# Work plan (agent should follow)

1. Confirm current branch is `main` and pull `origin/main` fast-forward.
2. Inspect current lifecycle schema/domain/repository and tests.
3. Add additive migration/schema support for `archived`.
4. Add domain/repository archive command and metrics/audit contracts.
5. Add focused tests for state transitions, constraints, and no hard delete path.
6. Create Stage 09A report and update ledger.
7. Run quality gates.
8. Direct-main commit/push and inspect CI/deploy as required by `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- Migration tests prove `archived` and timestamp constraints.
- Domain/repository tests prove allowed and rejected archive transitions.
- There is no physical delete path for exchange connections.
- Audit enum/event support includes `exchange_connection_archived`.
- Metrics contract includes archive/cleanup counters without secret-bearing labels.
- Docs index passes.
- Ledger records status, evidence, contract impact, and Stage 09B handoff.

# Implementation constraints

## API / contracts

- Do not expose public archive UI behavior in Stage 09A unless needed for internal tests.
- If adding internal command capability, keep it local-only and service-auth protected.

## Documentation

- Create the Stage 09A report.
- Update the shared ledger before final report.
- Keep all docs secret-free.

# Files to indicate (expected touched areas)

Primary touches are listed in front matter.

# Non-goals

- Public UI filter/history.
- Permission semantics.
- Cleanup old `stage08_*` rows.
- Live trading execution.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/migrations`
- `uv run ruff check src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/migrations`
- `uv run pyright src/trading/contexts/exchange_control tests/unit/contexts/exchange_control`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **Изменения**
3) **Контракты**
4) **Проверки**
5) **Direct-main delivery**
6) **Что дальше**
