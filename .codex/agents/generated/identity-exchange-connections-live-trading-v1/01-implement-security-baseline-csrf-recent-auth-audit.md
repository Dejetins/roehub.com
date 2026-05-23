---
prompt_name: identity_exchange_connections_v1_01_security_baseline_csrf_recent_auth_audit
repo: roehub.com
branch: main
scope: "Stage 1: harden current exchange-key mutations with CSRF fail-closed, Keycloak recent-auth, and exchange audit event schema."

language:
  implementation: python_fastapi_sql_tests_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and gates"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 1 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared stage execution ledger and direct-main delivery handoff facts"
    - path: docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md
      why: "accepted Stage 0 evidence"
  task_entrypoints:
    - path: src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
      why: "legacy mutation routes to harden"
      inspect_symbols:
        - create_exchange_key
        - delete_exchange_key
    - path: apps/api/routes/ui_account.py
      why: "current same-origin and account audit patterns"
      inspect_symbols:
        - _enforce_same_origin
        - _record_audit_event
    - path: migrations/postgres/0006_identity_account_settings_v1.sql
      why: "current audit event CHECK constraint"
      inspect_symbols:
        - identity_audit_events_type_check
        - identity_audit_events
    - path: tests/unit/apps/api/test_identity_exchange_keys_routes.py
      why: "focused route behavior coverage"
      inspect_symbols:
        - exchange key create tests
        - exchange key delete tests
  conditional_bundles:
    keycloak_recent_auth:
      read_when: "recent-auth session semantics are unclear"
      paths:
        - docs/architecture/identity/identity-keycloak-auth-model-v1.md
        - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
    account_tests:
      read_when: "ui account route behavior or audit helpers are touched"
      paths:
        - tests/unit/apps/api/test_ui_account_routes.py
    migration_tests:
      read_when: "audit event schema migration is added"
      paths:
        - tests/unit/apps/migrations
        - apps/migrations/bootstrap.py
  consult_if_needed:
    - path: docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
      read_when: "secret policy details are ambiguous"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md"
    - "docs/architecture/identity/identity-keycloak-auth-model-v1.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md"
  canonical_shape: "stage report with Markdown evidence tables: mutation scenario, expected result, observed result, DB/audit evidence, blocker"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "01"
  update_required: true
  update_timing: "after validation, before direct-main push and final report"
  direct_main_delivery_required: true

hard_requirements:
  iteration_ledger_update_required: true
  previous_stage_must_be_accepted: true
  csrf_fail_closed_required: true
  recent_auth_required_for_secret_mutations: true
  audit_schema_exchange_events_required: true
  no_secret_leak_required: true
  product_ready_dev_kek_fail_closed_required: true
  exact_acceptance_calls_required: true
  stage_execution_ledger_update_required: true
  direct_main_push_after_validation_required: true
  feature_branch_per_stage_forbidden: true
  draft_pr_forbidden: true
  work_on_main_from_start_required: true

task_toggles:
  implement_code_changes: true
  implement_migration: true
  update_docs: true
  publish_after_success: true
  direct_main_push_after_validation: true
  target_branch: main
  draft_pr_after_success: false

skill_routing:
  - skill: publish-ci-deploy
    use_when: "stage implementation, validation, stage report, and ledger update are complete"
    timing: "after validation and before final report"
    reason: "user requires direct push to main after accepted validation, with CI/deploy follow-through"
  - skill: contract-impact-analysis
    use_when: "changing API mutation behavior, audit schema, or auth requirements"
    timing: "before implementation and final report"
    reason: "security hardening changes public and persisted contracts"
  - skill: backend-quality-gates
    use_when: "running API, migration, ruff, pyright gates"
    timing: "during verification"
    reason: "backend/security stage requires focused gates"
  - skill: root-cause-debugging
    use_when: "focused security tests fail unexpectedly"
    timing: "if blocker"
    reason: "avoid patching symptoms around auth/security"


target_envs:
  - local-dev

required_literals:
  - "recent_auth_required"
  - "csrf_required"
  - "exchange_key_created"
  - "exchange_key_deleted"
  - "exchange_connection_created"
  - "exchange_connection_validated"
  - "exchange_connection_validation_failed"
  - "exchange_credential_rotated"
  - "exchange_connection_disabled"
  - "exchange_connection_deleted"

non_goals:
  - "Do not create exchange_connections or exchange_credential_versions in this stage."
  - "Do not add OpenBao/Vault Transit."
  - "Do not call Binance or Bybit."
  - "Do not implement exchange-execution or order placement."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Security contract"
    - "Audit и миграции"
    - "Проверки"
    - "Stage 2 readiness"
    - "Direct-main delivery"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/apps/migrations"
    expect: "passes if migration/bootstrap changed"
  - cmd: "uv run ruff check apps/api src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/migrations"
    expect: "passes for touched paths"
  - cmd: "uv run pyright apps/api src/trading/contexts/identity tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown changes"
  - cmd: 'test "$(git branch --show-current)" = main'
    expect: "passes before direct-main push; otherwise stop and do not create a stage branch"
  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"

  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  - "src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py"
  - "apps/api/routes/ui_account.py"
  - "migrations/postgres/0007_*_exchange_audit_events_*.sql"
  - "tests/unit/apps/api/test_identity_exchange_keys_routes.py"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md"

possible_secondary_touches:
  - "apps/migrations/bootstrap.py"
  - "tests/unit/apps/migrations"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Fail closed on missing CSRF/recent-auth for secret mutations."
  - "Audit metadata must never contain API secrets, ciphertext, HMAC, or raw exchange errors."
---

# Task

Implement Stage 1 security baseline for the existing exchange-key mutation surface.

Done means:

- secret-bearing mutations fail closed without valid CSRF/same-origin protection;
- add/delete exchange-key mutations require recent-auth;
- audit schema accepts explicit exchange event types;
- audit events are emitted without secrets;
- Stage 1 evidence report is created and Stage 2 can start.

## Context / Current State

Stage 0 must already be accepted. If `docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md` is missing or says Stage 0 is blocked, stop and report that Stage 1 cannot start.

Current known gap: `apps/api/routes/ui_account.py` has a same-origin guard pattern, but Stage 1 requires fail-closed behavior for exchange mutations and deterministic `recent_auth_required` / `csrf_required` outcomes.

## Requirements (Must)

- Before making changes, verify the current branch is `main` and `git pull --ff-only origin main` succeeds; if not, stop and mark the stage blocked instead of creating a side branch.
- Update the shared stage execution ledger after validation and before delivery; include stage status, evidence, blockers, compatibility/rollback notes, CI/deploy status, and facts next stages must know.
- After all required validation passes, deliver directly to `main`: stay/switch to `main`, run `git pull --ff-only origin main`, stage only scoped files, commit on `main`, push `origin main`, and follow CI/deploy status. Do not create a per-stage branch or draft PR.
- Require CSRF fail-closed for browser mutations carrying exchange credentials.
- Reject mutation without valid CSRF when both `Origin` and `Referer` are absent.
- Reject cross-origin mutation.
- Require Keycloak recent-auth for add/delete legacy exchange keys and future add/rotate/delete/disable hooks.
- Extend `identity_audit_events` event type constraint for the listed `exchange_*` events.
- Emit audit events for create/delete without secrets.
- Add focused tests for success and rejection cases.
- Create `docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md`.

## Requirements (Should)

- Keep route error responses deterministic and redacted.
- Use existing identity/account helper patterns where practical.

## Requirements (Nice-to-have)

- Add helper abstractions only if they reduce duplicate auth/CSRF logic across current and future exchange routes.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 0 report
3. architecture document Stage 1
4. task entrypoints
5. conditional bundles only for touched contracts or failing checks

Pre-implementation reading target:

- `<= 8 files`
- `<= ~40k tokens`

Stop reading once CSRF hook point, recent-auth source, audit migration shape, and focused tests are bounded.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not eagerly preload all listed files.

# Work plan (agent should follow)

0. Verify the local checkout is on `main`, run `git pull --ff-only origin main`, and confirm there are no unrelated changes in scope. Stop if this cannot be proven.
Skill routing for this task:

- `contract-impact-analysis`: use before implementation and final report for API/auth/schema impact.
- `backend-quality-gates`: use during verification for tests/lint/type/docs.
- `root-cause-debugging`: use only if security tests fail unexpectedly.

1. Confirm Stage 0 accepted.
2. Design the smallest CSRF/recent-auth enforcement point for exchange mutations.
3. Add audit event type migration and tests.
4. Add route/use-case audit emission without secret values.
5. Run focused gates and create the Stage 1 report.

After stage-specific verification:

- update `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` with accepted/blocked status, evidence, changed contracts, blockers, next-stage facts, and direct-main delivery status;
- perform direct-main delivery only after successful validation: confirm the current branch is `main`, fast-forward from `origin/main`, stage only scoped files, commit, push `origin main`, and watch CI/deploy status;
- if `main` cannot fast-forward, GitHub auth is unavailable, local gates fail, or unrelated worktree changes cannot be isolated, stop and mark the stage blocked in the ledger; do not create a stage branch or draft PR as a workaround.

# Acceptance criteria (Definition of Done)

- Iteration ledger is updated with facts required by the next stage.
- No-Origin/no-CSRF mutation is rejected.
- Cross-origin mutation is rejected.
- Same-origin/CSRF mutation without recent-auth returns `recent_auth_required`.
- Same-origin/CSRF/recent-auth mutation succeeds.
- Audit CHECK constraint includes required `exchange_*` event types.
- Audit records contain no secret-like values.
- Stage report includes exact curl, SQL, grep, and test evidence.
- Product/live-ready config with a dev-only KEK is rejected or explicitly blocked with evidence; it cannot be left as an unverified TODO.
- Shared ledger `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` is updated with stage status, evidence, blockers, next-stage facts, and direct-main delivery status.
- Direct-main push to `origin/main` is completed after validation and CI/deploy status is recorded, or the stage is blocked with the exact reason.
- No per-stage branch and no draft PR are created for this stage.

# Implementation constraints

## Determinism & ordering

- Keep error codes stable and deterministic.
- Keep migration ordering deterministic.

## API / contracts

- This is a deliberate compatible security hardening.
- Do not remove legacy `/api/exchange-keys`.

## Documentation

- Update the shared stage execution ledger before direct-main delivery; it is the canonical cross-stage handoff document.
- Record direct-main delivery evidence in the ledger: commit SHA, `git push origin main` result, CI/deploy status, runtime status when applicable, or exact blocker.
- Create the Stage 1 report.
- Update the architecture document only if implementation must deviate from it.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for the Stage 1 scenario matrix: no Origin/Referer, cross-origin, same-origin without recent-auth, same-origin with recent-auth, audit schema, dev-only KEK.
- Run docs-index check after Markdown changes.

## Tests

- Add direct API tests for CSRF/recent-auth/audit outcomes.
- Add migration tests if the migration chain requires it.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`
- `src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py`
- `apps/api/routes/ui_account.py`
- `migrations/postgres/0007_*_exchange_audit_events_*.sql`
- `tests/unit/apps/api/test_identity_exchange_keys_routes.py`
- `tests/unit/apps/api/test_ui_account_routes.py`
- `docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md`

Possible secondary touches:

- `apps/migrations/bootstrap.py`
- `tests/unit/apps/migrations`
- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/architecture/README.md`

# Non-goals

- New connection model.
- Transit/OpenBao.
- External exchange validation.
- UI redesign.
- Order execution.

# Quality gates (must run and pass)

- `test "$(git branch --show-current)" = main`
- `gh --version && gh auth status`
- `uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py`
- `uv run pytest -q tests/unit/apps/migrations` if migration/bootstrap changed
- `uv run ruff check apps/api src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/migrations`
- `uv run pyright apps/api src/trading/contexts/identity tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`
- `rg -n "TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE" logs output .playwright-cli || true`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

Your final message MUST include direct-main commit SHA, `git push origin main` status, CI/deploy status, and deploy/runtime status.

1. **Что реализовано**
2. **Security contract**
3. **Audit и миграции**
4. **Проверки**
5. **Stage 2 readiness**
6. **Direct-main delivery**
