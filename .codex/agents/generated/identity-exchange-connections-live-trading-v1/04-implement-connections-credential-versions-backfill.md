---
prompt_name: identity_exchange_connections_v1_04_connections_credential_versions_backfill
repo: roehub.com
branch: main
scope: "Stage 4: implement `exchange_connections`, `exchange_credential_versions`, stable `connection_id`, backfill, dual-read compatibility, and rollback evidence."

language:
  implementation: python_fastapi_sql_tests_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and migration rules"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 4 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared stage execution ledger and direct-main delivery handoff facts"
    - path: docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md
      why: "accepted Stage 3A OpenBao/Vault runtime provisioning evidence"
    - path: docs/architecture/identity/exchange-connections-stage-reports/03b-transit-application-integration.md
      why: "accepted Stage 3B Transit application integration evidence"
    - path: docs/architecture/identity/exchange-connections-stage-reports/03c-exchange-control-internal-command-api.md
      why: "accepted Stage 3C internal command API/client boundary evidence"
  task_entrypoints:
    - path: migrations/postgres/0003_identity_exchange_keys_v1.sql
      why: "legacy source table and market_type contract"
      inspect_symbols:
        - identity_exchange_keys
        - market_type
    - path: src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
      why: "legacy compatibility endpoint"
      inspect_symbols:
        - list_exchange_keys
        - create_exchange_key
        - delete_exchange_key
    - path: apps/api/routes/ui_account.py
      why: "target account endpoint surface"
      inspect_symbols:
        - account routes
        - current account summary
    - path: tests/unit/apps/api/test_identity_exchange_keys_routes.py
      why: "legacy endpoint regression tests"
      inspect_symbols:
        - list/create/delete tests
        - duplicate behavior tests
  conditional_bundles:
    exchange_control_domain:
      read_when: "creating exchange_control use cases or repositories"
      paths:
        - src/trading/contexts/exchange_control
        - tests/unit/contexts/exchange_control
    migration_chain:
      read_when: "adding new migration or backfill tests"
      paths:
        - apps/migrations/bootstrap.py
        - tests/unit/apps/migrations
    ui_account_tests:
      read_when: "adding `/api/ui/account/exchange-connections` routes"
      paths:
        - tests/unit/apps/api/test_ui_account_routes.py
        - apps/api/dto
  consult_if_needed:
    - path: docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
      read_when: "legacy secret storage compatibility is unclear"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/04-connections-credential-versions-backfill.md"
  canonical_shape: "stage report with Markdown evidence tables: migration phase, source of truth, command/SQL, expected result, rollback"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "04"
  update_required: true
  update_timing: "after validation, before direct-main push and final report"
  direct_main_delivery_required: true

hard_requirements:
  iteration_ledger_update_required: true
  previous_stage_must_be_accepted: true
  stage3a_runtime_provisioning_required: true
  stage3b_transit_application_integration_required: true
  stage3c_internal_command_api_required: true
  market_type_v1_must_remain_spot_futures: true
  connection_id_stable_required: true
  credential_versions_required: true
  dual_read_compatibility_required: true
  no_dual_write_long_term: true
  no_secret_leak_required: true
  rotate_disable_acceptance_calls_required: true
  stage_execution_ledger_update_required: true
  direct_main_push_after_validation_required: true
  feature_branch_per_stage_forbidden: true
  draft_pr_forbidden: true
  work_on_main_from_start_required: true

task_toggles:
  implement_schema: true
  implement_backfill: true
  implement_account_endpoints: true
  preserve_legacy_endpoint: true
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
    use_when: "adding persisted schema, DTOs, API endpoints, compatibility projection, or rollback"
    timing: "before implementation and final report"
    reason: "stage changes API and storage contracts"
  - skill: backend-quality-gates
    use_when: "running migration/API/domain tests, ruff, pyright"
    timing: "during verification"
    reason: "schema/backfill requires focused gates"
  - skill: production-risk-review
    use_when: "before declaring backfill/rollback safe"
    timing: "before final report"
    reason: "credential attribution and rollback are production-risk sensitive"


target_envs:
  - local-dev

required_literals:
  - "exchange_connections"
  - "exchange_credential_versions"
  - "connection_id"
  - "credential_version_id"
  - "identity_exchange_keys"
  - "spot"
  - "futures"
  - "exchange_connection_not_found"
  - "exchange_connection_not_owned"

non_goals:
  - "Do not use `linear` or `inverse` as accepted `market_type` values in v1."
  - "Do not delete `identity_exchange_keys`."
  - "Do not call Binance or Bybit."
  - "Do not implement UI completion beyond API/account surface needed for tests."
  - "Do not implement order execution."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Схема и совместимость"
    - "Backfill и rollback"
    - "Проверки"
    - "Stage 5 readiness"
    - "Direct-main delivery"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/migrations tests/unit/contexts/exchange_control tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown changes"
  - cmd: "curl -fsS -X POST \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections\" -H \"Origin: $ROEHUB_BASE_URL\" -H \"Cookie: $ROEHUB_RECENT_AUTH_SESSION_COOKIE\" -H \"X-CSRF-Token: $ROEHUB_CSRF_TOKEN\" -H \"Content-Type: application/json\" --data @fixtures/nonreal-binance-connection.json"
    expect: "creates a connection and returns connection_id without secrets"
  - cmd: "curl -fsS \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\""
    expect: "lists masked connections without secrets"
  - cmd: "curl -fsS \"$ROEHUB_BASE_URL/api/exchange-keys\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\""
    expect: "legacy compatibility projection still works"
  - cmd: "curl -fsS http://127.0.0.1:9205/internal/v1/capabilities -H \"Authorization: Bearer $ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN\" -H \"X-Roehub-Internal-Service: apps/api\" -H \"X-Request-Id: stage-4-preflight\""
    expect: "accepted Stage 3C internal boundary is reachable before create/rotate/disable evidence"
  - cmd: "rg -n \"ExchangeSecretCipher|decrypt|openbao|vault|binance|bybit|pybit|api_secret|passphrase\" apps/api || true"
    expect: "apps/api does not import direct secret/decrypt/native exchange adapters for Stage 4"
  - cmd: 'test "$(git branch --show-current)" = main'
    expect: "passes before direct-main push; otherwise stop and do not create a stage branch"
  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"

  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  - "migrations/postgres/0008_*_exchange_connections_*.sql"
  - "src/trading/contexts/exchange_control/**"
  - "apps/api/routes/ui_account.py"
  - "tests/unit/contexts/exchange_control/**"
  - "tests/unit/apps/api/test_identity_exchange_keys_routes.py"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/04-connections-credential-versions-backfill.md"

possible_secondary_touches:
  - "apps/migrations/bootstrap.py"
  - "tests/unit/apps/migrations"
  - "apps/api/dto/**"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Keep `connection_id` stable across credential rotation."
  - "Never return secret, ciphertext, fingerprint, or HMAC in legacy or new API responses."
---

# Task

Implement the Exchange Control v1 data model and compatibility migration path.

Done means:

- `exchange_connections` and `exchange_credential_versions` exist;
- legacy rows can be backfilled;
- new account endpoints expose stable `connection_id`;
- legacy `/api/exchange-keys` remains compatible;
- v1 rejects `linear`/`inverse` as `market_type`;
- Stage 4 report proves backfill, dual-read, and rollback behavior.

## Context / Current State

Stage 3A, Stage 3B, and Stage 3C must all be accepted. If any evidence report is missing, blocked, or superseded, stop.

The architecture says `identity_exchange_keys` remains as compatibility surface during migration. Do not delete it in this stage.

## Requirements (Must)

- Before making changes, verify the current branch is `main` and `git pull --ff-only origin main` succeeds; if not, stop and mark the stage blocked instead of creating a side branch.
- Update the shared stage execution ledger after validation and before delivery; include stage status, evidence, blockers, compatibility/rollback notes, CI/deploy status, and facts next stages must know.
- After all required validation passes, deliver directly to `main`: stay/switch to `main`, run `git pull --ff-only origin main`, stage only scoped files, commit on `main`, push `origin main`, and follow CI/deploy status. Do not create a per-stage branch or draft PR.
- Preserve `market_type` v1 as `spot|futures`.
- Add additive migrations for `exchange_connections` and `exchange_credential_versions`.
- Implement stable `connection_id` and replaceable `credential_version_id`.
- Implement create/rotate/disable through the accepted `exchange-control` internal command API/client boundary; `apps/api` remains a public facade and must not import secret/decrypt adapters directly.
- Implement compatibility read strategy: new tables first, fallback to `identity_exchange_keys` until fallback is explicitly retired in a later evidence report.
- Avoid long-term dual-write as a source of truth.
- Implement account endpoints for create/list/rotate/disable where this stage requires them.
- Prove create/list/rotate/disable through concrete API calls or focused route tests; rotation evidence must show stable `connection_id` and changed `credential_version_id`.
- Ensure all responses are secret-safe.
- Create Stage 4 report with SQL and API evidence.

## Requirements (Should)

- Keep old endpoint behavior stable enough for existing tests and UI.
- Add deterministic duplicate handling and ownership tests.

## Requirements (Nice-to-have)

- Include a reverse-backfill runbook sketch if rollback after phase C is non-trivial.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 3A, Stage 3B, and Stage 3C reports
3. architecture document Stage 4
4. task entrypoints
5. conditional bundles only for migration/API/domain ambiguity

Pre-implementation reading target:

- `<= 8 files`
- `<= ~50k tokens`

Stop reading once schema, repository/use-case boundaries, API DTOs, tests, and rollback report requirements are bounded.

# Reading manifest

Use front-matter `context_sources` as the canonical reading map. Do not preload all identity and strategy code.

# Work plan (agent should follow)

0. Verify the local checkout is on `main`, run `git pull --ff-only origin main`, and confirm there are no unrelated changes in scope. Stop if this cannot be proven.
Skill routing for this task:

- `contract-impact-analysis`: use for persisted schema, DTO, API, compatibility, rollback.
- `backend-quality-gates`: use during verification.
- `production-risk-review`: use before final report for migration/rollback safety.

1. Confirm Stage 3A, Stage 3B, and Stage 3C are accepted.
2. Add schema and backfill migration tests.
3. Implement exchange-control connection/credential use cases and persistence.
4. Wire account endpoints and legacy compatibility projection.
5. Run gates and create Stage 4 report.

After stage-specific verification:

- update `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` with accepted/blocked status, evidence, changed contracts, blockers, next-stage facts, and direct-main delivery status;
- perform direct-main delivery only after successful validation: confirm the current branch is `main`, fast-forward from `origin/main`, stage only scoped files, commit, push `origin main`, and watch CI/deploy status;
- if `main` cannot fast-forward, GitHub auth is unavailable, local gates fail, or unrelated worktree changes cannot be isolated, stop and mark the stage blocked in the ledger; do not create a stage branch or draft PR as a workaround.

# Acceptance criteria (Definition of Done)

- Iteration ledger is updated with facts required by the next stage.
- Creating a connection returns `connection_id`.
- List response contains masked key/status and no secret/ciphertext/HMAC.
- Rotation changes `credential_version_id` but not `connection_id`.
- `market_type=linear` or `inverse` is rejected in v1.
- SQL evidence shows legacy row count and connection row count.
- Legacy `GET /api/exchange-keys` works after backfill/compatibility projection.
- Disable and rotate flows are tested or called explicitly.
- Rollback phase notes are present in the Stage report.
- Shared ledger `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` is updated with stage status, evidence, blockers, next-stage facts, and direct-main delivery status.
- Direct-main push to `origin/main` is completed after validation and CI/deploy status is recorded, or the stage is blocked with the exact reason.
- No per-stage branch and no draft PR are created for this stage.

# Implementation constraints

## Determinism & ordering

- Backfill ordering must be deterministic.
- Generated IDs must be stable only where explicitly required; do not derive IDs from secrets.

## API / contracts

- Legacy endpoints remain until deprecation.
- New account endpoints are additive.
- No secret fields in DTOs.

## Documentation

- Update the shared stage execution ledger before direct-main delivery; it is the canonical cross-stage handoff document.
- Record direct-main delivery evidence in the ledger: commit SHA, `git push origin main` result, CI/deploy status, runtime status when applicable, or exact blocker.
- Create Stage 4 report.
- Update architecture doc only if implementation deviates.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for migration phases A-E, dual-read behavior, rollback path, API calls, SQL counts, and secret-safety evidence.
- Run docs-index check after Markdown changes.

## Tests

- Cover schema, backfill, create/list/rotate/disable, ownership, secret redaction, and unsupported market types.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`
- `migrations/postgres/0008_*_exchange_connections_*.sql`
- `src/trading/contexts/exchange_control/**`
- `apps/api/routes/ui_account.py`
- `tests/unit/contexts/exchange_control/**`
- `tests/unit/apps/api/test_identity_exchange_keys_routes.py`
- `tests/unit/apps/api/test_ui_account_routes.py`
- `docs/architecture/identity/exchange-connections-stage-reports/04-connections-credential-versions-backfill.md`

Possible secondary touches:

- `apps/migrations/bootstrap.py`
- `tests/unit/apps/migrations`
- `apps/api/dto/**`
- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/architecture/README.md`

# Non-goals

- External validation.
- UI completion.
- Open order execution.
- Removing legacy table.
- Supporting `linear|inverse` as public v1 enum values.

# Quality gates (must run and pass)

- `test "$(git branch --show-current)" = main`
- `gh --version && gh auth status`
- `uv run pytest -q tests/unit/apps/migrations tests/unit/contexts/exchange_control tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py`
- `uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`
- `curl -fsS -X POST "$ROEHUB_BASE_URL/api/ui/account/exchange-connections" -H "Origin: $ROEHUB_BASE_URL" -H "Cookie: $ROEHUB_RECENT_AUTH_SESSION_COOKIE" -H "X-CSRF-Token: $ROEHUB_CSRF_TOKEN" -H "Content-Type: application/json" --data @fixtures/nonreal-binance-connection.json`
- `curl -fsS "$ROEHUB_BASE_URL/api/ui/account/exchange-connections" -H "Cookie: $ROEHUB_SESSION_COOKIE"`
- `curl -fsS "$ROEHUB_BASE_URL/api/exchange-keys" -H "Cookie: $ROEHUB_SESSION_COOKIE"`
- `curl -fsS http://127.0.0.1:9205/internal/v1/capabilities -H "Authorization: Bearer $ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN" -H "X-Roehub-Internal-Service: apps/api" -H "X-Request-Id: stage-4-preflight"`
- `rg -n "ExchangeSecretCipher|decrypt|openbao|vault|binance|bybit|pybit|api_secret|passphrase" apps/api || true`
- `rg -n "TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE" logs output .playwright-cli || true`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

Your final message MUST include direct-main commit SHA, `git push origin main` status, CI/deploy status, and deploy/runtime status.

1. **Что реализовано**
2. **Схема и совместимость**
3. **Backfill и rollback**
4. **Проверки**
5. **Stage 5 readiness**
6. **Direct-main delivery**
