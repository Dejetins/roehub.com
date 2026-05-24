---
prompt_name: identity_exchange_connections_v1_03c_exchange_control_internal_command_api
repo: roehub.com
branch: main
scope: "Stage 3C: add the local-only internal command API and apps/api client boundary between apps/api and the supervised exchange-control service before schema/backfill or exchange validation."

language:
  implementation: python_fastapi_http_security_tests_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, service boundary, secret safety, direct-main delivery rules"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 3C source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared stage execution ledger and Stage 3B handoff facts"
    - path: docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md
      why: "accepted Stage 2 runtime/service identity evidence"
    - path: docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md
      why: "accepted Stage 3A secret backend evidence"
    - path: docs/architecture/identity/exchange-connections-stage-reports/03b-transit-application-integration.md
      why: "accepted Stage 3B application secret boundary evidence"
  task_entrypoints:
    - path: apps/exchange_control
      why: "supervised service entrypoint that needs internal command routes"
      inspect_symbols:
        - create_app
        - ExchangeControlRuntimeConfig
        - local bind contract
    - path: src/trading/contexts/exchange_control/adapters/inbound/http/app.py
      why: "current exchange-control FastAPI app exposes only health and metrics"
      inspect_symbols:
        - create_exchange_control_app
        - /health/ready
        - /metrics
    - path: apps/api/routes/ui_account.py
      why: "future public account routes must call exchange-control through a client, not in-process secret code"
      inspect_symbols:
        - account routes
        - exchange connection route placeholders
    - path: tests/unit/contexts/exchange_control/test_exchange_control_runtime.py
      why: "existing service runtime tests to extend for internal API auth/capabilities"
      inspect_symbols:
        - TestClient
        - config validation
  conditional_bundles:
    app_wiring:
      read_when: "apps/api client wiring or product config fail-closed behavior is unclear"
      paths:
        - apps/api/wiring/modules
        - apps/api/main/app.py
        - tests/unit/apps/api
    secret_boundary:
      read_when: "no-direct-import assertions need exact Stage 3B symbols"
      paths:
        - src/trading/contexts/exchange_control
        - docs/runbooks/exchange-secret-management.md
    ops_runtime:
      read_when: "runtime port/bind/launchd/Monit changes are required"
      paths:
        - infra/macos/launchd/com.roehub.exchange-control.plist
        - infra/scripts/monit/roehub-exchange-control.monitrc
        - infra/macos/prometheus/prometheus.prod.yml
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md
      read_when: "recent-auth/session envelope requirements are unclear"
    - path: docs/architecture/README.md
      read_when: "docs index or architecture navigation must be updated"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md"
    - "docs/runbooks/exchange-secret-management.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/03c-exchange-control-internal-command-api.md"
  canonical_shape: "stage report with Markdown evidence tables: boundary, endpoint/call, expected result, observed result, blocker"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "03C"
  update_required: true
  update_timing: "after validation, before direct-main push and final report"
  direct_main_delivery_required: true

hard_requirements:
  iteration_ledger_update_required: true
  previous_stage_must_be_accepted: true
  stage3a_must_be_accepted: true
  stage3b_must_be_accepted: true
  local_only_internal_api_required: true
  apps_api_client_boundary_required: true
  service_to_service_auth_required: true
  missing_internal_auth_fail_closed_required: true
  no_apps_api_direct_secret_import_required: true
  no_apps_api_exchange_sdk_import_required: true
  no_business_create_rotate_validate_yet: true
  no_secret_leak_required: true
  stage_execution_ledger_update_required: true
  direct_main_push_after_validation_required: true
  feature_branch_per_stage_forbidden: true
  draft_pr_forbidden: true
  work_on_main_from_start_required: true

task_toggles:
  implement_internal_capabilities_endpoint: true
  implement_internal_auth_guard: true
  implement_apps_api_client_port: true
  implement_product_config_fail_closed: true
  publish_after_success: true
  direct_main_push_after_validation: true
  target_branch: main
  draft_pr_after_success: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding internal HTTP contract, service auth env vars, apps/api client config, error envelope, timeout/retry policy"
    timing: "before implementation and final report"
    reason: "Stage 3C changes internal service and config contracts"
  - skill: backend-quality-gates
    use_when: "running exchange-control/apps-api tests, ruff, pyright, docs gates"
    timing: "during verification"
    reason: "internal API boundary needs deterministic backend gates"
  - skill: production-risk-review
    use_when: "before declaring apps/api no-direct-secret-import and internal auth fail-closed complete"
    timing: "before final report"
    reason: "credential custody service boundary is production-risk sensitive"
  - skill: publish-ci-deploy
    use_when: "stage implementation, validation, stage report, and ledger update are complete"
    timing: "after validation and before final report"
    reason: "user requires direct push to main after accepted validation, with CI/deploy follow-through"

target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN"
  - "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL"
  - "/internal/v1/capabilities"
  - "X-Roehub-Internal-Service"
  - "X-Request-Id"
  - "apps/api"
  - "exchange-control"
  - "127.0.0.1:9205"

non_goals:
  - "Do not implement exchange connection schema/backfill."
  - "Do not implement create/rotate/disable business handlers yet."
  - "Do not implement Binance/Bybit validation."
  - "Do not expose Transit decrypt path or native exchange SDKs to apps/api."
  - "Do not implement order execution."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Internal API boundary"
    - "Service auth и fail-closed"
    - "No-direct-import evidence"
    - "Проверки"
    - "Stage 4 readiness"
    - "Direct-main delivery"

quality_gates:
  - cmd: 'test "$(git branch --show-current)" = main'
    expect: "passes before direct-main push; otherwise stop and do not create a stage branch"
  - cmd: "git pull --ff-only origin main"
    expect: "passes before making delivery changes"
  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py"
    expect: "passes; add focused tests for internal auth/capabilities/client config"
  - cmd: "uv run ruff check apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "uv run pyright apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown changes"
  - cmd: "curl -fsS http://127.0.0.1:9205/health/ready"
    expect: "exchange-control remains ready"
  - cmd: "curl -fsS http://127.0.0.1:9205/internal/v1/capabilities -H \"Authorization: Bearer $ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN\" -H \"X-Roehub-Internal-Service: apps/api\" -H \"X-Request-Id: stage-3c-smoke\""
    expect: "authenticated internal call returns sanitized capabilities and service identity"
  - cmd: "curl -i http://127.0.0.1:9205/internal/v1/capabilities -H \"X-Roehub-Internal-Service: apps/api\""
    expect: "missing token is denied with 401/403"
  - cmd: "rg -n \"ExchangeSecretCipher|decrypt|openbao|vault|binance|bybit|pybit|api_secret|passphrase\" apps/api || true"
    expect: "no direct secret/decrypt/native exchange adapter imports in apps/api; only allowed literal references must be justified in report"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  - "apps/exchange_control/**"
  - "src/trading/contexts/exchange_control/adapters/inbound/http/**"
  - "apps/api/**"
  - "tests/unit/contexts/exchange_control/**"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/03c-exchange-control-internal-command-api.md"

possible_secondary_touches:
  - "infra/macos/launchd/com.roehub.exchange-control.plist"
  - "infra/scripts/monit/roehub-exchange-control.monitrc"
  - "docs/runbooks/exchange-secret-management.md"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Do not commit service tokens, session cookies, request bodies containing API secrets, passphrases, ciphertext, HMAC, fingerprints, or raw provider responses."
  - "Internal capabilities output must not contain user, connection, credential, or secret material."
---

# Task

Implement the `apps/api -> exchange-control` internal command API boundary before Stage 4 schema/backfill and Stage 5 exchange validation.

Done means:

- `exchange-control` exposes a local-only authenticated internal API surface;
- `apps/api` has an outbound client/port for `exchange-control`;
- missing/invalid internal auth fails closed;
- capabilities/contract smoke proves the runtime path;
- `apps/api` does not import Transit/decrypt/native exchange adapters directly;
- Stage 3C report and the shared iteration ledger make Stage 4 safe to start.

## Context / Current State

Stage 2 accepted a supervised `exchange-control` process, but the current app only exposes `/health/ready` and `/metrics`. The plan requires real secret-bearing write and validation operations to run behind the `exchange-control` boundary. Without Stage 3C, Stage 4/5 could accidentally implement create/rotate/validate inside `apps/api`, which would break the service isolation goal.

If Stage 3A or Stage 3B evidence is missing, blocked, or not accepted, stop before implementation and mark Stage 3C blocked.

This stage creates the internal command boundary and client contract. It does not implement the business handlers for create/rotate/disable/validate.

## Requirements (Must)

- Before making changes, verify the current branch is `main` and `git pull --ff-only origin main` succeeds; if not, stop and mark the stage blocked instead of creating a side branch.
- Confirm Stage 3A and Stage 3B are accepted.
- Update the shared stage execution ledger after validation and before delivery; include stage status, evidence, blockers, compatibility/rollback notes, CI/deploy status, and facts Stage 4/5 must know.
- After all required validation passes, deliver directly to `main`: stay/switch to `main`, run `git pull --ff-only origin main`, stage only scoped files, commit on `main`, push `origin main`, and follow CI/deploy status. Do not create a per-stage branch or draft PR.
- Add a local-only internal API namespace on `exchange-control`, with `GET /internal/v1/capabilities` as the concrete smoke endpoint.
- Add service-to-service auth for `apps/api -> exchange-control` using `ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN`; missing/invalid token must fail closed.
- Add `X-Roehub-Internal-Service: apps/api` and `X-Request-Id` handling for internal calls.
- Add an `apps/api` outbound client/port and product config for `ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL`.
- Product-ready config must fail closed when exchange connection public routes are enabled without internal base URL/token.
- Define sanitized error envelope and timeout/retry policy.
- Prove `apps/api` does not directly import or wire Transit/decrypt/native exchange adapters.
- Create Stage 3C evidence report.

## Requirements (Should)

- Keep internal endpoint responses compact and secret-free.
- Keep internal command naming stable enough for Stage 4/5 to add handlers without route churn.
- Prefer deterministic fake client tests in `apps/api`.

## Requirements (Nice-to-have)

- Add idempotency-key validation helpers for future mutating commands if it can be done narrowly.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 3A and Stage 3B reports
3. architecture document Stage 3C
4. task entrypoints
5. conditional bundles only if config, wiring, or runtime evidence is unclear

Pre-implementation reading target:

- `<= 8 files`
- `<= ~45k tokens`

Stop reading once internal endpoint, auth guard, apps/api client, config, tests, report, and ledger surfaces are bounded.

# Reading manifest

Use front-matter `context_sources` as the canonical reading map. Do not preload strategy, execution, or external exchange modules.

# Work plan (agent should follow)

0. Verify the local checkout is on `main`, run `git pull --ff-only origin main`, and confirm there are no unrelated changes in scope. Stop if this cannot be proven.

Skill routing for this task:

- `contract-impact-analysis`: use for internal API, env/config, timeout/retry, and error envelope contracts.
- `backend-quality-gates`: use during verification.
- `production-risk-review`: use before claiming apps/api isolation is complete.
- `publish-ci-deploy`: use only after validation, report, and ledger update are complete.

1. Confirm Stage 3A and Stage 3B accepted.
2. Add internal auth/config model and capabilities endpoint to `exchange-control`.
3. Add `apps/api` client/port and product fail-closed config.
4. Add no-direct-import assertions/tests.
5. Run local runtime acceptance calls.
6. Create Stage 3C report and update the iteration ledger.

After stage-specific verification:

- update `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` with accepted/blocked status, evidence, changed contracts, blockers, next-stage facts, and direct-main delivery status;
- perform direct-main delivery only after successful validation: confirm the current branch is `main`, fast-forward from `origin/main`, stage only scoped files, commit, push `origin main`, and watch CI/deploy status;
- if `main` cannot fast-forward, GitHub auth is unavailable, local gates fail, Stage 3A/3B evidence is missing, runtime internal API evidence cannot be collected, or unrelated worktree changes cannot be isolated, stop and mark the stage blocked in the ledger; do not create a stage branch or draft PR as a workaround.

# Acceptance criteria (Definition of Done)

- Iteration ledger is updated with facts required by Stage 4/5.
- Stage 3C report exists at `docs/architecture/identity/exchange-connections-stage-reports/03c-exchange-control-internal-command-api.md`.
- `GET /internal/v1/capabilities` succeeds with valid internal auth and returns no secrets.
- Missing/invalid internal auth returns `401`/`403`.
- `apps/api` has an exchange-control client/port and product fail-closed config.
- Grep/test evidence proves `apps/api` has no direct Transit/decrypt/native exchange adapter imports.
- No create/rotate/disable/validate business handlers are implemented yet except contract scaffolding/capabilities.
- Shared ledger is updated with status, evidence, blockers, next-stage facts, and direct-main delivery status.
- Direct-main push to `origin/main` is completed after validation and CI/deploy status is recorded, or the stage is blocked with the exact reason.
- No per-stage branch and no draft PR are created for this stage.

# Implementation constraints

## Determinism & ordering

- Stage 3C must not start until Stage 3A and Stage 3B are accepted.
- Tests must not depend on live Binance/Bybit or user credentials.
- Internal auth tests must be deterministic.

## API / contracts

- Public API behavior should remain unchanged in Stage 3C.
- Internal API is additive and local-only.
- Do not expose secrets, raw request bodies, raw exception text, user ids in metric labels, connection ids in metric labels, or credential ids in metric labels.

## Documentation

- Update the shared stage execution ledger before direct-main delivery; it is the canonical cross-stage handoff document.
- Record direct-main delivery evidence in the ledger: commit SHA, `git push origin main` result, CI/deploy status, runtime status when applicable, or exact blocker.
- Create Stage 3C report.
- Update architecture only if implementation deviates from the plan.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for internal endpoint, service auth, apps/api client, no-direct-import, runtime call, and blocker evidence.
- Run docs-index check after Markdown changes.

## Tests

- Cover capabilities endpoint success.
- Cover missing/invalid internal token.
- Cover product config fail-closed.
- Cover `apps/api` fake client behavior where feasible.
- Cover no direct secret/exchange imports with a deterministic test or documented grep.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`
- `apps/exchange_control/**`
- `src/trading/contexts/exchange_control/adapters/inbound/http/**`
- `apps/api/**`
- `tests/unit/contexts/exchange_control/**`
- `tests/unit/apps/api/test_ui_account_routes.py`
- `docs/architecture/identity/exchange-connections-stage-reports/03c-exchange-control-internal-command-api.md`

Possible secondary touches:

- `infra/macos/launchd/com.roehub.exchange-control.plist`
- `infra/scripts/monit/roehub-exchange-control.monitrc`
- `docs/runbooks/exchange-secret-management.md`
- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/architecture/README.md`

# Non-goals

- Connection tables.
- Backfill.
- External exchange validation.
- UI.
- Trading execution.

# Quality gates (must run and pass)

- `test "$(git branch --show-current)" = main`
- `git pull --ff-only origin main`
- `gh --version && gh auth status`
- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py`
- `uv run ruff check apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api`
- `uv run pyright apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`
- `curl -fsS http://127.0.0.1:9205/health/ready`
- `curl -fsS http://127.0.0.1:9205/internal/v1/capabilities -H "Authorization: Bearer $ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN" -H "X-Roehub-Internal-Service: apps/api" -H "X-Request-Id: stage-3c-smoke"`
- `curl -i http://127.0.0.1:9205/internal/v1/capabilities -H "X-Roehub-Internal-Service: apps/api"`
- `rg -n "ExchangeSecretCipher|decrypt|openbao|vault|binance|bybit|pybit|api_secret|passphrase" apps/api || true`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

Your final message MUST include direct-main commit SHA, `git push origin main` status, CI/deploy status, and deploy/runtime status.

1. **Что реализовано**
2. **Internal API boundary**
3. **Service auth и fail-closed**
4. **No-direct-import evidence**
5. **Проверки**
6. **Stage 4 readiness**
7. **Direct-main delivery**
