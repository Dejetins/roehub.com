---
prompt_name: identity_exchange_connections_v1_03b_transit_application_integration
repo: roehub.com
branch: main
scope: "Stage 3B: integrate exchange-control application code with the accepted OpenBao/Vault Transit runtime through secret cipher ports, fail-closed config, redaction, tests, and repeated ACL evidence."

language:
  implementation: python_security_config_tests_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and secret safety"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 3B source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared stage execution ledger and Stage 3A handoff facts"
    - path: docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md
      why: "accepted Stage 2 service identity and runtime boundary evidence"
    - path: docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md
      why: "accepted Stage 3A OpenBao/Vault runtime and Transit ACL evidence"
  task_entrypoints:
    - path: src/trading/contexts/exchange_control
      why: "new secret boundary module"
      inspect_symbols:
        - ExchangeSecretCipher
        - service identity
        - credential secret DTOs
    - path: apps/exchange_control
      why: "runtime config and app startup boundary"
      inspect_symbols:
        - ExchangeControlRuntimeConfig
        - health/readiness
    - path: tests/unit/contexts/exchange_control
      why: "focused secret/config tests"
      inspect_symbols:
        - secret cipher tests
        - config fail-closed tests
    - path: docs/runbooks/exchange-secret-management.md
      why: "Stage 3A runbook to update with application integration commands"
      inspect_symbols:
        - Transit env contract
        - rotation
        - emergency disable
  conditional_bundles:
    existing_secret_policy:
      read_when: "current encryption, fingerprint, or redaction policy is ambiguous"
      paths:
        - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
        - migrations/postgres/0004_identity_exchange_keys_v2.sql
    config_wiring:
      read_when: "product-ready fail-closed config wiring is unclear"
      paths:
        - apps/api
        - apps/exchange_control
        - src/trading/config
    migration_tests:
      read_when: "schema or bootstrap compatibility changes become necessary"
      paths:
        - tests/unit/apps/migrations
        - apps/migrations/bootstrap.py
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md
      read_when: "audit/recent-auth security baseline affects secret error reporting"
    - path: infra
      read_when: "runtime env injection contract is unclear after Stage 3A"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/runbooks/exchange-secret-management.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/03b-transit-application-integration.md"
  canonical_shape: "stage report with Markdown evidence tables: component, secret operation, command/test, expected result, observed result, blocker"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "03B"
  update_required: true
  update_timing: "after validation, before direct-main push and final report"
  direct_main_delivery_required: true

hard_requirements:
  iteration_ledger_update_required: true
  previous_stage_must_be_accepted: true
  stage3a_must_be_accepted: true
  transit_acl_after_service_identity: true
  implement_secret_cipher_port_required: true
  implement_transit_adapter_required: true
  api_process_no_decrypt: true
  exchange_control_limited_decrypt: true
  product_ready_fail_closed_without_transit: true
  runtime_transit_acl_evidence_required: true
  no_secret_leak_required: true
  stage_execution_ledger_update_required: true
  direct_main_push_after_validation_required: true
  feature_branch_per_stage_forbidden: true
  draft_pr_forbidden: true
  work_on_main_from_start_required: true

task_toggles:
  implement_secret_cipher_port: true
  implement_transit_adapter: true
  implement_dev_fallback_guarded: true
  update_secret_runbook: true
  publish_after_success: true
  direct_main_push_after_validation: true
  target_branch: main
  draft_pr_after_success: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing config, secret cipher, service capabilities, DTO redaction, or storage metadata"
    timing: "before implementation and final report"
    reason: "secret boundary is a production contract"
  - skill: backend-quality-gates
    use_when: "running secret/config tests, ruff, pyright, docs gates"
    timing: "during verification"
    reason: "security implementation needs deterministic gates"
  - skill: production-risk-review
    use_when: "before declaring Transit application integration complete"
    timing: "before final report"
    reason: "credential custody is production-risk sensitive"
  - skill: publish-ci-deploy
    use_when: "stage implementation, validation, stage report, and ledger update are complete"
    timing: "after validation and before final report"
    reason: "user requires direct push to main after accepted validation, with CI/deploy follow-through"

target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "openbao_transit_v1"
  - "vault_transit_v1"
  - "OPENBAO_ADDR"
  - "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN"
  - "ROEHUB_API_TRANSIT_TOKEN"
  - "roehub-exchange-credentials"
  - "ExchangeSecretCipher"

non_goals:
  - "Do not provision OpenBao/Vault runtime in Stage 3B; Stage 3A owns that."
  - "Do not migrate identity_exchange_keys into new tables yet."
  - "Do not implement Binance/Bybit validation."
  - "Do not expose plaintext credentials to apps/api."
  - "Do not implement order execution."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Secret boundary"
    - "ACL и fail-closed"
    - "Проверки"
    - "Stage 3C readiness"
    - "Direct-main delivery"

quality_gates:
  - cmd: 'test "$(git branch --show-current)" = main'
    expect: "passes before direct-main push; otherwise stop and do not create a stage branch"
  - cmd: "git pull --ff-only origin main"
    expect: "passes before making delivery changes"
  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/migrations"
    expect: "passes; create focused tests if needed"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown changes"
  - cmd: "curl -fsS \"$OPENBAO_ADDR/v1/sys/health\""
    expect: "accepted Stage 3A runtime is still healthy"
  - cmd: "curl -fsS -X POST \"$OPENBAO_ADDR/v1/transit/encrypt/roehub-exchange-credentials\" -H \"X-Vault-Token: $ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN\" --data '{\"plaintext\":\"U1RBR0UzQl9TTU9LRQ==\"}'"
    expect: "exchange-control identity can encrypt"
  - cmd: "curl -i -X POST \"$OPENBAO_ADDR/v1/transit/decrypt/roehub-exchange-credentials\" -H \"X-Vault-Token: $ROEHUB_API_TRANSIT_TOKEN\" --data '{\"ciphertext\":\"vault:v1:stage3b-placeholder\"}'"
    expect: "apps/api identity is denied decrypt capability"
  - cmd: "rg -n \"TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE|api_secret|passphrase\" logs output .playwright-cli || true"
    expect: "no secret markers are present; missing log directories are acceptable only if recorded"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  - "src/trading/contexts/exchange_control/**"
  - "apps/exchange_control/**"
  - "tests/unit/contexts/exchange_control/**"
  - "docs/runbooks/exchange-secret-management.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/03b-transit-application-integration.md"

possible_secondary_touches:
  - "src/trading/config/**"
  - "tests/unit/apps/migrations"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Do not log plaintext, ciphertext, HMAC, fingerprints, tokens, API keys, passphrases, or raw OpenBao errors containing request data."
  - "Any dev fallback must be impossible to enable in product/live-ready mode."
---

# Task

Implement Transit-backed application integration inside the `exchange-control` boundary after Stage 3A has accepted the real OpenBao/Vault runtime.

Done means:

- `ExchangeSecretCipher` or equivalent port exists;
- Transit/OpenBao adapter exists and uses Stage 3A env/runtime contract;
- product-ready mode fails closed without accepted Transit config;
- `apps/api` has no decrypt path;
- `exchange-control` uses limited Transit capabilities;
- redaction, deterministic tests, runtime ACL calls, runbook updates, Stage 3B report, and iteration ledger prove the integration.

## Context / Current State

Stage 3A provisions the secret backend. Stage 3B integrates application code with that backend.

If `docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md` is missing, blocked, not accepted, or lacks the required runtime/env/ACL evidence, stop before implementation and mark Stage 3B blocked. Do not recreate the old combined Stage 3 behavior.

This stage prepares secret custody. It does not yet migrate legacy keys into `exchange_connections`.

## Requirements (Must)

- Before making changes, verify the current branch is `main` and `git pull --ff-only origin main` succeeds; if not, stop and mark the stage blocked instead of creating a side branch.
- Confirm Stage 3A is accepted with runtime evidence for `OPENBAO_ADDR`, `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN`, `ROEHUB_API_TRANSIT_TOKEN`, key `roehub-exchange-credentials`, encrypt allowed, and API decrypt denied.
- Update the shared stage execution ledger after validation and before delivery; include stage status, evidence, blockers, compatibility/rollback notes, CI/deploy status, and facts next stages must know.
- After all required validation passes, deliver directly to `main`: stay/switch to `main`, run `git pull --ff-only origin main`, stage only scoped files, commit on `main`, push `origin main`, and follow CI/deploy status. Do not create a per-stage branch or draft PR.
- Implement a secret cipher port and Transit-compatible adapter inside the `exchange-control` boundary.
- Keep local/dev fallback explicit and blocked in product-ready mode.
- Encode service capability separation: API no decrypt, exchange-control limited decrypt.
- Add redacted `repr` / error handling for credential objects or secret DTOs.
- Add tests proving encrypt/decrypt/HMAC/fingerprint and fail-closed behavior without real secrets.
- Repeat runtime OpenBao/Vault acceptance evidence for `exchange-control` encrypt allowed and `apps/api` decrypt denied; if it is unavailable, Stage 3B must be marked blocked for production acceptance, not accepted.
- Update `docs/runbooks/exchange-secret-management.md`.
- Create Stage 3B evidence report.

## Requirements (Should)

- Include rewrap/rotation command design in the runbook even if implementation is partial.
- Normalize external Transit errors into sanitized internal errors.

## Requirements (Nice-to-have)

- Add a deterministic in-memory Transit client for tests if no suitable fake exists.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 3A report
3. Stage 2 report
4. architecture document Stage 3B
5. task entrypoints
6. conditional bundles only if config/storage/runbook details are unclear

Pre-implementation reading target:

- `<= 8 files`
- `<= ~45k tokens`

Stop reading once port, adapter, config, test, runtime ACL, runbook, report, and ledger surfaces are bounded.

# Reading manifest

Use front-matter `context_sources` as the canonical reading map. Do not preload broad infra directories unless Stage 3A evidence is missing or ambiguous.

# Work plan (agent should follow)

0. Verify the local checkout is on `main`, run `git pull --ff-only origin main`, and confirm there are no unrelated changes in scope. Stop if this cannot be proven.

Skill routing for this task:

- `contract-impact-analysis`: use before implementation and final report for secret/config contracts.
- `backend-quality-gates`: use during verification.
- `production-risk-review`: use before final report if claiming product-ready secret custody.
- `publish-ci-deploy`: use only after validation, report, and ledger update are complete.

1. Confirm Stage 3A accepted. If not accepted, stop and update the ledger/report as blocked.
2. Define the secret cipher port and product config contract.
3. Implement Transit adapter plus deterministic test fake.
4. Add fail-closed checks and redaction.
5. Repeat runtime ACL evidence from Stage 3A using the application env contract.
6. Update runbook, create Stage 3B report, and update the iteration ledger.
7. Run quality gates and secret grep.

After stage-specific verification:

- update `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` with accepted/blocked status, evidence, changed contracts, blockers, next-stage facts, and direct-main delivery status;
- perform direct-main delivery only after successful validation: confirm the current branch is `main`, fast-forward from `origin/main`, stage only scoped files, commit, push `origin main`, and watch CI/deploy status;
- if `main` cannot fast-forward, GitHub auth is unavailable, local gates fail, Stage 3A evidence is missing, runtime ACL evidence cannot be repeated, or unrelated worktree changes cannot be isolated, stop and mark the stage blocked in the ledger; do not create a stage branch or draft PR as a workaround.

# Acceptance criteria (Definition of Done)

- Iteration ledger is updated with facts required by Stage 3C.
- Stage 3A report is accepted and cited as prerequisite evidence.
- Transit encrypt acceptance call shape is documented and repeated.
- API decrypt denial is documented and test-covered or manually evidenced.
- Production startup without Transit config fails closed.
- Runtime acceptance commands from the architecture document are executed against the Stage 3A runtime; otherwise the report marks production acceptance blocked.
- Secret grep finds no test secret markers in logs/output/artifacts.
- Stage report includes exact commands, env names, product-mode fail-closed behavior, and residual risks.
- Shared ledger is updated with stage status, evidence, blockers, next-stage facts, and direct-main delivery status.
- Direct-main push to `origin/main` is completed after validation and CI/deploy status is recorded, or the stage is blocked with the exact reason.
- No per-stage branch and no draft PR are created for this stage.

# Implementation constraints

## Determinism & ordering

- Tests must not depend on live OpenBao unless explicitly marked external/manual.
- Fake clients must be deterministic.
- Stage 3B must not start until Stage 3A is accepted.

## API / contracts

- Do not expose plaintext through API, logs, reprs, exceptions, or metrics.
- Config changes are compatible only if defaults preserve non-production local behavior safely and product-ready mode fails closed.
- `apps/api` must not gain decrypt capability.

## Documentation

- Update the shared stage execution ledger before direct-main delivery; it is the canonical cross-stage handoff document.
- Record direct-main delivery evidence in the ledger: commit SHA, `git push origin main` result, CI/deploy status, runtime status when applicable, or exact blocker.
- Create Stage 3B report.
- Update `docs/runbooks/exchange-secret-management.md`.
- Update architecture only if implementation deviates from the plan.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for Transit ACL evidence: service, token/env, operation, expected result, observed result, blocker.
- Run docs-index check after Markdown changes.

## Tests

- Cover encryption, decryption capability separation, fail-closed startup, sanitized errors, and redaction.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`
- `src/trading/contexts/exchange_control/**`
- `apps/exchange_control/**`
- `tests/unit/contexts/exchange_control/**`
- `docs/runbooks/exchange-secret-management.md`
- `docs/architecture/identity/exchange-connections-stage-reports/03b-transit-application-integration.md`

Possible secondary touches:

- `src/trading/config/**`
- `tests/unit/apps/migrations`
- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/architecture/README.md`

# Non-goals

- OpenBao/Vault runtime provisioning.
- Backfill.
- Connection tables.
- External exchange validation.
- UI.
- Order execution.

# Quality gates (must run and pass)

- `test "$(git branch --show-current)" = main`
- `git pull --ff-only origin main`
- `gh --version && gh auth status`
- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/migrations`
- `uv run ruff check src/trading/contexts/exchange_control tests/unit/contexts/exchange_control`
- `uv run pyright src/trading/contexts/exchange_control tests/unit/contexts/exchange_control`
- `python -m tools.docs.generate_docs_index --check`
- `curl -fsS "$OPENBAO_ADDR/v1/sys/health"`
- `curl -fsS -X POST "$OPENBAO_ADDR/v1/transit/encrypt/roehub-exchange-credentials" -H "X-Vault-Token: $ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN" --data '{"plaintext":"U1RBR0UzQl9TTU9LRQ=="}'`
- `curl -i -X POST "$OPENBAO_ADDR/v1/transit/decrypt/roehub-exchange-credentials" -H "X-Vault-Token: $ROEHUB_API_TRANSIT_TOKEN" --data '{"ciphertext":"vault:v1:stage3b-placeholder"}'`
- `rg -n "TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE|api_secret|passphrase" logs output .playwright-cli || true`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

Your final message MUST include direct-main commit SHA, `git push origin main` status, CI/deploy status, and deploy/runtime status.

1. **Что реализовано**
2. **Secret boundary**
3. **ACL и fail-closed**
4. **Проверки**
5. **Stage 3C readiness**
6. **Direct-main delivery**
