---
prompt_name: identity_exchange_connections_v1_03_secret_engine_transit_after_service_identity
repo: roehub.com
branch: main
scope: "Stage 3: add OpenBao/Vault Transit secret boundary after `exchange-control` service identity exists."

language:
  implementation: python_security_config_tests_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and secret safety"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 3 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared stage execution ledger and direct-main delivery handoff facts"
    - path: docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md
      why: "accepted Stage 2 evidence"
  task_entrypoints:
    - path: src/trading/contexts/exchange_control
      why: "new secret boundary module"
      inspect_symbols:
        - ExchangeSecretCipher
        - service identity
    - path: src/trading/contexts/identity
      why: "existing exchange key encryption patterns"
      inspect_symbols:
        - encrypted exchange key storage
        - fingerprint handling
    - path: tests/unit/contexts/exchange_control
      why: "new focused secret tests"
      inspect_symbols:
        - secret cipher tests
        - config fail-closed tests
    - path: tests/unit/apps/migrations
      why: "migration compatibility tests if schema changes"
      inspect_symbols:
        - bootstrap apply flow
  conditional_bundles:
    existing_secret_policy:
      read_when: "current encryption or redaction policy is ambiguous"
      paths:
        - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
        - migrations/postgres/0004_identity_exchange_keys_v2.sql
    ops_secret_docs:
      read_when: "creating OpenBao/Vault runbook or config docs"
      paths:
        - docs/runbooks/mac-studio-monitoring-plan.md
        - docs/runbooks
    config_wiring:
      read_when: "adding production fail-closed config"
      paths:
        - apps/api
        - apps/exchange_control
        - src/trading/config
  consult_if_needed:
    - path: infra
      read_when: "repository has existing secret/config deployment patterns"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/03-secret-engine-transit.md"
  additional_new_docs:
    - "docs/runbooks/exchange-secret-management.md"
  canonical_shape: "stage report with Markdown evidence tables: actor, Transit capability, command/test, expected result, observed result"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "03"
  update_required: true
  update_timing: "after validation, before direct-main push and final report"
  direct_main_delivery_required: true

hard_requirements:
  iteration_ledger_update_required: true
  previous_stage_must_be_accepted: true
  transit_acl_after_service_identity: true
  api_process_no_decrypt: true
  exchange_control_limited_decrypt: true
  product_ready_fail_closed_without_transit: true
  no_secret_leak_required: true
  runtime_transit_acl_evidence_required: true
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
  - skill: publish-ci-deploy
    use_when: "stage implementation, validation, stage report, and ledger update are complete"
    timing: "after validation and before final report"
    reason: "user requires direct push to main after accepted validation, with CI/deploy follow-through"
  - skill: contract-impact-analysis
    use_when: "changing config, secret cipher, storage metadata, or service capabilities"
    timing: "before implementation and final report"
    reason: "secret boundary is a production contract"
  - skill: backend-quality-gates
    use_when: "running secret/config tests, ruff, pyright, docs gates"
    timing: "during verification"
    reason: "security implementation needs deterministic gates"
  - skill: production-risk-review
    use_when: "before declaring Transit ACL/product-ready behavior complete"
    timing: "before final report"
    reason: "credential custody is production-risk sensitive"


target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "openbao_transit_v1"
  - "vault_transit_v1"
  - "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN"
  - "ROEHUB_API_TRANSIT_TOKEN"
  - "roehub-exchange-credentials"
  - "ExchangeSecretCipher"

non_goals:
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
    - "Stage 4 readiness"
    - "Direct-main delivery"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/migrations"
    expect: "passes; create focused tests if needed"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown changes"
  - cmd: "curl -fsS \"$OPENBAO_ADDR/v1/sys/health\""
    expect: "OpenBao/Vault is healthy on the target runtime; otherwise Stage 3 is blocked for production acceptance"
  - cmd: "curl -fsS -X POST \"$OPENBAO_ADDR/v1/transit/encrypt/roehub-exchange-credentials\" -H \"X-Vault-Token: $ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN\" --data '{\"plaintext\":\"VEVTVF9TRUNSRVQ=\"}'"
    expect: "exchange-control identity can encrypt"
  - cmd: "curl -i -X POST \"$OPENBAO_ADDR/v1/transit/decrypt/roehub-exchange-credentials\" -H \"X-Vault-Token: $ROEHUB_API_TRANSIT_TOKEN\" --data '{\"ciphertext\":\"vault:v1:example\"}'"
    expect: "apps/api identity is denied decrypt capability"
  - cmd: 'test "$(git branch --show-current)" = main'
    expect: "passes before direct-main push; otherwise stop and do not create a stage branch"
  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"

  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  - "src/trading/contexts/exchange_control/**"
  - "tests/unit/contexts/exchange_control/**"
  - "docs/runbooks/exchange-secret-management.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/03-secret-engine-transit.md"

possible_secondary_touches:
  - "apps/exchange_control/**"
  - "src/trading/config/**"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Do not log plaintext, ciphertext, HMAC, tokens, API keys, or raw OpenBao errors containing request data."
  - "Any dev fallback must be impossible to enable in product/live-ready mode."
---

# Task

Implement the Transit-backed secret engine foundation after `exchange-control` service identity exists.

Done means:

- `ExchangeSecretCipher` or equivalent port exists;
- Transit/OpenBao adapter exists with product-ready fail-closed config;
- `apps/api` has no decrypt capability;
- `exchange-control` has only the limited capabilities required by the plan;
- Stage 3 evidence report proves ACL and secret-grep behavior.

## Context / Current State

The architecture requires Transit ACL only after Stage 2 service identity is created. If Stage 2 evidence is missing or blocked, stop.

This stage prepares secret custody. It does not yet migrate legacy keys into `exchange_connections`.

## Requirements (Must)

- Before making changes, verify the current branch is `main` and `git pull --ff-only origin main` succeeds; if not, stop and mark the stage blocked instead of creating a side branch.
- Update the shared stage execution ledger after validation and before delivery; include stage status, evidence, blockers, compatibility/rollback notes, CI/deploy status, and facts next stages must know.
- After all required validation passes, deliver directly to `main`: stay/switch to `main`, run `git pull --ff-only origin main`, stage only scoped files, commit on `main`, push `origin main`, and follow CI/deploy status. Do not create a per-stage branch or draft PR.
- Implement a secret cipher port and Transit-compatible adapter.
- Keep local/dev fallback explicit and blocked in product-ready mode.
- Encode service capability separation: API no decrypt, exchange-control limited decrypt.
- Add redacted `repr` / error handling for credential objects or secret DTOs.
- Add tests proving encrypt/decrypt/HMAC/fingerprint and fail-closed behavior without real secrets.
- Include runtime OpenBao/Vault acceptance evidence for `exchange-control` encrypt allowed and `apps/api` decrypt denied when OpenBao/Vault is available; if it is unavailable, Stage 3 must be marked blocked for production acceptance, not accepted.
- Create `docs/runbooks/exchange-secret-management.md`.
- Create Stage 3 evidence report.

## Requirements (Should)

- Include rewrap/rotation command design in the runbook even if implementation is partial.
- Normalize external Transit errors into sanitized internal errors.

## Requirements (Nice-to-have)

- Add a fake in-memory Transit client for deterministic tests.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 2 report
3. architecture document Stage 3
4. task entrypoints
5. conditional bundles only if config/storage/runbook details are unclear

Pre-implementation reading target:

- `<= 8 files`
- `<= ~45k tokens`

Stop reading once port, adapter, config, test, and runbook surfaces are bounded.

# Reading manifest

Use front-matter `context_sources` as the canonical reading map. Do not preload broad infra directories unless a config blocker appears.

# Work plan (agent should follow)

0. Verify the local checkout is on `main`, run `git pull --ff-only origin main`, and confirm there are no unrelated changes in scope. Stop if this cannot be proven.
Skill routing for this task:

- `contract-impact-analysis`: use before implementation and final report for secret/config contracts.
- `backend-quality-gates`: use during verification.
- `production-risk-review`: use before final report if claiming product-ready secret custody.

1. Confirm Stage 2 accepted.
2. Define the secret cipher port and product config contract.
3. Implement Transit adapter plus test fake.
4. Add fail-closed checks and redaction.
5. Write runbook and Stage 3 report.

After stage-specific verification:

- update `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` with accepted/blocked status, evidence, changed contracts, blockers, next-stage facts, and direct-main delivery status;
- perform direct-main delivery only after successful validation: confirm the current branch is `main`, fast-forward from `origin/main`, stage only scoped files, commit, push `origin main`, and watch CI/deploy status;
- if `main` cannot fast-forward, GitHub auth is unavailable, local gates fail, or unrelated worktree changes cannot be isolated, stop and mark the stage blocked in the ledger; do not create a stage branch or draft PR as a workaround.

# Acceptance criteria (Definition of Done)

- Iteration ledger is updated with facts required by the next stage.
- Transit encrypt acceptance call shape is documented.
- API decrypt denial is documented and test-covered or manually evidenced.
- Production startup without Transit config fails closed.
- Runtime acceptance commands from the architecture document are executed, or the report marks production acceptance blocked because OpenBao/Vault is unavailable.
- Secret grep finds no test secret markers in logs/output/artifacts.
- Stage report includes exact commands, env names, and residual risks.
- Shared ledger `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` is updated with stage status, evidence, blockers, next-stage facts, and direct-main delivery status.
- Direct-main push to `origin/main` is completed after validation and CI/deploy status is recorded, or the stage is blocked with the exact reason.
- No per-stage branch and no draft PR are created for this stage.

# Implementation constraints

## Determinism & ordering

- Tests must not depend on live OpenBao unless explicitly marked external/manual.
- Fake clients must be deterministic.

## API / contracts

- Do not expose plaintext through API, logs, reprs, exceptions, or metrics.
- Config changes are compatible only if defaults preserve non-production local behavior safely.

## Documentation

- Update the shared stage execution ledger before direct-main delivery; it is the canonical cross-stage handoff document.
- Record direct-main delivery evidence in the ledger: commit SHA, `git push origin main` result, CI/deploy status, runtime status when applicable, or exact blocker.
- Create the secret-management runbook and Stage 3 report.
- Update architecture only if implementation deviates from the plan.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for Transit ACL evidence: service, token/env, operation, expected result, observed result, blocker.
- Run docs-index check after Markdown changes.

## Tests

- Cover encryption, decryption capability separation, fail-closed startup, and redaction.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`
- `src/trading/contexts/exchange_control/**`
- `tests/unit/contexts/exchange_control/**`
- `docs/runbooks/exchange-secret-management.md`
- `docs/architecture/identity/exchange-connections-stage-reports/03-secret-engine-transit.md`

Possible secondary touches:

- `apps/exchange_control/**`
- `src/trading/config/**`
- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/architecture/README.md`

# Non-goals

- Backfill.
- Connection tables.
- External exchange validation.
- UI.
- Order execution.

# Quality gates (must run and pass)

- `test "$(git branch --show-current)" = main`
- `gh --version && gh auth status`
- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/migrations`
- `uv run ruff check src/trading/contexts/exchange_control tests/unit/contexts/exchange_control`
- `uv run pyright src/trading/contexts/exchange_control tests/unit/contexts/exchange_control`
- `python -m tools.docs.generate_docs_index --check`
- `curl -fsS "$OPENBAO_ADDR/v1/sys/health"`
- `curl -fsS -X POST "$OPENBAO_ADDR/v1/transit/encrypt/roehub-exchange-credentials" -H "X-Vault-Token: $ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN" --data '{"plaintext":"VEVTVF9TRUNSRVQ="}'`
- `curl -i -X POST "$OPENBAO_ADDR/v1/transit/decrypt/roehub-exchange-credentials" -H "X-Vault-Token: $ROEHUB_API_TRANSIT_TOKEN" --data '{"ciphertext":"vault:v1:example"}'`
- `rg -n "TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE" logs output .playwright-cli || true`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

Your final message MUST include direct-main commit SHA, `git push origin main` status, CI/deploy status, and deploy/runtime status.

1. **Что реализовано**
2. **Secret boundary**
3. **ACL и fail-closed**
4. **Проверки**
5. **Stage 4 readiness**
6. **Direct-main delivery**
