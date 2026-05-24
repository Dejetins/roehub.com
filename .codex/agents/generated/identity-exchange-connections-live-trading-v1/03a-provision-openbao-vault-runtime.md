---
prompt_name: identity_exchange_connections_v1_03a_openbao_vault_runtime_provisioning
repo: roehub.com
branch: main
scope: "Stage 3A: provision supervised OpenBao/Vault Transit runtime, Transit key, ACL policies/tokens, monitoring, recovery, and runtime evidence before application integration."

language:
  implementation: ops_security_config_docs_runtime
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, secret safety, direct-main delivery rules"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 3A source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared stage execution ledger and Stage 2 handoff facts"
    - path: docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md
      why: "accepted Stage 2 runtime/service identity evidence"
  task_entrypoints:
    - path: infra/macos/launchd
      why: "Mac Studio supervised runtime patterns"
      inspect_symbols:
        - roehub service plist patterns
        - local-only service boundaries
    - path: infra/scripts/monit
      why: "Monit process supervision and restart patterns"
      inspect_symbols:
        - roehub process checks
        - restart expectations
    - path: infra/macos/prometheus/prometheus.prod.yml
      why: "production scrape config patterns"
      inspect_symbols:
        - scrape_configs
        - local target labels
    - path: docs/runbooks/mac-studio-monitoring-plan.md
      why: "existing runtime monitoring/runbook shape"
      inspect_symbols:
        - Monit
        - Prometheus
        - restart evidence
  conditional_bundles:
    existing_secret_docs:
      read_when: "secret engine naming, custody, or rotation language is unclear"
      paths:
        - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
        - docs/runbooks
    service_identity_contract:
      read_when: "Transit ACL principal mapping is unclear"
      paths:
        - src/trading/contexts/exchange_control/application/service_identity.py
        - apps/exchange_control
    deployment_scripts:
      read_when: "OpenBao/Vault runtime must be integrated into existing reload/bootstrap flow"
      paths:
        - infra/scripts
        - scripts
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md
      read_when: "recent-auth/audit security baseline affects secret provisioning documentation"
    - path: docs/architecture/README.md
      read_when: "docs index or architecture navigation must be updated"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/runbooks/mac-studio-monitoring-plan.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md"
  additional_new_docs:
    - "docs/runbooks/exchange-secret-management.md"
  canonical_shape: "stage report with Markdown evidence tables: runtime component, command/call, expected result, observed result, blocker"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "03A"
  update_required: true
  update_timing: "after validation, before direct-main push and final report"
  direct_main_delivery_required: true

hard_requirements:
  iteration_ledger_update_required: true
  previous_stage_must_be_accepted: true
  openbao_vault_runtime_required: true
  transit_secret_engine_required: true
  transit_key_required: true
  transit_key_name_must_be_roehub_exchange_credentials: true
  exchange_control_token_encrypt_required: true
  api_token_decrypt_denied_required: true
  service_identity_acl_required: true
  supervised_runtime_required: true
  prometheus_metrics_required: true
  monit_control_required: true
  recovery_backup_runbook_required: true
  no_secret_material_in_repo_required: true
  stage_execution_ledger_update_required: true
  direct_main_push_after_validation_required: true
  feature_branch_per_stage_forbidden: true
  draft_pr_forbidden: true
  work_on_main_from_start_required: true

task_toggles:
  provision_runtime: true
  configure_transit_key: true
  configure_acl_policies: true
  configure_monit_and_prometheus: true
  update_secret_runbook: true
  publish_after_success: true
  direct_main_push_after_validation: true
  target_branch: main
  draft_pr_after_success: false

skill_routing:
  - skill: publish-ci-deploy
    use_when: "runtime provisioning, validation, Stage 3A report, and ledger update are complete"
    timing: "after validation and before final report"
    reason: "user requires direct push to main after accepted validation, with CI/deploy follow-through"
  - skill: contract-impact-analysis
    use_when: "adding runtime config, service identities, env vars, ACL policies, monitoring, or runbook contracts"
    timing: "before implementation and final report"
    reason: "secret backend provisioning changes operational contracts"
  - skill: production-risk-review
    use_when: "before declaring Stage 3A accepted"
    timing: "before final report"
    reason: "credential custody infrastructure is production-risk sensitive"
  - skill: backend-quality-gates
    use_when: "running docs, config, script, or focused repository checks"
    timing: "during verification"
    reason: "provisioning docs/config changes still need deterministic gates"

target_envs:
  - mac-studio
  - local-dev

required_literals:
  - "OPENBAO_ADDR"
  - "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN"
  - "ROEHUB_API_TRANSIT_TOKEN"
  - "roehub-exchange-credentials"
  - "transit"
  - "exchange-control"
  - "apps/api"

non_goals:
  - "Do not implement `ExchangeSecretCipher` application code in Stage 3A."
  - "Do not migrate or backfill `identity_exchange_keys`."
  - "Do not add Binance/Bybit validation."
  - "Do not expose plaintext credentials to apps/api."
  - "Do not implement order execution."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "OpenBao/Vault runtime"
    - "Transit ACL"
    - "Monitoring и recovery"
    - "Проверки"
    - "Stage 3B readiness"
    - "Direct-main delivery"

quality_gates:
  - cmd: 'test "$(git branch --show-current)" = main'
    expect: "passes before direct-main push; otherwise stop and do not create a stage branch"
  - cmd: "git pull --ff-only origin main"
    expect: "passes before making delivery changes"
  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"
  - cmd: "curl -fsS \"$OPENBAO_ADDR/v1/sys/health\""
    expect: "OpenBao/Vault is healthy on the target runtime"
  - cmd: "curl -fsS -X POST \"$OPENBAO_ADDR/v1/transit/encrypt/roehub-exchange-credentials\" -H \"X-Vault-Token: $ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN\" --data '{\"plaintext\":\"U1RBR0UzQV9TTU9LRQ==\"}'"
    expect: "exchange-control identity can encrypt through Transit; this also proves the key exists without requiring metadata-read permission"
  - cmd: "curl -i -X POST \"$OPENBAO_ADDR/v1/transit/decrypt/roehub-exchange-credentials\" -H \"X-Vault-Token: $ROEHUB_API_TRANSIT_TOKEN\" --data '{\"ciphertext\":\"vault:v1:stage3a-placeholder\"}'"
    expect: "apps/api identity is denied decrypt capability; HTTP status must prove denial"
  - cmd: "monit summary | rg -i \"openbao|vault|transit|roehub\""
    expect: "runtime supervision is visible; exact service name must be recorded in the report"
  - cmd: "curl -fsS \"$ROEHUB_PROMETHEUS_URL/api/v1/query?query=up\" | rg -i \"openbao|vault|transit|exchange-control\""
    expect: "Prometheus evidence exists for the provisioned runtime, or the exact target-runtime query is recorded"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown changes"
  - cmd: "rg -n \"OPENBAO_TOKEN|VAULT_TOKEN|ROEHUB_.*TRANSIT_TOKEN|api_secret|apiKey|passphrase\" docs infra .codex/agents/generated || true"
    expect: "no raw secret values are committed; only variable names and placeholders are acceptable"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md"
  - "docs/runbooks/exchange-secret-management.md"
  - "infra/macos/launchd/**"
  - "infra/scripts/monit/**"
  - "infra/macos/prometheus/prometheus.prod.yml"

possible_secondary_touches:
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"
  - "scripts/**"

safety_notes:
  - "Do not commit root tokens, recovery keys, unseal keys, service tokens, wrapped tokens, API keys, secret values, ciphertext examples from real credentials, or provider responses containing account data."
  - "Runtime evidence may include HTTP status, policy names, service names, and sanitized command shapes only."
---

# Task

Provision the OpenBao/Vault Transit-compatible runtime required for exchange credential custody.

Done means:

- OpenBao or Vault-compatible Transit runtime is supervised on the target runtime;
- Transit secret engine is enabled and key `roehub-exchange-credentials` exists;
- service-specific ACL is proven with direct runtime calls;
- `exchange-control` can encrypt;
- `apps/api` cannot decrypt;
- Monit, Prometheus, recovery, and backup/restore expectations are documented;
- Stage 3A report and the shared iteration ledger make Stage 3B safe to start; Stage 3C/4/5 must still wait for their own prerequisites.

## Context / Current State

Stage 2 created the `exchange-control` runtime boundary and service identity. Stage 3A exists because the application cannot safely integrate Transit until the actual secret backend, key, ACL policies, runtime tokens, monitoring, and recovery path have been provisioned and proven.

If Stage 2 report is missing, not accepted, or contradicted by runtime evidence, stop and mark Stage 3A blocked.

This stage is provisioning and operational evidence only. It must not add `ExchangeSecretCipher` application code, migrate credentials, or add exchange validation.

## Requirements (Must)

- Before making changes, verify the current branch is `main` and `git pull --ff-only origin main` succeeds; if not, stop and mark the stage blocked instead of creating a side branch.
- Update the shared stage execution ledger after validation and before delivery; include stage status, evidence, blockers, compatibility/rollback notes, CI/deploy status, and facts Stage 3B/3C/4/5 must know.
- After all required validation passes, deliver directly to `main`: stay/switch to `main`, run `git pull --ff-only origin main`, stage only scoped files, commit on `main`, push `origin main`, and follow CI/deploy status. Do not create a per-stage branch or draft PR.
- Provision or document the exact already-provisioned OpenBao/Vault runtime endpoint in `OPENBAO_ADDR`.
- Enable Transit and create key `roehub-exchange-credentials`.
- Prove key existence through sanitized provisioning output or a successful scoped encrypt call; do not require metadata-read permission for `exchange-control`.
- Create separate policy/token identities for `exchange-control` and `apps/api`.
- Prove `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN` can encrypt with `roehub-exchange-credentials`.
- Prove `ROEHUB_API_TRANSIT_TOKEN` cannot decrypt with `roehub-exchange-credentials`.
- Define where env vars are injected at runtime without committing their values.
- Add/confirm Monit supervision and Prometheus evidence for the secret backend.
- Document recovery, backup/restore, token rotation, key rotation, and emergency disable procedures.
- Create Stage 3A report with concrete commands/calls and observed results.
- If OpenBao/Vault cannot be provisioned or required env/token evidence is missing, mark Stage 3A blocked; do not accept it with local-only or theoretical evidence.

## Requirements (Should)

- Keep OpenBao/Vault bound to the smallest practical network surface for the current deployment.
- Prefer policy names and service labels that match `exchange-control` and `apps/api`.
- Include a rollback path that revokes tokens and disables the service without database rollback.

## Requirements (Nice-to-have)

- Add a sanitized smoke script for Transit ACL checks if the repo already has a suitable scripts pattern.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 2 report
3. architecture document Stage 3A
4. task entrypoints
5. conditional bundles only if runtime, ACL, monitoring, or recovery details are unclear

Pre-implementation reading target:

- `<= 8 files`
- `<= ~45k tokens`

Stop reading once runtime service, key, ACL policy, token injection, monitoring, recovery, report, and ledger surfaces are bounded.

# Reading manifest

Use front-matter `context_sources` as the canonical reading map. Do not preload application code for `ExchangeSecretCipher`; that is Stage 3B.

# Work plan (agent should follow)

0. Verify the local checkout is on `main`, run `git pull --ff-only origin main`, and confirm there are no unrelated changes in scope. Stop if this cannot be proven.

Skill routing for this task:

- `contract-impact-analysis`: use for env vars, runtime ACL, monitoring, rollback, and operational contracts.
- `production-risk-review`: use before claiming Stage 3A accepted.
- `backend-quality-gates`: use for docs/config/script checks.

1. Confirm Stage 2 accepted and the `exchange-control` runtime identity is stable.
2. Provision or verify OpenBao/Vault runtime and Transit mount.
3. Create/verify key `roehub-exchange-credentials`.
4. Create/verify `exchange-control` and `apps/api` policies/tokens without committing token values.
5. Prove health, encrypt allowed, decrypt denied, Monit, Prometheus, and recovery evidence.
6. Update `docs/runbooks/exchange-secret-management.md`, create Stage 3A report, and update the iteration ledger.
7. Run docs/secret-grep gates.

After stage-specific verification:

- update `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` with accepted/blocked status, evidence, changed contracts, blockers, next-stage facts, and direct-main delivery status;
- perform direct-main delivery only after successful validation: confirm the current branch is `main`, fast-forward from `origin/main`, stage only scoped files, commit, push `origin main`, and watch CI/deploy status;
- if `main` cannot fast-forward, GitHub auth is unavailable, local gates fail, runtime evidence is missing, or unrelated worktree changes cannot be isolated, stop and mark the stage blocked in the ledger; do not create a stage branch or draft PR as a workaround.

# Acceptance criteria (Definition of Done)

- Iteration ledger is updated with Stage 3A status and facts required by Stage 3B.
- Stage 3A report exists at `docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md`.
- OpenBao/Vault health is proven on the target runtime.
- Transit key `roehub-exchange-credentials` is proven to exist by sanitized provisioning evidence or successful scoped encrypt call.
- `exchange-control` encrypt call succeeds through Transit.
- `apps/api` decrypt call is denied through Transit.
- Monit and Prometheus evidence are recorded.
- Recovery, backup/restore, token rotation, key rotation, and emergency disable procedures are documented.
- No raw tokens, keys, passphrases, real ciphertext from user credentials, or sensitive provider data are committed.
- Shared ledger is updated with status, evidence, blockers, next-stage facts, and direct-main delivery status.
- Direct-main push to `origin/main` is completed after validation and CI/deploy status is recorded, or the stage is blocked with the exact reason.
- No per-stage branch and no draft PR are created for this stage.

# Implementation constraints

## Determinism & ordering

- Stage 3A must finish before Stage 3B can start.
- Do not accept Stage 3A based only on local docs or fake clients.
- Do not store token values in repo files, reports, ledger, logs, shell history excerpts, or screenshots.

## API / contracts

- Public API and database contracts should remain unchanged in Stage 3A.
- Operational contract changes are allowed only if documented as compatible changes with rollback.

## Documentation

- Update the shared stage execution ledger before direct-main delivery; it is the canonical cross-stage handoff document.
- Record direct-main delivery evidence in the ledger: commit SHA, `git push origin main` result, CI/deploy status, runtime status, or exact blocker.
- Create `docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md`.
- Update or create `docs/runbooks/exchange-secret-management.md`.
- Update architecture only if implementation deviates from the plan.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for runtime, ACL, monitoring, recovery, rollback, and blocker evidence.
- Run docs-index check after Markdown changes.

## Tests / runtime checks

- This stage is accepted by concrete runtime calls, not only unit tests.
- If runtime calls cannot be executed because env vars or admin access are unavailable, record `blocked_runtime_provisioning_evidence_missing`.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`
- `docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md`
- `docs/runbooks/exchange-secret-management.md`
- `infra/macos/launchd/**`
- `infra/scripts/monit/**`
- `infra/macos/prometheus/prometheus.prod.yml`

Possible secondary touches:

- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/architecture/README.md`
- `scripts/**`

# Non-goals

- `ExchangeSecretCipher` application integration.
- Backfill.
- Connection tables.
- External exchange validation.
- UI.
- Order execution.

# Quality gates (must run and pass)

- `test "$(git branch --show-current)" = main`
- `git pull --ff-only origin main`
- `gh --version && gh auth status`
- `curl -fsS "$OPENBAO_ADDR/v1/sys/health"`
- `curl -fsS -X POST "$OPENBAO_ADDR/v1/transit/encrypt/roehub-exchange-credentials" -H "X-Vault-Token: $ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN" --data '{"plaintext":"U1RBR0UzQV9TTU9LRQ=="}'`
- `curl -i -X POST "$OPENBAO_ADDR/v1/transit/decrypt/roehub-exchange-credentials" -H "X-Vault-Token: $ROEHUB_API_TRANSIT_TOKEN" --data '{"ciphertext":"vault:v1:stage3a-placeholder"}'`
- `monit summary | rg -i "openbao|vault|transit|roehub"`
- `python -m tools.docs.generate_docs_index --check`
- `rg -n "OPENBAO_TOKEN|VAULT_TOKEN|ROEHUB_.*TRANSIT_TOKEN|api_secret|apiKey|passphrase" docs infra .codex/agents/generated || true`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

Your final message MUST include direct-main commit SHA, `git push origin main` status, CI/deploy status, and deploy/runtime status.

1. **Что реализовано**
2. **OpenBao/Vault runtime**
3. **Transit ACL**
4. **Monitoring и recovery**
5. **Проверки**
6. **Stage 3B readiness**
7. **Direct-main delivery**
