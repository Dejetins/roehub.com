---
prompt_name: 08-openbao-secrets-and-recovery
repo: roehub.com
scope: "Containerize OpenBao and establish least-privilege secret references, rotation, backup, unseal and recovery contracts."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "08", prerequisites: ["04", "05", "06"], previous_stage_gate: "Stages 04, 05 and 06 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: secrets and runtime evidence rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: OpenBao ownership contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: stage dependencies}
  task_entrypoints:
    - {path: src/trading/contexts/exchange_control/adapters/outbound/openbao_transit.py, why: current Transit adapter}
    - {path: infra/, why: current OpenBao provisioning/runtime}
    - {path: docs/runbooks/exchange-secret-management.md, why: current secret operations}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before secret-source changes, reason: secret refs, auth and recovery semantics}
  - {skill: backend-quality-gates, timing: verification, reason: adapters and policy checks}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [infra/openbao/, infra/docker/, configs/, src/trading/contexts/exchange_control/, src/trading/platform/, tests/, docs/runbooks/, docs/architecture/platform/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated production provisioning and never read or copy secret values
file_manifest:
  expected_primary_touches: [infra/openbao/, infra/docker/, configs/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/08-openbao-secrets-and-recovery.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [src/trading/contexts/exchange_control/, src/trading/platform/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [container initialization, sealed/unsealed states, least privilege, secret reference resolution, rotation, backup and recovery, redaction]}
proof_boundary: {label: N/A, exclusions: [real exchange or Telegram secret migration, production OpenBao mutation]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Provide a self-hosted OpenBao profile with recoverable initialization and service-specific policies while moving application contracts from raw environment secrets to opaque secret references.

# Requirements

- Persist OpenBao state outside the container and document supported initialization/unseal/recovery ownership.
- Create least-privilege service identities; API must not receive exchange decrypt capability or root/unseal material.
- Model typed secret references for exchange, Telegram, OIDC and plugins without returning secret values through API/UI.
- Support rotation and version selection with audit and rollback semantics.
- Backup metadata/state using an approved encrypted destination; never commit or print recovery material.
- Fail closed when sealed/unavailable and connect the state to `ops.roehub.io/v1`.

# Validation and stop rules

Use a disposable OpenBao container to prove init, sealed/unsealed readiness, policy denials, allowed operations, rotation, restart, backup/recovery and forbidden-output scans. Block on secrets in logs/evidence/DB, shared broad tokens, manual undocumented state, container-local persistence or recovery that requires chat disclosure. Update ledger after validation.
