---
prompt_name: 21-backup-restore-upgrade-and-rollback
repo: roehub.com
scope: "Implement and prove backup, restore, N-1 to N upgrade, rollback and emergency recovery across every state owner."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "21", prerequisites: ["14", "17", "18", "20"], previous_stage_gate: "Stages 14, 17, 18 and 20 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: recovery/runtime proof rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: state owners and rollback contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: artifact/topology/control/observability prerequisites}
  task_entrypoints:
    - {path: apps/roehubctl/, why: backup/restore CLI}
    - {path: apps/control_agent/, why: lifecycle operations}
    - {path: docs/runbooks/, why: current recovery guidance}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: architecture-design, timing: before recovery protocol, reason: consistency and state-owner coordination}
  - {skill: contract-impact-analysis, timing: before manifests/rollback, reason: persisted compatibility and operational semantics}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [apps/roehubctl/, apps/control_agent/, src/trading/contexts/operations/, tools/backup/, infra/docker/, schemas/backup/, tests/, docs/runbooks/, docs/architecture/platform/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated backups and never overwrite source data during drills
file_manifest:
  expected_primary_touches: [apps/roehubctl/, apps/control_agent/, tools/backup/, schemas/backup/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/21-backup-restore-upgrade-and-rollback.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [src/trading/contexts/operations/, infra/docker/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [end-to-end backup/restore smoke, PostgreSQL, ClickHouse, Redis checkpoint policy, OpenBao, artifacts, config, plugin state, upgrade N-1 to N, rollback, measured RPO/RTO]}
proof_boundary: {label: N/A, exclusions: [production restore, claiming unmeasured SLA]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Make the installation recoverable as one product while respecting different state-owner consistency and secret-handling requirements.

# Requirements

- Define a signed/versioned backup manifest covering release/config, PostgreSQL, ClickHouse, required Redis durable/checkpoint state, OpenBao, artifacts, plugin/operation metadata and audit.
- Coordinate quiesce/snapshot/application-consistent modes and explicitly record consistency limitations.
- Encrypt backup secrets/material with an operator-owned key; never include unseal/recovery values in reports.
- Implement preflight, progress, cancellation, partial failure, resumability and restore-to-new-installation.
- Prove N-1 → N upgrade and rollback rules; irreversible migrations require pre-upgrade backup and forward recovery plan.
- Measure observed RPO/RTO in drills; do not promise an SLA from estimates.

# Validation

Run an end-to-end backup/restore smoke into a separate clean installation, compare rows/time ranges/digests/users/config/plugin state/audit, then exercise N-1 → N upgrade, injected failure and rollback/recovery. Confirm `roehubctl` works with Web/API down and observability remains available.

# Stop rules

Block on incomplete state-owner coverage, unencrypted sensitive backup, restore over source, unverifiable consistency, missing rollback, data mismatch or unmeasured claimed RPO/RTO. Update ledger after evidence.
