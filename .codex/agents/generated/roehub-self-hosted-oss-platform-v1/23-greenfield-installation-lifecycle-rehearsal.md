---
prompt_name: 23-greenfield-installation-lifecycle-rehearsal
repo: roehub.com
scope: "Rehearse a complete clean installation and self-hosted lifecycle from the signed offline bundle without reading or copying current production state."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "23", prerequisites: ["21", "22"], previous_stage_gate: "Stages 21 and 22 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: runtime, secrets and proof boundaries}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: greenfield lifecycle and rollback plan}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: lifecycle prerequisites}
  task_entrypoints:
    - {path: tools/release/, why: signed offline bundle and release tooling}
    - {path: apps/roehubctl/, why: bootstrap, backup, restore and rollback interface}
    - {path: docs/runbooks/, why: generated lifecycle operations}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger and goal stop rules}
  - {skill: backend-quality-gates, timing: local tooling verification, reason: lifecycle tool gates}
  - {skill: browser-qa-evidence, timing: bootstrap and admin smoke, reason: real greenfield user flow}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [tools/release/, apps/roehubctl/, tests/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated repository and runtime work; never connect to or copy current production databases, identity stores, secrets or artifacts
file_manifest:
  expected_primary_touches: [tools/release/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/23-greenfield-installation-lifecycle-rehearsal.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [apps/roehubctl/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [offline clean install, owner bootstrap, representative fresh-state creation, backup and restore, release-to-release upgrade and rollback, repeatable teardown and reinstall, browser/API/database reconciliation]}
proof_boundary: {label: N/A, exclusions: [current production state access, source migration, production deployment, changed-code production proof]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Prove that a user can install Roehub from the signed offline bundle into empty stores, create representative state through supported contracts, and complete backup/restore/upgrade/rollback without any dependency on the current Mac Studio/VPS data.

# Requirements

- Start from empty volumes and no reused configuration, identity, secret or artifact state.
- Bootstrap a disposable `installation_owner` through `roehubctl`, then create two organizations and representative users/resources through the public API/use-case boundaries.
- Configure only disposable OIDC/provider/testnet fixtures and secret references created for this isolated installation. Never read production credentials.
- Create a bounded representative dataset covering PostgreSQL, ClickHouse, Redis durable/checkpoint semantics, OpenBao metadata and artifact digests without personal or production data.
- Back up the new installation, restore it into a second empty destination and reconcile counts, time ranges, ownership, digests, identity membership and audit.
- Rehearse a release-to-release schema/config upgrade and rollback. If no published `N-1` exists for first v1, use the accepted versioned previous-schema fixture from Stage `21` and label that boundary exactly.
- Tear down and repeat the clean install to prove deterministic bootstrap and document duration/capacity.

# Validation

Run a real isolated runtime rehearsal from the offline bundle, focused tooling gates, real API/database checks and a bounded browser bootstrap/admin smoke. Produce a sanitized lifecycle report. The current production database, Keycloak, OpenBao, Redis checkpoints and artifact paths must not be queried or mounted. Fixtures cannot substitute for the required real clean-install runtime, but they are the correct source of representative data.

# Stop rules

Any dependency on current production state, secret exposure, non-empty starting store, reconciliation mismatch without disposition, non-repeatable bootstrap, failed restore/rollback, cross-org leakage or need for production mutation is `blocked`. Update the ledger after evidence.
