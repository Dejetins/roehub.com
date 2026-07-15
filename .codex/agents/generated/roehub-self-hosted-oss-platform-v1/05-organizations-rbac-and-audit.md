---
prompt_name: 05-organizations-rbac-and-audit
repo: roehub.com
scope: "Implement greenfield installation/site, multi-organization membership, RBAC, ownership integrity and immutable administrative audit."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "05", prerequisites: ["04"], previous_stage_gate: "Stage 04 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: identity and contract rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: accepted role and tenancy model}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: greenfield schema handoff}
  task_entrypoints:
    - {path: src/trading/contexts/identity/, why: identity domain and persistence}
    - {path: apps/api/, why: current principal and authorization boundary}
    - {path: migrations/postgres/, why: organization-aware schema}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: architecture-design, timing: before boundary changes, reason: organization and authorization ownership}
  - {skill: contract-impact-analysis, timing: before schema/API edits, reason: principal, DTO and persisted ownership break}
  - {skill: backend-quality-gates, timing: verification, reason: focused Python gates}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/identity/, src/trading/shared_kernel/, apps/api/, migrations/postgres/, tests/, docs/architecture/identity/, docs/architecture/platform/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated identity/live-trading work and stage only owned hunks if later authorized
file_manifest:
  expected_primary_touches: [src/trading/contexts/identity/, migrations/postgres/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/05-organizations-rbac-and-audit.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [src/trading/shared_kernel/, apps/api/, docs/architecture/identity/, docs/architecture/README.md]
validation_strategy: {depth: api, acceptance_surfaces: [fresh schema, two-organization isolation, role matrix, last-owner invariant, admin role/plugin permissions, audit, ownership referential integrity]}
proof_boundary: {label: N/A, exclusions: [browser auth, plugin implementation, production users]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Make Roehub DB authoritative for installations, organizations, memberships, roles and permissions, with stable internal `user_id` values created in the new installation.

# Requirements

- Model `installation_owner`, organization `owner/admin/operator/trader/viewer`, memberships, invitations, scoped permissions and audit events.
- `admin` manages members, roles and plugins within granted installation/org scope; it cannot remove/demote the last owner or bypass owner-only mainnet/recovery invariants.
- Organization scope is derived server-side, never trusted from request payload alone.
- Create organization-aware ownership columns and constraints from the first v1 schema; do not add a current-data backfill or dual-read window.
- Enforce same-organization semantic references so provenance, exchange connections, strategies and positions cannot form cross-owner links; reject orphan references at the database or application invariant boundary.
- Installation support access is absent by default and time-bounded/audited when explicitly elevated.
- Remove subscription decisions from authorization incrementally; do not delete `paid_level` until consumer search proves safe.

# Acceptance

Fresh-schema tests and a disposable PostgreSQL proof pass. API/use-case tests prove two-organization isolation, every role, last-owner protection, admin role/plugin management, recent-auth markers for privileged operations, same-org/reference invariants, and complete audit without sensitive payloads.

# Stop rules

Any cross-organization read/write, unclear fresh-resource ownership, orphan/cross-owner semantic reference, privilege escalation, missing audit, or dependency on current production data is `blocked`. Update ledger after evidence.
