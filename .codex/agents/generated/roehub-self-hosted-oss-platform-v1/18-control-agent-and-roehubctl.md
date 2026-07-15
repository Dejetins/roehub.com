---
prompt_name: 18-control-agent-and-roehubctl
repo: roehub.com
scope: "Implement the typed control-agent trust boundary and host-side roehubctl emergency lifecycle without giving Docker access to Web/API."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "18", prerequisites: ["12", "17"], previous_stage_gate: "Stages 12 and 17 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: runtime, safety and Git rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: control-agent and emergency CLI contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: plugin/topology prerequisites}
  task_entrypoints:
    - {path: apps/cli/, why: existing CLI conventions}
    - {path: apps/api/, why: administrative operation caller}
    - {path: infra/docker/, why: controlled runtime targets}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: architecture-design, timing: before control protocol, reason: host privilege and failure boundary}
  - {skill: contract-impact-analysis, timing: before operations, reason: auth/idempotency/retry/audit semantics}
  - {skill: backend-quality-gates, timing: verification, reason: focused code gates}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/operations/, apps/control_agent/, apps/roehubctl/, apps/cli/, apps/api/, infra/docker/, schemas/operations/, tests/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated CLI/runtime work and never grant API direct Docker access
file_manifest:
  expected_primary_touches: [src/trading/contexts/operations/, apps/control_agent/, apps/roehubctl/, schemas/operations/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/18-control-agent-and-roehubctl.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [apps/cli/, apps/api/, infra/docker/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [typed operation protocol, real-boundary degraded runtime smoke, operation idempotency/journal, image allowlist, forbidden shell/mount/env, doctor/recover/rollback with Web/API/Keycloak/PostgreSQL stopped]}
proof_boundary: {label: N/A, exclusions: [production host mutation, publishing updates]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Create the only component allowed to control Docker and a host-side emergency CLI that remains usable when platform services are unavailable.

# Requirements

- Define allowlisted typed operations for inspect, start/stop/restart, install/update/rollback, plugin lifecycle, backup/restore and diagnostics.
- API submits operations through authenticated service identity; it never receives Docker socket or arbitrary command execution.
- `control-agent` accepts only release-manifest image digests, mounts, environment schema and resource limits.
- Persist a local append-only emergency operation journal independent of PostgreSQL; reconcile API state after recovery.
- `roehubctl` can validate config, show redacted effective state, run `doctor`, initialize owner, manage providers/plugins/artifacts and perform emergency recovery.
- Operations use `operation_id`/idempotency and explicit unknown-state reconciliation.

# Validation

Run focused gates and a real-boundary degraded runtime smoke: stop Web, API, Keycloak and PostgreSQL; use `roehubctl` to inspect, diagnose and restore the allowed topology; prove operation replay/idempotency, crash recovery, image/mount/env rejection, no arbitrary shell and audit reconciliation after API returns.

# Stop rules

Block on Docker access outside `control-agent`, generic shell execution, non-allowlisted images/mounts/env, journal dependence on PostgreSQL, blind operation retry or secret output. Update ledger after evidence.
