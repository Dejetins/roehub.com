---
prompt_name: 25-greenfield-production-launch
repo: roehub.com
scope: "Perform an explicitly approved greenfield production launch of the verified container release without importing current native-system data."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: manual_sequential, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: false, goal_artifact_required: false}
stage: {id: "25", prerequisites: ["24"], previous_stage_gate: "Stage 24 accepted and the current user explicitly authorized a greenfield production launch in a new request."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: publish/deploy/Mac Studio proof contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: greenfield launch and rollback policy}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: release candidate and approval gate}
  task_entrypoints:
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/24-platform-matrix-release-candidate.md, why: required release candidate evidence}
    - {path: docs/runbooks/, why: launch, backup and rollback operations}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: verify explicit manual approval gate}
  - {skill: publish-ci-deploy, timing: after approval and local readiness, reason: owns main/CI/deploy/post-deploy lifecycle}
  - {skill: contract-impact-analysis, timing: before launch, reason: production topology and new state-source activation}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [deployment workflows and runtime artifacts explicitly authorized by the current user, docs/architecture/platform/, docs/runbooks/, README.md, .codex/PLANS.md, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated work and the current native installation; do not include either in delivery
file_manifest:
  expected_primary_touches: [docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/25-greenfield-production-launch.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [deployment workflows explicitly authorized, docs/runbooks/, README.md, .codex/PLANS.md, docs/architecture/README.md]
validation_strategy: {depth: delivery, acceptance_surfaces: [main revision, green CI, signed release, empty-store bootstrap, post_main_production_runtime_proof, browser/API/database/runtime/observability, rollback decision]}
proof_boundary: {label: post_main_production_runtime_proof, exclusions: [current native-state import, unapproved real-money canary, deleting or modifying old state]}
authority: {implementation_write: false_without_new_user_approval, git_publish: false_without_new_user_approval, production_mutation: false_without_new_user_approval}
---

# Mandatory approval gate

This stage is not part of autonomous goal execution. If the current user has not explicitly authorized a greenfield production launch after Stage `24`, do not mutate files or runtime. Record/retain the approval blocker and stop.

# Objective after approval

Deliver the verified revision through the repository publication workflow, initialize empty production stores, bootstrap the new installation, prove runtime behavior and preserve a tested rollback window inside the new self-hosted lifecycle.

# Requirements

- Revalidate Stage `24`, destination capacity, launch plan, user communication, empty-store precondition and rollback thresholds.
- The current native PostgreSQL, ClickHouse, Redis, Keycloak, OpenBao and artifact state must not be read, copied, reconciled, mounted, changed or used as rollback state.
- For any remote SQL, JSON or multiline bootstrap payload, use quoted heredoc/stdin such as `<<'SQL'`, `<<'JSON'`, `--queries-file /dev/stdin` or `query=@-`; nested inline payload quoting and temporary files created only to bypass quoting are forbidden.
- Use `publish-ci-deploy`: target revision on `main`, green relevant CI/image/deploy workflows, verified sync/deploy, then runtime proof.
- Bootstrap `installation_owner`, organizations, providers and initial resources only through the new supported contracts. Keep `mainnet` fail closed unless separately approved; no real-money canary is implied.
- Prove API/Web/auth/admin/plugins/jobs/observability and backup readiness under `post_main_production_runtime_proof`.
- If existing endpoints are reassigned, treat that traffic switch as a separate explicitly approved operation; absence of data continuity is intentional and must be communicated before launch.
- Keep the old native installation untouched. It may be archived or retired only by a later separately authorized plan.

# Stop and rollback rules

Rollback the new release on non-empty destination state, critical health/auth/tenant/execution/backup failure, unknown side effects or violated thresholds. Rollback means restore the previous new-platform release/backup or stop the new installation; it never imports or mutates old native state. Update the ledger and Russian report with revision, CI/deploy/runtime evidence, sanitized bootstrap evidence, rollback decision and residual risks.
