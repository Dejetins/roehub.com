---
prompt_name: 17-containerize-all-services-and-profiles
repo: roehub.com
scope: "Containerize every current app, worker and scheduler and assemble complete base/trading/ml profiles with persistent state and readiness."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "17", prerequisites: ["03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"], previous_stage_gate: "Stages 03 through 16 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: runtime and proof boundaries}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: full topology/profile contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: all service prerequisites}
    - {path: docs/architecture/project-map/project-map.json, why: every current app and worker must be covered}
  task_entrypoints:
    - {path: apps/, why: all runnable processes}
    - {path: infra/docker/, why: target Compose and images}
    - {path: configs/, why: generated service config}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: architecture-design, timing: before final topology, reason: failure/load/trust boundaries}
  - {skill: contract-impact-analysis, timing: before ports/defaults, reason: topology/config/readiness changes}
  - {skill: backend-quality-gates, timing: local gates, reason: focused backend checks}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [apps/, infra/docker/, configs/, scripts/, tests/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md, README.md]
  foreign_changes_policy: preserve unrelated application work and use explicit owned files/hunks
file_manifest:
  expected_primary_touches: [apps/, infra/docker/, configs/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/17-containerize-all-services-and-profiles.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [scripts/, docs/runbooks/, docs/architecture/README.md, README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [real docker compose config/build/up runtime smoke, every app/worker/scheduler, base/trading/ml profiles, readiness/dependency order, restart persistence, service-to-service addressing, resource limits]}
proof_boundary: {label: N/A, exclusions: [production deployment, import of current production data, final multi-architecture matrix]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Make the generated container topology complete: no current app, worker or scheduler remains native-only or absent from the release profiles.

# Requirements

- Reconcile current `project-map.json` with all runnable entrypoints and assign each to `base`, `trading`, `ml` or an isolated on-demand job.
- Build reproducible non-root images with pinned bases/dependencies, health endpoints, resource defaults and externalized persistent state.
- Use service DNS, never `127.0.0.1`, for inter-container dependencies.
- Generate Compose from release/config manifests; no user-maintained internal `.env`/Compose.
- `base` clean-starts safe modes and includes Telegram/artifacts; `trading` never implies `mainnet`; `ml` is optional.
- Define startup/readiness/degradation/restart rules without making one optional service crash the whole control plane.

# Validation

Run focused gates and the real docker compose config/build/up runtime smoke for each profile. Prove every declared service readiness, service-to-service connectivity, restart behavior, persisted volumes, safe defaults, resource limits and teardown/restart without data loss. An uncovered component or unavailable Docker boundary is `blocked`, not accepted from static Compose parsing.

# Stop rules

Block on missing app/worker, `latest`, root containers without justification, container-local durable state, localhost coupling, unsafe default mode, ungenerated user edits or failed restart persistence. Update ledger after evidence.
