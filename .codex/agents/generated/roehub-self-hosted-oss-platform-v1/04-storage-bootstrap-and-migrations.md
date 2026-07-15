---
prompt_name: 04-storage-bootstrap-and-migrations
repo: roehub.com
scope: "Unify PostgreSQL and ClickHouse bootstrap/migrations and implement embedded/external storage profiles for clean installation."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "04", prerequisites: ["03"], previous_stage_gate: "Stage 03 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: persistence and verification contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: state ownership and storage profiles}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: stage gate}
  task_entrypoints:
    - {path: apps/migrations/, why: migration application}
    - {path: migrations/postgres/, why: PostgreSQL schema history}
    - {path: migrations/clickhouse/, why: ClickHouse schema history}
    - {path: infra/docker/, why: embedded database topology}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before migrations, reason: persisted schema and bootstrap semantics}
  - {skill: backend-quality-gates, timing: verification, reason: migration implementation gates}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [apps/migrations/, migrations/, infra/docker/, schemas/config/, tests/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated migrations and never rewrite applied history silently
file_manifest:
  expected_primary_touches: [apps/migrations/, migrations/, infra/docker/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/04-storage-bootstrap-and-migrations.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [schemas/config/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: integration, acceptance_surfaces: [fresh PostgreSQL, fresh ClickHouse, idempotent rerun, partial-failure recovery, external profile readiness, schema-version reporting]}
proof_boundary: {label: N/A, exclusions: [production databases, import or repair of current state]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Prove that all required schemas self-bootstrap on a clean container installation and that embedded/external PostgreSQL, ClickHouse and Redis profiles fail closed with clear readiness.

# Requirements

- One ordered migration/bootstrap command covers PostgreSQL and ClickHouse with durable migration state.
- Never use Redis as the only migration or durable truth.
- Test fresh install, idempotent rerun and interrupted migration recovery from empty stores.
- Define capability checks for external stores; do not pretend arbitrary database engines are compatible.
- Preserve applied migration history inside the new self-hosted lifecycle; never read or transform the current production database.
- Add backup prerequisites and schema-version reporting for `roehubctl` consumers.

# Acceptance and stop rules

Focused gates pass plus real disposable database evidence for PostgreSQL and ClickHouse. Block on non-idempotent DDL, data loss, container-local persistence, localhost service addressing, manual pre-created volumes or an external profile that can start without schema/readiness proof. Update ledger and Stage report after validation.
