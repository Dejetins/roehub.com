---
prompt_name: 02-machine-readable-runbooks
repo: roehub.com
scope: "Create the canonical English ops.roehub.io/v1 runbook schema, validation, indexes and Russian user rendering."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "02", prerequisites: ["00"], previous_stage_gate: "Stage 00 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository and documentation contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: machine-readable runbook decision}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: stage gate}
  task_entrypoints:
    - {path: docs/runbooks/, why: current human runbooks}
    - {path: tools/docs/, why: documentation generators}
    - {path: infra/macos/prometheus/, why: existing runbook annotations and alerts}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before schema freeze, reason: operational schema and alert semantics}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [schemas/ops/, tools/docs/, docs/runbooks/, docs/architecture/platform/, docs/architecture/README.md, tests/]
  foreign_changes_policy: preserve unrelated files and hunks
file_manifest:
  expected_primary_touches: [schemas/ops/, tools/docs/, docs/runbooks/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/02-machine-readable-runbooks.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [tests/, docs/architecture/README.md, infra/monitoring/]
validation_strategy: {depth: integration, acceptance_surfaces: [JSON Schema, English canonical YAML, Russian render, JSON index, link integrity, redaction]}
proof_boundary: {label: N/A, exclusions: [runtime incident resolution, Web UI rendering]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Define one agent-readable operational contract and prove it can generate user-readable Russian instructions and a JSON index.

# Requirements

- Define versioned `ops.roehub.io/v1` fields for identity, symptoms, severity, prerequisites, diagnostics, allowed actions, rollback, evidence, secret redaction, owner and related alerts.
- Canonical YAML narrative and machine fields are English; generated user documents are Russian.
- Migrate a representative cross-section: database, worker, auth/OpenBao, exchange execution and Web/API incident.
- Link each problem to a runbook and only to allowlisted typed actions; never embed arbitrary shell or secrets.
- Validate stable IDs, references, action capabilities, locale coverage and deterministic rendering.

# Acceptance

Schema and fixture tests pass; representative YAML validates; generated Russian Markdown and JSON index are deterministic; docs index and `git diff --check` pass. Record unmigrated legacy runbooks as explicit later-stage inventory, not hidden completion.

# Stop rules

Block on ambiguous action authority, raw secret fields, unstable identifiers, a renderer that loses safety warnings, or broken alert/runbook links. Update ledger after evidence and before the Russian report.
