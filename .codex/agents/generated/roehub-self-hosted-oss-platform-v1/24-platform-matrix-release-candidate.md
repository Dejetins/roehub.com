---
prompt_name: 24-platform-matrix-release-candidate
repo: roehub.com
status: historical_no_execution_authority
superseded_by: docs/architecture/platform/roehub-product-transformation-requirements-v1.md
scope: "Validate the complete greenfield release candidate on Linux amd64, Linux arm64 and Docker Desktop macOS including MacBook Pro M3 Pro clean-install and lifecycle paths."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: manual_sequential, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: false, goal_artifact_required: false}
stage: {id: "24", prerequisites: ["22", "23"], previous_stage_gate: "Never. This stage is superseded and must not execute."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: QA, browser, performance and proof boundaries}
    - {path: docs/architecture/platform/roehub-product-transformation-requirements-v1.md, why: current product requirements baseline}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: superseded stage state}
  task_entrypoints: []
skill_routing: []
change_ownership:
  parallel_main_expected: false
  owned_change_scope: []
  foreign_changes_policy: preserve all files; execution is forbidden
file_manifest: {created: [], modified: [], deleted: []}
validation_strategy: {depth: none, acceptance_surfaces: []}
proof_boundary: {label: N/A, exclusions: [all runtime, browser, release and installation actions because this prompt is superseded]}
authority: {implementation_write: false, git_publish: false, production_mutation: false, external_registry_write: false}
---

# Historical and non-executable

Do not execute this prompt. Its native-host matrix and continuation rules are
historical. Product requirements are preserved in
`docs/architecture/platform/roehub-product-transformation-requirements-v1.md`,
but new release certification begins only from a separately selected current
ticket under the active delivery contract.

# Stage readiness anchors

- Execution mode: historical only; no stage gate can open;
  no stage gate can open.
- Ledger state: Stage `24=superseded`, `current_stage=none`.
- File manifest: created `[]`, modified `[]`, deleted `[]`.
- Roehub smoke Keycloak username: `N/A`; historical
  `smoke_e2e_keycloak` MUST NOT be used because no browser authentication runs.
- Host-local smoke password environment source: `N/A`;
  `ROEHUB_SMOKE_E2E_PASSWORD` MUST NOT be read because credential lookup is
  forbidden.
- Credential redaction: no secrets, cookies, sessions or provider payloads may
  be read because all execution is forbidden.
- Verdict: stop and follow the superseding master; do not record a new Stage
  `24` result.
