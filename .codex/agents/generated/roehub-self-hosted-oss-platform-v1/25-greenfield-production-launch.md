---
prompt_name: 25-greenfield-production-launch
repo: roehub.com
status: historical_no_execution_authority
superseded_by: docs/architecture/platform/roehub-product-transformation-requirements-v1.md
scope: "Perform an explicitly approved greenfield production launch of the verified container release without importing current native-system data."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: manual_sequential, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: false, goal_artifact_required: false}
stage: {id: "25", prerequisites: ["24"], previous_stage_gate: "Never. This stage is superseded and must not execute."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: current self-hosted proof contract}
    - {path: docs/architecture/platform/roehub-product-transformation-requirements-v1.md, why: current product requirements baseline}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: superseded stage state}
  task_entrypoints: []
skill_routing: []
change_ownership:
  parallel_main_expected: false
  owned_change_scope: []
  foreign_changes_policy: preserve all files and installations; execution is forbidden
file_manifest: {created: [], modified: [], deleted: []}
validation_strategy: {depth: none, acceptance_surfaces: []}
proof_boundary: {label: N/A, exclusions: [all execution because this prompt is superseded]}
authority: {implementation_write: false, git_publish: false, production_mutation: false}
---

# Historical and non-executable

Do not execute this prompt. Roehub no longer has a central production-host
launch stage. A future installation mutation starts only from a separately
selected current ticket, an explicitly identified user-owned installation and
the proof boundary required by the active delivery contract.

# Stage readiness anchors

- Execution mode: `manual_sequential`, but inert because
  `status=superseded`; no approval can reopen this prompt.
- Ledger state: Stage `25=superseded`, `current_stage=none`.
- File manifest: created `[]`, modified `[]`, deleted `[]`.
- Roehub smoke Keycloak username: `N/A`; historical
  `smoke_e2e_keycloak` MUST NOT be used because no browser authentication runs.
- Host-local smoke password environment source: `N/A`;
  `ROEHUB_SMOKE_E2E_PASSWORD` MUST NOT be read because credential lookup is
  forbidden.
- Credential redaction: no secret, session, installation or provider material
  may be read.
- Proof boundary: `N/A`; no pre-main, post-main, deploy or installation proof
  is valid under this prompt.
- Verdict: stop and require a new executable plan for any future release or
  user-owned installation mutation.
