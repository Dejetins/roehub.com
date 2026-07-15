---
prompt_name: 22-signed-offline-release-bundle
repo: roehub.com
scope: "Build reproducible multi-architecture images and a signed online/offline release bundle installable without Git and without phone-home behavior."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "22", prerequisites: ["01", "03", "17", "21"], previous_stage_gate: "Stages 01, 03, 17 and 21 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: release/delivery authority rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: product release unit and platforms}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: release prerequisites}
  task_entrypoints:
    - {path: .github/workflows/, why: current image/build workflows}
    - {path: tools/release/, why: release generation}
    - {path: infra/docker/, why: images and generated Compose}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before release freeze, reason: version/manifest/install compatibility}
  - {skill: pre-ship-gate, timing: final local assessment, reason: readiness evidence without publishing}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [.github/workflows/, tools/release/, infra/docker/, schemas/release/, scripts/, tests/, README.md, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated delivery workflows and do not push/publish/tag
file_manifest:
  expected_primary_touches: [tools/release/, infra/docker/, schemas/release/, tests/, README.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/22-signed-offline-release-bundle.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [.github/workflows/, scripts/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [reproducible images, linux/amd64 and linux/arm64 manifests, signature/SBOM/provenance, offline bundle, real-boundary air-gapped install runtime smoke, no-phone-home network observation]}
proof_boundary: {label: N/A, exclusions: [publishing images/releases, claiming final platform matrix acceptance]}
authority: {implementation_write: true, git_publish: false, production_mutation: false, external_registry_write: false}
---

# Objective

Produce the installable product bundle locally without publishing it.

# Requirements

- Build pinned multi-architecture OCI image manifests, release manifest, checksums, signatures, SBOM, NOTICE, schemas, generated Compose/config, `roehubctl`, migrations, runbooks and small signed demo artifacts.
- No `latest`, branch images, mutable dependencies or Git checkout dependency.
- Support online image references and a fully offline bundle containing required images/assets.
- Verify signatures/digests before activation and show clear compatibility/errors.
- Default installation makes no telemetry, analytics, update or catalog requests; update check remains explicit opt-in.
- Reproducibility differences must be explained and bounded.

# Validation

Build twice where feasible, compare artifacts, inspect architecture manifests/signatures/SBOM/licenses, and run a real-boundary air-gapped install runtime smoke with network egress denied and no Git metadata. Observe attempted connections and fail on any undeclared phone-home. Do not publish/tag/push.

# Stop rules

Block on mutable image references, missing architecture, unsigned/unchecked component, incomplete offline content, undeclared egress, missing license notice or install dependence on source checkout. Update ledger after evidence.
