---
prompt_name: 01-open-source-license-and-release-governance
repo: roehub.com
scope: "Establish Apache-2.0 project governance, security policy, version source, dependency notices and release metadata."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "01", prerequisites: ["00"], previous_stage_gate: "Stage 00 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: release and license decisions}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: predecessor evidence and gate}
  task_entrypoints:
    - {path: LICENSE, why: current empty license file}
    - {path: pyproject.toml, why: package version and dependencies}
    - {path: .github/workflows/, why: current release and security automation}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before metadata changes, reason: public distribution contract}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [LICENSE, NOTICE, SECURITY.md, CONTRIBUTING.md, pyproject.toml, tools/release/, .github/workflows/, docs/architecture/platform/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated changes; stage only owned hunks if publishing is later authorized
file_manifest:
  expected_primary_touches: [LICENSE, NOTICE, SECURITY.md, CONTRIBUTING.md, pyproject.toml, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/01-open-source-license-and-release-governance.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [tools/release/, .github/workflows/, docs/architecture/README.md]
validation_strategy: {depth: integration, acceptance_surfaces: [license text, third-party license inventory, version source, preliminary SBOM, security reporting flow]}
proof_boundary: {label: N/A, exclusions: [publishing a release, external registry mutation]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Turn Roehub into a legally and operationally coherent Apache-2.0 open-source project without publishing anything.

# Requirements

- Fill `LICENSE` with the official Apache-2.0 text and add a correct `NOTICE` policy.
- Add contributor and vulnerability-reporting guidance without promising an unavailable private channel.
- Replace version `0.0.0` with one canonical version source and define SemVer/release-manifest compatibility.
- Inventory direct dependencies, container images, fonts/assets and bundled binaries; classify licenses and block incompatible distribution.
- Add deterministic preliminary SBOM and third-party notice generation/checking if repository tooling permits.
- Keep update checks and telemetry opt-in; do not add network calls.
- Do not publish packages, images, tags or releases.

# Validation and acceptance

Run focused format/tests for any new tooling, license/SBOM checks, package metadata build inspection, docs-index generation/check and `git diff --check`. Acceptance requires no unresolved incompatible dependency and a Stage `01` report with exact evidence and unknown transitive-license risks.

# Ledger and stop rules

Update the ledger after validation. Stop with `blocked` on license incompatibility, multiple competing version sources, unredacted security contact data or a release workflow that would publish during validation.
