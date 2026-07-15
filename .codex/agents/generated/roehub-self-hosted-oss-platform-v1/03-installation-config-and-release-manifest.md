---
prompt_name: 03-installation-config-and-release-manifest
repo: roehub.com
scope: "Introduce roehub.yaml, base/trading/ml profiles, release manifest and deterministic internal configuration generation."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "03", prerequisites: ["01", "02"], previous_stage_gate: "Stages 01 and 02 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: config and prompt-pack policy}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: profile/config contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: dependencies and handoff}
  task_entrypoints:
    - {path: configs/, why: current product/runtime configuration}
    - {path: infra/docker/docker-compose.yml, why: current Compose baseline}
    - {path: apps/, why: current environment consumers}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before schema/default changes, reason: configuration is a public contract}
  - {skill: backend-quality-gates, timing: verification, reason: generator and config validation}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [schemas/config/, configs/, tools/release/, src/trading/platform/, infra/docker/, tests/, docs/architecture/platform/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated changes and do not rewrite user env files
file_manifest:
  expected_primary_touches: [schemas/config/, configs/, tools/release/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/03-installation-config-and-release-manifest.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [src/trading/platform/, infra/docker/, tests/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [schema validation, deterministic render, real Docker Compose parsing for every profile, disposable config-consumer container, secret rejection, digest pinning]}
proof_boundary: {label: N/A, exclusions: [starting the full platform, current production configuration import]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Make `roehub.yaml` the only user-edited installation configuration and create a deterministic release manifest/config generator for profiles `base`, `trading` and `ml`.

# Requirements

- Separate installation config from PostgreSQL product config and OpenBao secret references.
- Include domain, ports, directories, profiles, embedded/external stores, resource limits, TLS/proxy and opt-in update checks.
- `base` includes notifications/Telegram capability and local artifact storage by default.
- Reject secret-looking raw values, unknown dangerous keys, `latest` images and unsupported architectures.
- Generate Compose fragments, service config, OIDC/OpenBao/Prometheus inputs and a redacted effective-config view.
- Same input plus release manifest must produce byte-stable generated output.
- Inventory current env/config consumers only to remove hidden implementation dependencies. The v1 user contract starts at `roehub.yaml`; no converter for the current installation is required.

# Validation and stop rules

Run schema/property/golden tests, focused ruff/pyright/pytest, deterministic double generation and the profile matrix. The required real-boundary runtime smoke parses every generated profile through the installed real `docker compose config` command and starts a disposable non-networked config-consumer container that loads the generated service configuration and exits successfully. If Docker is unavailable, record the runtime boundary as `blocked`; do not accept the stage from golden tests alone. Also run the docs-index check and `git diff --check`. Block on hidden env dependencies, generated secrets, unpinned images, profile ambiguity or config values that enable `mainnet`.

Update ledger after validation and record all config `breaking-change` surfaces. Do not add compatibility aliases solely for the current installation.
