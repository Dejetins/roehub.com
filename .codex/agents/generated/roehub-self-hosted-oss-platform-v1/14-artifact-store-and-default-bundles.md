---
prompt_name: 14-artifact-store-and-default-bundles
repo: roehub.com
scope: "Implement ArtifactStore/v1 with local content-addressed storage by default, S3 compatibility, signed manifests and a small demo bundle."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "14", prerequisites: ["04", "09", "12"], previous_stage_gate: "Stages 04, 09 and 12 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: persistence/performance contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: artifact state ownership}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: storage/tenancy/plugin prerequisites}
  task_entrypoints:
    - {path: src/trading/contexts/backtest_artifacts/, why: filesystem artifact implementation}
    - {path: src/trading/contexts/rl_trading/, why: model registry and backup paths}
    - {path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md, why: current artifact runtime contract}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: architecture-design, timing: before artifact boundary, reason: immutable blob/catalog/materialization responsibilities}
  - {skill: contract-impact-analysis, timing: before identity/layout changes, reason: artifact addresses, manifests and persistence}
  - {skill: backend-quality-gates, timing: verification, reason: focused gates}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/backtest_artifacts/, src/trading/contexts/rl_trading/, src/trading/integration/, apps/api/, apps/worker/, migrations/, configs/, tests/, tools/artifacts/, docs/architecture/backtest/, docs/architecture/ml/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated artifact and RL model work; current artifact stores are outside the target and must not be read or deleted
file_manifest:
  expected_primary_touches: [src/trading/contexts/backtest_artifacts/, src/trading/integration/, migrations/, tools/artifacts/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/14-artifact-store-and-default-bundles.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [src/trading/contexts/rl_trading/, apps/api/, apps/worker/, configs/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [local CAS end-to-end smoke, digest verification, atomic publish, pin/lease/quota/GC, materialization cache, backup/restore, S3-compatible adapter fixture]}
proof_boundary: {label: N/A, exclusions: [reading or copying current Mac Studio artifact state, production deletion]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Replace path-shaped artifact coupling with immutable content-addressed blobs and manifests while retaining local materialization for mmap-heavy compute.

# Requirements

- Define `ArtifactStore/v1`: put/get, digest, manifest/catalog metadata, atomic publish, pin/lease, quota, GC, backup and local materialization capabilities.
- Ship `local_cas` by default; add an S3-compatible adapter behind the same contract.
- Store catalog/ownership in PostgreSQL and bytes outside containers; secrets only as OpenBao refs.
- Provide a small signed demo bundle with known digests; do not import the current RL/backtest corpus.
- Support `roehubctl artifacts install <bundle>` use case for later CLI wiring.

# Validation

Run focused gates and a real-boundary local CAS end-to-end smoke: concurrent atomic publish, corruption rejection, materialization, restart persistence, pin/lease/GC, quota, backup/restore and signed demo-bundle installation. Exercise the S3 adapter against a controlled compatible fixture. Measure hot-path materialization impact when touched.

# Stop rules

Block on mutable artifact identity, container-local state, digest bypass, GC of leased data, non-atomic publish, cross-org catalog leakage or dependency on current artifact paths. Update ledger after evidence.
