---
prompt_name: 12-plugin-platform-api-and-sdk
repo: roehub.com
scope: "Create the extensions bounded context, signed plugin package/instance lifecycle, permissions, isolated RPC, public API and SDK foundation."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "12", prerequisites: ["03", "05", "08"], previous_stage_gate: "Stages 03, 05 and 08 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: plugin and service-integration rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: accepted plugin architecture}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: config/org/secrets prerequisites}
  task_entrypoints:
    - {path: src/trading/integration/, why: current integration package and target wire-contract ownership}
    - {path: src/trading/contexts/notifications/application/ports/notification_provider.py, why: first provider-shaped port}
    - {path: apps/api/, why: plugin management API}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: architecture-design, timing: before extension boundaries, reason: package/instance/RPC ownership}
  - {skill: contract-impact-analysis, timing: before public contract, reason: manifest/API/version/permission semantics}
  - {skill: backend-quality-gates, timing: verification, reason: backend/SDK gates}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/extensions/, src/trading/integration/, apps/api/, apps/plugin_gateway/, migrations/, schemas/plugins/, sdk/, tools/plugins/, tests/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated ports and do not bulk-convert every adapter in this stage
file_manifest:
  expected_primary_touches: [src/trading/contexts/extensions/, src/trading/integration/, apps/api/, migrations/, schemas/plugins/, sdk/, tools/plugins/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/12-plugin-platform-api-and-sdk.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [apps/plugin_gateway/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [manifest/schema, signature/digest, package versus instance, permission diff, real-boundary isolated plugin runtime smoke, RPC negotiation, API idempotency, Python/TypeScript SDK conformance]}
proof_boundary: {label: N/A, exclusions: [public marketplace, production plugin install, arbitrary execution plugins]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Create `Plugin API v1alpha1` as a public boundary without importing third-party code into the API process.

# Requirements

- Define `roehub.plugin.yaml`, package/version/installation/instance/permission/operation/event models and additive persistence.
- Verify digest, signature, publisher key, license, SBOM, Roehub/API compatibility, architectures and requested permissions.
- `admin` manages plugins within granted scope; permission expansion requires `recent-auth` and audit. Trust-key changes remain reserved.
- Run backend code in a non-root read-only OCI container with limits, no Docker socket, no platform DB, explicit egress and short-lived service identity.
- Expose typed asynchronous management operations and per-capability endpoints; no generic arbitrary `/execute`.
- Publish schema/OpenAPI/RPC definitions, Python and TypeScript SDK scaffolds, conformance fixtures and offline validation tooling.
- Unsigned development mode is explicit, unavailable to `mainnet` and disabled by default.

# Validation

Run focused gates plus a real-boundary runtime smoke that validates, installs and calls a signed fixture plugin in an isolated container, proves permission denials/network/filesystem limits, negotiates protocol version, records health/metrics/audit and rolls back. If the container boundary cannot run, mark `blocked`.

# Stop rules

Block on in-process third-party imports, arbitrary shell/mount/env, direct DB access, signature bypass, raw secrets, unversioned contracts or an admin permission escalation without audit/recent-auth. Update ledger after evidence.
