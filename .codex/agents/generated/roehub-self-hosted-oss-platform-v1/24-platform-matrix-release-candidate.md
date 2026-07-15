---
prompt_name: 24-platform-matrix-release-candidate
repo: roehub.com
scope: "Validate the complete greenfield release candidate on Linux amd64, Linux arm64 and Docker Desktop macOS including MacBook Pro M3 Pro clean-install and lifecycle paths."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "24", prerequisites: ["22", "23"], previous_stage_gate: "Stages 22 and 23 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: QA, browser, performance and proof boundaries}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: final acceptance matrix}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: all predecessor evidence}
    - {path: docs/architecture/project-map/project-map.json, why: final component coverage}
  task_entrypoints:
    - {path: tools/release/, why: release candidate bundle}
    - {path: tests/, why: validation suites}
    - {path: apps/web/, why: browser flows}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate and completion rule}
  - {skill: backend-quality-gates, timing: backend gates, reason: focused then broad checks}
  - {skill: backend-performance-evidence, timing: performance matrix, reason: comparable backtest/ML/runtime evidence}
  - {skill: browser-qa-evidence, timing: browser matrix, reason: real user/admin flows}
  - {skill: pre-ship-gate, timing: final assessment, reason: release readiness without publishing}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [tests/, tools/release/, scripts/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated changes; repair only release-candidate blockers within explicit owned scope
file_manifest:
  expected_primary_touches: [tests/, tools/release/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/24-platform-matrix-release-candidate.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [scripts/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [Linux amd64 runtime smoke, Linux arm64 runtime smoke, macOS M3 Pro runtime smoke, clean install, backup/restore/release-upgrade, real browser user/admin QA, API/database/security, plugin isolation, benchmark and no-phone-home]}
proof_boundary: {label: N/A, exclusions: [publishing the release, production deployment/cutover, native MPS guarantee]}
authority: {implementation_write: true, git_publish: false, production_mutation: false, external_registry_write: false}
---

# Objective

Issue an evidence-backed `Release`, `Release after fixes` or `Block` verdict for the local release candidate; do not publish it.

# Requirements

- Test the signed/offline candidate on Linux `amd64`, Linux `arm64` and Docker Desktop macOS; MacBook Pro 14 M3 Pro is mandatory.
- Cover clean install, bootstrap owner, local auth, organizations/RBAC, providers/Telegram stub, plugins/data source/panels, artifacts/jobs, paper/testnet-safe trading, admin/control-agent/roehubctl, observability and no-phone-home.
- Cover the Stage `23` greenfield lifecycle artifact, backup/restore, and release-to-release upgrade/rollback. For the first v1 candidate, use the accepted versioned previous-schema fixture when no published `N-1` exists.
- Run real browser user/admin flows with console/network, accessibility and responsive evidence.
- Compare backtest/fastpath and ML CPU baselines; report macOS CPU-only limitations without claiming MPS.
- Reconcile every current project-map component and every release-manifest service.
- Linux `amd64` and Linux `arm64` require native execution evidence. Emulation may be recorded as supplemental diagnostic evidence but cannot satisfy the required platform acceptance surface.

# Browser authentication and evidence safety

- Clean-install acceptance uses disposable local identities and disposable OIDC-provider identities created in each isolated target installation. Current Keycloak users and current production credentials are out of scope.
- If the approved source or required local bootstrap path is unavailable, mark that browser surface `blocked`; never ask for credentials in chat.
- Passwords, recovery codes, WebAuthn material, cookies, tokens, authorization headers, private organization/trading data, secret-bearing logs and raw provider responses are forbidden in screenshots, traces, command output, reports and the ledger.

# Validation and verdict

Use real runtime smoke on each platform, real browser QA, API/database integration, security/isolation probes, restore/upgrade drills and comparable benchmarks. Tests are gates only. A missing required platform or boundary is `Block`, not an inferred pass. Record sanitized artifacts and exact environment/version metadata.

This stage is assessment and evidence closure, not an unbounded repair stage. Any release-candidate defect outside the explicit owned paths must block Stage `24` and receive a new bounded repair prompt or an explicit prompt/file-manifest update before edits.

# Stop rules

Block on any critical defect, unverified platform, data loss, failed rollback, auth/tenant/plugin/mainnet boundary issue, phone-home, missing component or unverifiable package/signature. Stage `24=accepted` completes the autonomous implementation goal; Stage `25` remains disallowed without new user approval.
