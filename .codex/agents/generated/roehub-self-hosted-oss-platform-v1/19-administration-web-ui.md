---
prompt_name: 19-administration-web-ui
repo: roehub.com
scope: "Build the separate administration Web UI for organizations, roles, plugins, operational state, backups and updates over typed APIs."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "19", prerequisites: ["05", "06", "12", "18"], previous_stage_gate: "Stages 05, 06, 12 and 18 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: browser/auth/UI rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: admin role and operation contracts}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: API/control prerequisites}
    - {path: docs/architecture/apps/web/web-ui-design-manifest-v1.md, why: existing visual contract}
  task_entrypoints:
    - {path: apps/web/, why: Web application and design tokens}
    - {path: apps/api/, why: typed admin APIs}
    - {path: src/trading/contexts/operations/, why: operation state}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: ui-ux-pro-max, timing: before implementation, reason: dense institutional admin UX and accessibility}
  - {skill: contract-impact-analysis, timing: before routes/defaults, reason: role and browser-visible behavior}
  - {skill: browser-qa-evidence, timing: verification, reason: real browser role and operation flows}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [apps/web/, apps/api/, src/trading/contexts/identity/, src/trading/contexts/extensions/, src/trading/contexts/operations/, tests/, docs/architecture/apps/web/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated product UI and use existing design vocabulary
file_manifest:
  expected_primary_touches: [apps/web/, apps/api/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/19-administration-web-ui.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [src/trading/contexts/identity/, src/trading/contexts/extensions/, src/trading/contexts/operations/, docs/architecture/apps/web/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: browser, acceptance_surfaces: [real browser owner/admin/operator/trader/viewer flows, role/plugin management, permission diff/recent-auth, secret redaction, loading/empty/error/degraded/success, responsive and accessibility]}
proof_boundary: {label: N/A, exclusions: [production admin mutation, replacing roehubctl emergency path]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Provide a high-quality separate administration surface while keeping typed API and `roehubctl` as recoverable boundaries.

# Requirements

- Admin navigation covers installation status, organizations, members/roles, plugin packages/instances/permissions, provider accounts, backups, updates, audit and allowed service actions.
- `admin` can manage roles and plugins within granted scope; last-owner, installation-owner, trust-key and mainnet invariants remain protected.
- Privileged mutations show impact, require confirmation and `recent-auth`, submit an asynchronous operation and display durable progress/unknown/rollback state.
- Never show secret values, raw provider payloads, Docker commands or unrestricted log data.
- Preserve dense institutional drill-down, semantic tokens, predictable navigation and complete loading/empty/error/degraded/success/disabled states.

# Browser authentication and evidence safety

- Create disposable local identities for the target role matrix. When checking compatibility with the existing Keycloak-backed runtime, use username `smoke_e2e_keycloak` and obtain its password only from host-local `ROEHUB_SMOKE_E2E_PASSWORD` in `/Users/daniildegtyarev/.config/roehub/roehub.env`.
- Never request credentials in chat. Passwords, recovery codes, cookies, tokens, authorization headers, secret references, private organization data and raw logs must not appear in screenshots, traces, command output, reports or the ledger.

# Validation

Run source gates and real browser QA for every role, recent-auth success/failure, last-owner denial, plugin permission diff, operation progress/recovery, secret redaction, keyboard/focus, table/chart alternatives, light/dark/reduced motion and 375/768/1024/1440 widths. Inspect console and failed network requests.

# Stop rules

Block on a role bypass, missing recent-auth, secret exposure, Docker access from browser/API, inaccessible critical flow, responsive overflow or an operation that cannot be reconciled after unknown status. Update ledger after evidence.
