---
prompt_name: 10-trading-context-organization-isolation
repo: roehub.com
scope: "Add organization and account isolation to strategy, risk, exchange-control and live-execution while preserving fail-closed paper/testnet behavior."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "10", prerequisites: ["05", "08"], previous_stage_gate: "Stages 05 and 08 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: money-moving and unknown-state rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: trading tenancy and execution boundary}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: predecessor evidence}
  task_entrypoints:
    - {path: src/trading/contexts/strategy/, why: strategies and intents}
    - {path: src/trading/contexts/risk/, why: risk ownership}
    - {path: src/trading/contexts/exchange_control/, why: accounts and credentials}
    - {path: src/trading/contexts/live_execution/, why: orders and reconciliation}
    - {path: apps/exchange_execution/, why: execution composition}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before trading identity changes, reason: side effects, idempotency and persistence}
  - {skill: backend-quality-gates, timing: verification, reason: focused gates}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/strategy/, src/trading/contexts/risk/, src/trading/contexts/exchange_control/, src/trading/contexts/live_execution/, apps/exchange_control/, apps/exchange_execution/, apps/worker/strategy_live_runner/, migrations/, tests/, docs/architecture/platform/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated mainnet and exchange work; no real orders
file_manifest:
  expected_primary_touches: [src/trading/contexts/strategy/, src/trading/contexts/risk/, src/trading/contexts/exchange_control/, src/trading/contexts/live_execution/, migrations/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/10-trading-context-organization-isolation.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [apps/exchange_control/, apps/exchange_execution/, apps/worker/strategy_live_runner/, docs/architecture/README.md]
validation_strategy: {depth: integration, acceptance_surfaces: [two-organization paper end-to-end, safe testnet smoke when approved credentials exist, account ownership, negative authorization, unknown-state reconciliation]}
proof_boundary: {label: N/A, exclusions: [mainnet activation, real-money orders, production credentials]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Ensure strategies, accounts, risk policies, intents, orders and executions are organization-scoped before plugin and container work.

# Requirements

- Carry server-validated `organization_id` and account identity through strategy → intent → risk → execution.
- Resolve exchange secret references only inside trusted exchange-control/execution boundaries.
- Define client-order/idempotency identity with a versioned organization/account namespace from the first v1 write; no alias to current records is required.
- Enforce `research/paper/testnet` defaults; this stage cannot enable `mainnet`.
- Any unknown submit status must reconcile provider state before retry.
- Create strategies/accounts/orders only with explicit same-organization ownership and enforce referential integrity; do not backfill current records.

# Validation

Run focused gates and a two-organization paper end-to-end smoke through real application boundaries, including negative authorization, account mismatch, risk denial, duplicate intent and unknown-state simulation/reconciliation. A bounded testnet smoke may run only from an approved host-local credential source and must not be required when unavailable; record that boundary honestly.

# Stop rules

Block on cross-org access, exchange keys outside trusted components, blind retry, `mainnet` activation, orphan/cross-owner fresh order state, dependency on current production data or raw provider payloads in evidence. Update ledger after validation.
