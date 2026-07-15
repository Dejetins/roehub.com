---
prompt_name: 16-execution-gateway-and-mainnet-safety
repo: roehub.com
scope: "Enforce the final intent-to-order security boundary, mainnet domain approval, risk, kill-switch, idempotency and reconcile-before-retry semantics."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "16", prerequisites: ["10", "11", "12", "15"], previous_stage_gate: "Stages 10, 11, 12 and 15 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: money-moving and proof rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: mainnet policy and execution boundary}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: trading/plugin/job prerequisites}
    - {path: docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md, why: current execution semantics}
    - {path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, why: current safety evidence}
  task_entrypoints:
    - {path: src/trading/contexts/live_execution/, why: intent/order/reconciliation}
    - {path: src/trading/contexts/risk/, why: pre-submit risk}
    - {path: apps/exchange_execution/, why: trusted gateway}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before execution changes, reason: money-moving idempotency and unknown state}
  - {skill: backend-quality-gates, timing: verification, reason: focused gates}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/live_execution/, src/trading/contexts/risk/, src/trading/contexts/exchange_control/, src/trading/contexts/strategy/, apps/exchange_execution/, apps/exchange_control/, apps/worker/strategy_live_runner/, migrations/, configs/, tests/, docs/architecture/live_execution/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated mainnet work and never place unapproved real-money orders
file_manifest:
  expected_primary_touches: [src/trading/contexts/live_execution/, src/trading/contexts/risk/, apps/exchange_execution/, migrations/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/16-execution-gateway-and-mainnet-safety.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [src/trading/contexts/exchange_control/, src/trading/contexts/strategy/, apps/exchange_control/, apps/worker/strategy_live_runner/, configs/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: integration, acceptance_surfaces: [exchange emulator end-to-end, approved testnet smoke, mode gate, recent-auth, risk, kill-switch, idempotency, unknown response reconciliation, audit and alerts]}
proof_boundary: {label: N/A, exclusions: [mainnet enablement, real-money canary, production exchange mutation]}
authority: {implementation_write: true, git_publish: false, production_mutation: false, real_money_orders: false}
---

# Objective

Make `mainnet` a persisted owner-approved domain policy rechecked immediately before submit, never a Compose profile or environment toggle.

# Requirements

- Canonical intent includes organization, account, instrument, side, size, constraints and idempotency identity.
- Trusted gateway verifies organization/account, mode, owner approval/recent-auth, risk, kill-switch, adapter capability and secret reference immediately before submit.
- Unknown submit state is persisted and reconciled from provider order/private-stream state before any retry.
- Mainnet approval is revocable, time/audit bound where designed, and invalidated by material risk/account/plugin changes.
- Execution providers are core/verified allowlisted only; general third-party plugins cannot submit orders.
- Every transition and denied action is auditable without secret/provider payload leakage.

# Validation

Run focused gates, exchange emulator end-to-end scenarios and a bounded testnet smoke only when approved host-local credentials exist. Cover duplicate intent, timeout-before/after provider acceptance, restart, stale approval, risk denial, kill-switch during flight, reconciliation and alert/runbook actions. No real-money order is authorized.

# Stop rules

Any blind retry, env/Compose mainnet toggle, bypassable risk/kill-switch, strategy access to keys, untrusted execution plugin or unresolved unknown state is `blocked`. Update ledger after evidence.
