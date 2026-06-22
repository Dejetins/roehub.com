---
prompt_name: 06-supervised-strategy-producer
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Harden the existing strategy_live_runner as supervised strategy producer runtime with launchd/Monit, allowlists, admin switch, and metrics."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and Mac Studio runtime paths"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
      why: "plan"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "stage handoff"
  task_entrypoints:
    - path: apps/worker/strategy_live_runner
      why: "existing strategy runtime"
    - path: src/trading/contexts/strategy/application/ports/execution_producer.py
      why: "producer-to-execution boundary"
    - path: infra/scripts/monit
      why: "service supervision"
    - path: infra/macos/launchd
      why: "launchd service definitions"
skill_routing:
  - skill: backend-quality-gates
    use_when: "running focused Python gates"
    timing: during verification
    reason: "service code needs lint/type/test evidence"
  - skill: publish-ci-deploy
    use_when: "accepted service changes need deployment"
    timing: before ship
    reason: "runtime service must be deployed and smoke-checked on Mac Studio"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["launchd", "monit", "prometheus", "api-health", "database", "redis"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/06-supervised-strategy-producer.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "06"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - apps/worker/strategy_live_runner
  - src/trading/contexts/strategy
  - infra/scripts/monit
  - infra/macos/launchd
  - infra/macos/prometheus
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/06-supervised-strategy-producer.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - tests/unit/apps
  - tests/unit/contexts/strategy
  - docs/architecture/operations
  - docs/architecture/README.md
safety_notes:
  - "Default producer switch must be disabled or allowlist-bound until stages prove scenarios."
  - "No mainnet execution may be enabled."
---

# Task

Harden `apps/worker/strategy_live_runner` into the supervised strategy producer runtime for this plan. It must run paper/testnet strategies through the existing execution producer boundary, be controlled by admin switch and per-user/per-strategy allowlists, and be supervised by launchd/Monit with Prometheus metrics.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `05` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `publish-ci-deploy` direct-main discipline for scoped publish on `main`; do not create a branch, draft PR, worktree, temporary checkout, local folder, stash, or auxiliary workflow artifact unless the user explicitly requests that exact workflow. Before marking the stage `accepted`, verify `origin/main` contains the changes and record main-branch delivery evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Reuse `apps/worker/strategy_live_runner` as the producer. Do not create a new app/process unless you first prove a blocker, update the architecture plan and ledger, and record a specific decision explaining why reuse is unsafe.
- Add or harden runtime config for `paper`/`testnet` only.
- Add per-user/per-strategy allowlist and admin kill switch.
- Expose health/readiness/metrics for producer state, last cycle, lag, errors, skipped blocked strategies, source-event count, and latency timestamps.
- Prove service stop/restart, fail-closed disabled state, allowlist block/allow, and no mainnet path.
- Update runbook/stage report/ledger.

## Acceptance Criteria

- Mac Studio launchd and Monit show the producer service loaded and controlled.
- `/metrics` exposes bounded producer metrics with no user/order high-cardinality labels.
- A controlled allowed strategy can produce a source event without direct exchange SDK access.
- Disabled switch or missing allowlist blocks production and records a reason.

## Quality Gates

- `uv run ruff check apps/worker/strategy_live_runner src/trading/contexts/strategy tests`
- `uv run pyright apps/worker/strategy_live_runner src/trading/contexts/strategy tests`
- `uv run pytest -q tests/unit/apps tests/unit/contexts/strategy`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with runtime evidence, Monit/Prometheus proof, blocked/allowed cases, delivery status, and handoff.
