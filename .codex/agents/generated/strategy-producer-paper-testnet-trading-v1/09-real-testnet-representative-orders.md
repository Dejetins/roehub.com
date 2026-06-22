---
prompt_name: 09-real-testnet-representative-orders
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Execute representative real testnet orders for supported Binance/Bybit spot/futures direction and sizing groups on BTCUSDT."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and Mac Studio runtime rules"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
      why: "plan"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "stage handoff"
  task_entrypoints:
    - path: apps/exchange_execution
      why: "native adapter and exchange boundary"
    - path: src/trading/contexts/live_execution
      why: "source/risk/order/reconciliation ledger"
    - path: src/trading/contexts/exchange_control
      why: "testnet credential/account-state boundary"
    - path: infra/macos/launchd/com.roehub.exchange-execution.plist
      why: "exchange-execution runtime supervision"
skill_routing:
  - skill: root-cause-debugging
    use_when: "testnet order/adapter/reconciliation proof fails"
    timing: if blocker
    reason: "money-boundary failures require root-cause evidence"
  - skill: backend-performance-evidence
    use_when: "reporting latency/slippage numbers"
    timing: during verification
    reason: "latency claims need comparable measurement evidence"
  - skill: publish-ci-deploy
    use_when: "accepted changes need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["real-testnet-exchange", "database", "redis", "metrics", "monit"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/09-real-testnet-representative-orders.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "09"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - apps/exchange_execution
  - src/trading/contexts/live_execution
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/09-real-testnet-representative-orders.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - src/trading/contexts/exchange_control
  - tests/unit/apps/exchange_execution
  - tests/unit/contexts/live_execution
  - docs/architecture/README.md
safety_notes:
  - "Only testnet orders are allowed. Mainnet attempts are critical blockers."
  - "Futures short requires safe isolated 1x proof. If missing, use only an explicit testnet account-config operator command with read-back proof; never mutate settings implicitly during order submit."
---

# Task

Run representative real testnet order coverage for `BTCUSDT`: Binance/Bybit, spot/futures, supported long/short branches, and sizing groups derived from Stage 03. Use the existing execution path from source event through risk, Redis, exchange-execution, native adapter, order/fill/reconciliation, and outbox.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `08` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `publish-ci-deploy` direct-main discipline for scoped publish on `main`; do not create a branch, draft PR, worktree, temporary checkout, local folder, stash, or auxiliary workflow artifact unless the user explicitly requests that exact workflow. Before marking the stage `accepted`, verify `origin/main` contains the changes and record main-branch delivery evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Use user-added testnet keys from `/settings`; do not ask for or store secrets.
- Keep allocation `$50` per strategy and respect min-notional/precision.
- For futures short, require account-state proof of isolated margin and leverage `1x`. Proof may come from read-only account-state or from an explicit testnet account-config command followed by read-back; the command must not place orders.
- Treat spot-short as blocked/unsupported unless a real margin/borrow product is already implemented and accepted; do not submit fake spot-short orders.
- Record latency timestamps: signal/source event, intent, risk, dispatch, submit, ack, fill/reconcile.
- Unknown exchange state must be reconciled before any retry.
- Cancel/close testnet positions where appropriate and record final state.

## Acceptance Criteria

- Stage report contains a representative matrix with pass/block result for every required exchange/market/direction/sizing group, including explicit blocked proof for spot-short.
- At least one real testnet order path per accepted representative bucket has DB order/fill/reconciliation evidence or a documented provider-specific terminal status.
- Redis ack-after-durable, pending, retry, DLQ evidence is recorded.
- Metrics include submit latency, limiter waits, errors, private stream/reconciliation where applicable.
- No mainnet submit and no secret leakage.

## Quality Gates

- `uv run ruff check apps/exchange_execution src/trading/contexts/live_execution src/trading/contexts/exchange_control tests`
- `uv run pyright apps/exchange_execution src/trading/contexts/live_execution src/trading/contexts/exchange_control tests`
- `uv run pytest -q tests/unit/apps/exchange_execution tests/unit/contexts/live_execution tests/unit/contexts/exchange_control`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with representative matrix, real testnet evidence, latency/slippage, blockers, cleanup/position state, and handoff.
