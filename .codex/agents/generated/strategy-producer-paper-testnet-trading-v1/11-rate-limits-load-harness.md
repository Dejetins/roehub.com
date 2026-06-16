---
prompt_name: 11-rate-limits-load-harness
repo: roehub.com
branch: main
scope: "Implement and run controlled load coverage for dozens/hundreds of testnet-mode strategies with limiter/backpressure evidence."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
      why: "plan"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "stage handoff"
  task_entrypoints:
    - path: apps/worker/strategy_live_runner
      why: "strategy producer load surface"
    - path: apps/exchange_execution
      why: "exchange adapter rate limits/backpressure"
    - path: src/trading/contexts/live_execution
      why: "dispatch, risk, order ledger and metrics"
    - path: infra/macos/prometheus
      why: "metrics/rules"
skill_routing:
  - skill: backend-performance-evidence
    use_when: "designing and reporting load/latency measurements"
    timing: before and during verification
    reason: "load and latency claims need measured evidence"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth and scoped staging; branches/PRs are temporary and must be delivered to main and cleaned up before acceptance"
  - skill: publish-ci-deploy
    use_when: "accepted changes need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces: ["load-run", "redis", "metrics", "database", "exchange-testnet-or-controlled-adapter"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/11-rate-limits-load-harness.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "11"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - apps/worker/strategy_live_runner
  - apps/exchange_execution
  - src/trading/contexts/live_execution
  - infra/macos/prometheus
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/11-rate-limits-load-harness.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - tests/unit/apps
  - tests/unit/contexts/live_execution
  - docs/architecture/README.md
safety_notes:
  - "Load must respect exchange and internal limits; do not intentionally DDoS testnet endpoints."
---

# Task

Implement and run a controlled load harness for dozens/hundreds of testnet-mode strategies. The goal is not to hit exchange limits; the goal is to prove Roehub respects internal/exchange rate limits, backpressure, queue lag, retry budgets, and latency measurement under realistic strategy counts.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `10` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `github:yeet`/`publish-ci-deploy` discipline for scoped publish, but do not leave the stage on a per-stage branch or draft PR. Temporary branches are allowed only when useful; before marking the stage `accepted`, deliver the changes to `main`, push `origin main`, verify main contains the changes, delete any temporary local/remote branch, and record main-branch delivery plus branch-cleanup evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Acceptance requires testnet-mode strategy load. Paper load may be used only as a supporting baseline and must not replace testnet-mode evidence.
- Respect per-exchange limiter and record limiter wait/backpressure metrics.
- Measure signal-to-source, source-to-intent, risk, dispatch, submit, ack/fill, Redis pending, DLQ, CPU/memory where available.
- Define pass/fail thresholds in the stage report before interpreting results.
- Stop and clean up testnet strategies/orders/positions where applicable.

## Acceptance Criteria

- Load run evidence includes testnet-mode strategy count, duration, mode mix, request/order count, p95/p99 latencies, Redis lag, DLQ/retry counts, limiter waits, CPU/memory notes.
- No mainnet submit, no secret leakage, no uncontrolled retry loop.
- Any bottleneck is recorded as blocker or residual risk, not hidden.

## Quality Gates

- `uv run ruff check apps src/trading/contexts/live_execution tests`
- `uv run pyright apps src/trading/contexts/live_execution tests`
- `uv run pytest -q tests/unit/apps tests/unit/contexts/live_execution`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with load method, raw metrics summary, thresholds, pass/fail decision, delivery status, and handoff.
