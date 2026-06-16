---
prompt_name: 12-supervised-24h-soak
repo: roehub.com
branch: main
scope: "Run mandatory 24-hour supervised paper/testnet soak and record acceptance evidence."
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
      why: "strategy producer runtime"
    - path: apps/exchange_execution
      why: "order runtime"
    - path: infra/scripts/monit
      why: "runtime supervision"
    - path: infra/macos/prometheus
      why: "metrics/rules"
skill_routing:
  - skill: backend-performance-evidence
    use_when: "summarizing 24h latency and throughput evidence"
    timing: during verification
    reason: "soak acceptance relies on measured metrics"
  - skill: browser-qa-evidence
    use_when: "capturing final UI state after soak"
    timing: during verification
    reason: "user-visible state must be observed"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth, stages scoped files, commits, pushes, and opens a draft PR"
  - skill: publish-ci-deploy
    use_when: "accepted changes or reports need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["24h-runtime", "monit", "prometheus", "database", "redis", "browser"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-supervised-24h-soak.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "12"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-supervised-24h-soak.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - apps/worker/strategy_live_runner
  - apps/exchange_execution
  - infra/macos/prometheus
  - docs/architecture/README.md
safety_notes:
  - "This stage cannot be accepted with a short smoke. It requires 24 elapsed hours of logged evidence."
---

# Task

Run the mandatory 24-hour supervised paper/testnet soak gate. This is an acceptance stage, not a unit-test stage.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `11` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, publish using `github:yeet`; do not mark the stage `accepted` until the stage report and ledger record main-branch delivery evidence and, for runtime/code stages, Mac Studio host sync/deploy smoke. Use `publish-ci-deploy` only for CI/deploy/host-sync work that `github:yeet` does not cover.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Before starting, define exact start time, planned end time, strategy count, mode mix, exchanges/markets, kill switches, cleanup plan, and pass/fail thresholds.
- Run for 24 elapsed hours unless a critical blocker stops the gate.
- Record Monit uptime, Prometheus snapshots, DB counts, Redis pending/DLQ/retry, strategy producer metrics, exchange-execution metrics, and final browser status.
- Mainnet attempts, secret leakage, unreconciled unknown order state, runaway retry loop, or uncontrolled service crash are blockers.

## Acceptance Criteria

- Stage report contains 24h start/end timestamps and evidence snapshots across the whole window.
- SQL/Redis/Prometheus prove no hidden backlog or unknown unreconciled state beyond documented thresholds.
- Browser shows final user-visible strategy status.
- Ledger marks Stage 12 accepted only if the full 24h gate passes.

## Quality Gates

- Pre-soak local gates from changed areas.
- `python -m tools.docs.generate_docs_index --check`
- 24h runtime evidence as specified above.

## Final Output

Russian report with 24h evidence, pass/fail decision, blockers/residual risk, cleanup status, delivery status, and handoff.
