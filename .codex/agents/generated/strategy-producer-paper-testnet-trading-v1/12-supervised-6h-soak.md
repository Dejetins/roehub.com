---
prompt_name: 12-supervised-6h-soak
repo: roehub.com
branch: main
scope: "Run mandatory 6-hour supervised paper/testnet soak with controlled burst load and resource-impact evidence."
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
    use_when: "summarizing 6h latency, throughput, CPU, and RAM evidence"
    timing: during verification
    reason: "soak acceptance relies on measured latency/load/resource metrics"
  - skill: browser-qa-evidence
    use_when: "capturing final UI state after soak"
    timing: during verification
    reason: "user-visible state must be observed"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth and scoped staging; branches/PRs are temporary and must be delivered to main and cleaned up before acceptance"
  - skill: publish-ci-deploy
    use_when: "accepted changes or reports need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["6h-runtime", "controlled-burst-load", "resource-telemetry", "monit", "prometheus", "database", "redis", "browser"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-supervised-6h-soak.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "12"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-supervised-6h-soak.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - apps/worker/strategy_live_runner
  - apps/exchange_execution
  - infra/macos/prometheus
  - docs/runbooks/mac-studio-monitoring-plan.md
  - docs/runbooks/prod-dashboard-metrics-reference-ru.md
  - docs/architecture/README.md
safety_notes:
  - "This stage cannot be accepted with a short smoke. It requires 6 elapsed hours of logged evidence plus controlled burst-load and CPU/RAM evidence."
  - "Do not invent a new resource-monitoring path unless existing Mac Studio monitoring is proven missing and the gap is documented before the soak starts."
---

# Task

Run the mandatory 6-hour supervised paper/testnet soak gate with one controlled amplified-load interval. This is an acceptance stage, not a unit-test stage.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `11` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `github:yeet`/`publish-ci-deploy` discipline for scoped publish, but do not leave the stage on a per-stage branch or draft PR. Temporary branches are allowed only when useful; before marking the stage `accepted`, deliver the changes to `main`, push `origin main`, verify main contains the changes, delete any temporary local/remote branch, and record main-branch delivery plus branch-cleanup evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Before starting, define exact start time, planned end time, baseline strategy count, controlled burst strategy count, burst start/end plan, mode mix, exchanges/markets, kill switches, cleanup plan, and pass/fail thresholds.
- Run for 6 elapsed hours unless a critical blocker stops the gate. The source of truth is durable telemetry/logs/stage report evidence, not a continuously open agent chat session.
- Include one controlled amplified-load interval inside the 6-hour window. Reuse the accepted Stage `11` load harness or an equivalent existing harness; keep the run `paper`/`testnet` only, respect internal/exchange rate limiters, and do not create a mainnet or real-provider-HTTP load path.
- Discover and record the existing Mac Studio monitoring commands/queries before the soak starts. Required resource evidence must use existing Prometheus/Monit/node-exporter/service metrics or already accepted benchmark-monitoring methods, including host CPU/load/memory and process-level CPU/RSS where available for `com.roehub.strategy-live-runner`, `com.roehub.exchange-execution`, and `apps/api`. If CPU/RAM telemetry cannot be obtained from existing monitoring, block acceptance before starting the soak and document the gap instead of accepting with a TODO.
- Record Monit uptime, Prometheus snapshots, DB counts, Redis pending/DLQ/retry, strategy producer metrics, exchange-execution metrics, CPU/RAM snapshots, controlled burst impact, and final browser status.
- Capture at minimum baseline/pre-burst, during-burst, post-burst, and final snapshots. For each snapshot, include the actual Prometheus queries, Monit commands, SQL/Redis commands, and summarized values used for the decision.
- Mainnet attempts, secret leakage, unreconciled unknown order state, runaway retry loop, or uncontrolled service crash are blockers.
- Sustained CPU/RAM saturation, OOM/restart storm, unbounded process RSS growth, or burst impact that does not return to the predeclared acceptable band is a blocker unless the stage is explicitly marked blocked with evidence.

## Acceptance Criteria

- Stage report contains 6h start/end timestamps and evidence snapshots across the whole window.
- Stage report contains baseline, burst, post-burst, and final CPU/RAM evidence from existing monitoring, including the exact commands/queries and thresholds used.
- SQL/Redis/Prometheus prove no hidden backlog or unknown unreconciled state beyond documented thresholds.
- Controlled burst evidence proves the platform respects limiter/backpressure contracts and returns to an acceptable resource/queue state.
- Browser shows final user-visible strategy status.
- Ledger marks Stage 12 accepted only if the full 6h gate, controlled burst, resource telemetry, and cleanup pass.

## Quality Gates

- Pre-soak local gates from changed areas.
- `python -m tools.docs.generate_docs_index --check`
- 6h runtime evidence, controlled burst evidence, and CPU/RAM impact evidence as specified above.

## Final Output

Russian report with 6h evidence, controlled burst result, CPU/RAM impact, pass/fail decision, blockers/residual risk, cleanup status, delivery status, and handoff.
