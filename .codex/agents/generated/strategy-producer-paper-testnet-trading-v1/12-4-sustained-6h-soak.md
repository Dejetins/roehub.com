---
prompt_name: 12-4-sustained-6h-soak
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Run the actual 6-hour sustained soak with active strategies after readiness, canary, and burst gates pass."
language:
  implementation: python/shell/markdown
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
      why: "active strategy producer"
    - path: apps/exchange_execution
      why: "execution runtime"
    - path: infra/scripts/monit
      why: "service supervision"
    - path: infra/macos/prometheus
      why: "metrics"
skill_routing:
  - skill: backend-performance-evidence
    use_when: "summarizing 6h latency/resource evidence"
    timing: during verification
    reason: "soak acceptance depends on comparable measurements"
  - skill: browser-qa-evidence
    use_when: "capturing final and/or periodic /strategies state"
    timing: during verification
    reason: "user-visible runtime state must match ledgers"
  - skill: github:yeet
    use_when: "accepted docs/report changes need GitHub publish"
    timing: before ship
    reason: "successful gates must be delivered to main"
  - skill: publish-ci-deploy
    use_when: "accepted changes need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["6h-runtime", "active-strategies", "prometheus", "monit", "database", "redis", "browser"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "12.4"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
readiness_anchors:
  previous_stage_ledger_gate: "Previous stage prerequisite: before implementation, read the stage ledger and verify Stage 12.3 is accepted in the ledger; record evidence in the Stage 12.4 report."
  file_manifest_required: true
  smoke_keycloak_username: smoke_e2e_keycloak
  host_local_smoke_password_env_var_source: "/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD"
  credential_redaction_rule: "Do not write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output."
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "Do not run the 6h soak when active strategies disappear, producer is disabled, or allowlists are empty."
  - "The source of truth is durable telemetry/logs/stage report evidence, not a continuously open chat session."
---

# Task

Run Stage `12.4` sustained 6-hour soak.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...` and record it in the stage report.
- Previous stage ledger gate: read `stage_execution_ledger.path` before implementation and verify Stage `12.3` is accepted in the ledger; record the ledger evidence in the Stage `12.4` report.
- Verify Stage `12.3` is `accepted`; stop if it is not.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Credential redaction rule: never write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output.
- If authenticated browser/API proof is collected in this gate, use Keycloak username `smoke_e2e_keycloak`. On `macstudio`, read the password only from `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; outside `macstudio`, use securely exported local `ROEHUB_SMOKE_E2E_PASSWORD`. Do not ask for or print the password.
- Reconfirm the Stage `12.1` readiness invariants immediately before starting: producer enabled, allowlists non-empty, selected active strategy runs still running, and telemetry available.
- Run for 6 elapsed hours with active paper/testnet strategies. Do not count idle time with `running_strategy_runs = 0` toward acceptance.
- Collect periodic snapshots at start, at least hourly, and final. Each snapshot must include Monit, Prometheus, CPU/RAM/process RSS, producer/execution metrics, Redis pending/retry/DLQ, DB source-event/intent/order/reconciliation/outbox deltas, and active strategy state.
- Track deltas relative to the Stage `12.4` start baseline, not only absolute historical counts.
- No new unknown unreconciled state, retry/DLQ growth beyond thresholds, mainnet attempt, secret leak, uncontrolled crash, or sustained resource saturation is allowed.
- Capture final browser/API state for `/strategies` or defer final browser proof to `12.5` only if `12.4` records the exact reason and API evidence is complete.
- If accepted and files changed, publish scoped report/ledger/docs changes through `publish-ci-deploy` direct-main discipline.

## Acceptance Criteria

- Full 6 elapsed hours are covered by durable snapshots while active strategies remain running.
- Producer metrics and DB deltas show active strategy processing during the window.
- Redis/DB/Prometheus/Monit show no hidden backlog, unknown state growth, or sustained resource pressure.
- Ledger marks `12.4 accepted`; `12.5` may start only after this.

## Quality Gates

- Focused local gates only if code/config changed.
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with 6h window, snapshot table, deltas, blockers/residual risk, delivery status, and handoff for Stage `12.5`.
