---
prompt_name: 12-2-functional-canary
repo: roehub.com
branch: main
scope: "Prove real active strategies produce signals/events before load or long soak."
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
      why: "producer polling and signal generation"
    - path: src/trading/contexts/live_execution
      why: "source-event/intent/order ledgers"
    - path: apps/api/routes/strategies.py
      why: "strategy read models"
skill_routing:
  - skill: backend-performance-evidence
    use_when: "summarizing canary latency and runtime deltas"
    timing: during verification
    reason: "canary must produce measured source-event/execution evidence"
  - skill: browser-qa-evidence
    use_when: "capturing /strategies latest signals/journal state"
    timing: during verification
    reason: "user-visible strategy state must match runtime state"
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
  acceptance_surfaces: ["producer-runtime", "database", "redis", "prometheus", "browser"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-2-functional-canary.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "12.2"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
readiness_anchors:
  previous_stage_ledger_gate: "Previous stage prerequisite: before implementation, read the stage ledger and verify Stage 12.1 is accepted in the ledger; record evidence in the Stage 12.2 report."
  file_manifest_required: true
  smoke_keycloak_username: smoke_e2e_keycloak
  host_local_smoke_password_env_var_source: "/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD"
  credential_redaction_rule: "Do not write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output."
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-2-functional-canary.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "Do not replace this canary with synthetic load. Stage 12.2 must prove real active strategies are polled and emit source events/signals."
  - "No mainnet submit; no raw provider payloads or credentials in evidence."
---

# Task

Run Stage `12.2` functional canary for active paper/testnet strategies.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...` and record it in the stage report.
- Previous stage ledger gate: read `stage_execution_ledger.path` before implementation and verify Stage `12.1` is accepted in the ledger; record the ledger evidence in the Stage `12.2` report.
- Verify Stage `12.1` is `accepted`; stop if it is blocked, pending, or superseded.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Credential redaction rule: never write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output.
- For authenticated browser/API smoke, use Keycloak username `smoke_e2e_keycloak`. On `macstudio`, read the password only from `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; outside `macstudio`, use securely exported local `ROEHUB_SMOKE_E2E_PASSWORD`. Do not ask for or print the password.
- Run a bounded 30-60 minute canary with the active strategy runs accepted by Stage `12.1`.
- Prove with DB/API/metrics that producer polling actually happens and that source events/signals appear during the canary window.
- Prove paper/testnet paths write expected rows for the canary scope: source events, intents/risk rows, paper accounting and/or testnet execution rows as appropriate for the selected active runs.
- Record pre/post deltas for Redis pending/retry/DLQ, DB unknown/reconciliation/outbox rows, producer metrics, and execution metrics.
- If no source event/signal appears within the declared canary window, mark `12.2 blocked`; do not proceed to burst or 6h soak.
- Capture `/strategies` browser/API evidence for the active runs and latest signal/journal state.
- If accepted and files changed, publish scoped report/ledger/docs changes through `github:yeet`/`publish-ci-deploy` discipline.

## Acceptance Criteria

- Canary duration is at least 30 minutes and at most 60 minutes unless it blocks early on a critical failure.
- Producer poll counters and DB rows prove the selected active strategies were processed.
- Source events/signals created during the canary are visible in durable DB/API evidence.
- No new mainnet attempt, secret leak, uncontrolled retry/DLQ growth, or unknown unreconciled state appears.
- Ledger marks `12.2 accepted`; `12.3` may start only after this.

## Quality Gates

- Focused local gates only if code/config changed.
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with canary window, active runs, evidence deltas, pass/fail decision, delivery status, and handoff for Stage `12.3`.
