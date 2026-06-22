---
prompt_name: 12-3-burst-resource-gate
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Run controlled burst/load and prove CPU/RAM/Redis/DB recovery without replacing the functional canary."
language:
  implementation: python/shell/markdown
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and Mac Studio path rules"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
      why: "plan"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "stage handoff"
  task_entrypoints:
    - path: apps/exchange_execution
      why: "Stage 11 load harness and execution metrics"
    - path: infra/macos/prometheus
      why: "resource and queue metrics"
    - path: docs/runbooks/prod-dashboard-metrics-reference-ru.md
      why: "existing CPU/RAM PromQL"
skill_routing:
  - skill: backend-performance-evidence
    use_when: "collecting comparable baseline/during/post resource and latency evidence"
    timing: during verification
    reason: "burst/resource acceptance is performance evidence"
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
  acceptance_surfaces: ["stage11-load-harness", "prometheus", "redis", "database", "monit"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-3-burst-resource-gate.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "12.3"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
readiness_anchors:
  previous_stage_ledger_gate: "Previous stage prerequisite: before implementation, read the stage ledger and verify Stage 12.2 is accepted in the ledger; record evidence in the Stage 12.3 report."
  file_manifest_required: true
  smoke_keycloak_username: smoke_e2e_keycloak
  host_local_smoke_password_env_var_source: "/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD"
  credential_redaction_rule: "Do not write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output."
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-3-burst-resource-gate.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "Use the accepted Stage 11 harness or equivalent existing harness. Do not invent a new load path."
  - "This gate is not accepted if the burst replaces real strategy canary evidence from Stage 12.2."
---

# Task

Run Stage `12.3` burst/resource gate.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...` and record it in the stage report.
- Previous stage ledger gate: read `stage_execution_ledger.path` before implementation and verify Stage `12.2` is accepted in the ledger; record the ledger evidence in the Stage `12.3` report.
- Verify Stage `12.2` is `accepted`; stop if it is not.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Credential redaction rule: never write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output.
- Before burst, record baseline CPU/RAM/process RSS, Redis, DB, Monit, Prometheus, producer metrics, and exchange-execution metrics.
- Run the controlled burst through the accepted Stage `11` load harness with explicit Mac Studio tool paths (`/opt/homebrew/bin/uv`, `/opt/homebrew/bin/redis-cli`, `/opt/homebrew/bin/psql`) or sourced runtime env.
- Keep the load `paper`/`testnet` only, with no mainnet and no real-provider-HTTP load path.
- Capture during-burst and post-burst snapshots. Post-burst must prove queues/resource usage return to the predeclared acceptable band.
- Record exact PromQL/commands, values, thresholds, and deltas.
- If the harness fails, retry/DLQ grows, Redis pending remains nonzero, CPU/RAM saturation persists, or resource usage does not recover, mark `12.3 blocked`.
- If accepted and files changed, publish scoped report/ledger/docs changes through `publish-ci-deploy` direct-main discipline.

## Acceptance Criteria

- Harness result is successful with `violations=[]` or documented equivalent accepted output.
- CPU/RAM/process RSS/Redis/DB deltas are recorded before, during, and after burst.
- Redis pending returns to `0`; retry/DLQ/unknown/reconciliation deltas remain within declared thresholds.
- Ledger marks `12.3 accepted`; `12.4` may start only after this.

## Quality Gates

- Focused local gates only if code/config changed.
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with burst configuration, resource deltas, threshold decision, blockers/residual risk, delivery status, and handoff for Stage `12.4`.
