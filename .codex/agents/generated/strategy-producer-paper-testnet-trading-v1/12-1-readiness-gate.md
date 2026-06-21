---
prompt_name: 12-1-readiness-gate
repo: roehub.com
branch: main
scope: "Prove Stage 12 readiness before any functional canary, burst, or 6h soak starts."
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
    - path: apps/worker/strategy_live_runner
      why: "strategy producer runtime"
    - path: apps/api/routes/strategies.py
      why: "strategy run/readiness API"
    - path: apps/exchange_execution
      why: "execution runtime readiness"
    - path: infra/scripts/monit
      why: "runtime supervision"
    - path: infra/macos/prometheus
      why: "metrics/rules"
skill_routing:
  - skill: backend-performance-evidence
    use_when: "recording CPU/RAM/resource baseline and monitoring queries"
    timing: during verification
    reason: "later gates compare against this baseline"
  - skill: browser-qa-evidence
    use_when: "checking /strategies user-visible active strategy state"
    timing: during verification
    reason: "readiness includes browser-visible strategy status"
  - skill: github:yeet
    use_when: "accepted docs/report changes need GitHub publish"
    timing: before ship
    reason: "successful gates must not remain only in local worktree"
  - skill: publish-ci-deploy
    use_when: "accepted report/ledger changes need main delivery and host-sync notation"
    timing: before ship
    reason: "record CI/deploy/runtime handoff when applicable"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["macstudio", "api", "database", "redis", "monit", "prometheus", "browser"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-1-readiness-gate.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "12.1"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
readiness_anchors:
  previous_stage_ledger_gate: "Previous stage prerequisite: before implementation, read the stage ledger and verify Stage 11 is accepted in the ledger; record evidence in the Stage 12.1 report."
  file_manifest_required: true
  smoke_keycloak_username: smoke_e2e_keycloak
  host_local_smoke_password_env_var_source: "/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD"
  credential_redaction_rule: "Do not write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output."
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-1-readiness-gate.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "Fail fast. Do not start canary, burst, or 6h soak when producer is disabled, allowlists are empty, or running_strategy_runs is 0."
  - "Do not ask for secrets in chat; use host-local env and redact cookies/tokens/DSNs."
---

# Task

Run Stage `12.1` readiness gate. This gate proves the platform is ready to start functional canary and soak. It must not run the 6h soak or the burst harness.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. Record it in the stage report.
- Previous stage ledger gate: read `stage_execution_ledger.path` before implementation and verify Stage `11` is accepted in the ledger; record the ledger evidence in the Stage `12.1` report.
- Verify Stage `11` is `accepted`. Treat the old monolithic Stage `12` attempt as superseded/blocked evidence, not as acceptance.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Credential redaction rule: never write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output.
- For authenticated browser/API smoke, use Keycloak username `smoke_e2e_keycloak`. On `macstudio`, read the password only from `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; outside `macstudio`, use securely exported local `ROEHUB_SMOKE_E2E_PASSWORD`. Do not ask for or print the password.
- Confirm no active stale Stage `12` collector/process is still running. If one is running, stop and report a blocker unless the user explicitly asked to resume/stop it.
- Use Mac Studio path rules: remote git checks only under `/Users/daniildegtyarev/Projects/roehub.com`; runtime checks may use `/opt/roehub/app`; use explicit `/opt/homebrew/bin/*` tools or sourced host env.
- Prove readiness with real calls:
  - producer health/readiness says enabled for scoped paper/testnet operation;
  - producer allowlists are non-empty and include the selected user/strategy runs;
  - selected paper/testnet strategy runs exist and `running_strategy_runs > 0`;
  - `/strategies` or API/browser shows active strategy state for those runs;
  - Redis, Postgres, Monit, Prometheus, node-exporter, strategy-producer metrics, and exchange-execution metrics are reachable;
  - exchange-execution remains `testnet` and no mainnet path is enabled;
  - CPU/RAM baseline queries and process RSS sampling are available.
- If any readiness condition fails, write a blocked report and ledger update; do not start Stage `12.2`.
- If accepted and files changed, publish scoped docs/report changes through `github:yeet`/`publish-ci-deploy` discipline. Runtime sync may be `N/A docs/report-only` unless code/config changed.

## Acceptance Criteria

- Stage report contains the exact selected user/run/profile/strategy identifiers, redacted where needed.
- SQL/API proves `running_strategy_runs > 0` and at least the required active paper/testnet coverage for Stage `12.2`.
- Prometheus/Monit/Redis/Postgres/browser checks are recorded with commands/queries and summarized values.
- Ledger marks `12.1 accepted`; `12.2` may start only after this.

## Quality Gates

- `python -m tools.docs.generate_docs_index --check`
- No secrets in report, logs, screenshots, ledger, or final output.

## Final Output

Russian report with readiness verdict, blockers if any, evidence links/summaries, delivery status, and handoff for Stage `12.2`.
