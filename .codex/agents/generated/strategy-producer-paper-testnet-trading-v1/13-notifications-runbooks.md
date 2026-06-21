---
prompt_name: 13-notifications-runbooks
repo: roehub.com
branch: main
scope: "Finalize notification outbox compatibility, alert severity/owner/escalation, and operator runbooks."
language:
  implementation: python/yaml/markdown
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
    - path: src/trading/contexts/live_execution/domain/notification.py
      why: "notification outbox contract"
    - path: apps/api/routes/ui_execution.py
      why: "notification/status API"
    - path: infra/macos/prometheus/rules
      why: "alert rules"
    - path: docs/architecture/operations
      why: "runbook area"
skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing outbox/event DTOs"
    timing: during implementation
    reason: "future delivery services depend on event contract"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth and scoped staging; branches/PRs are temporary and must be delivered to main and cleaned up before acceptance"
  - skill: publish-ci-deploy
    use_when: "accepted changes need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces: ["outbox-api-db", "prometheus-rules", "runbook-drill"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/13-notifications-runbooks.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "13"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
readiness_anchors:
  previous_stage_ledger_gate: "Previous stage prerequisite: before implementation, read the stage ledger and verify Stage 12.5 is accepted in the ledger and Stage 12.1 through Stage 12.4 are accepted in the ledger; record evidence in the Stage 13 report."
  file_manifest_required: true
  smoke_keycloak_username: smoke_e2e_keycloak
  host_local_smoke_password_env_var_source: "/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD"
  credential_redaction_rule: "Do not write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output."
expected_primary_touches:
  - src/trading/contexts/live_execution
  - apps/api/routes/ui_execution.py
  - infra/macos/prometheus/rules
  - docs/architecture/operations
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/13-notifications-runbooks.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - tests/unit/contexts/live_execution
  - tests/unit/apps/api
  - docs/architecture/README.md
safety_notes:
  - "Delivery channels such as Telegram/email are out of scope; only outbox/event compatibility is in scope."
---

# Task

Finalize delivery-neutral notification outbox compatibility and operator alert/runbook coverage for paper/testnet strategy producer operations.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Previous stage ledger gate: read `stage_execution_ledger.path` before implementation and verify Stage `12.5` is accepted in the ledger, and verify `12.1` through `12.4` are also accepted in the ledger; record the ledger evidence in the Stage `13` report.
- Before implementation, verify Stage `12.5` is `accepted` in the stage ledger, and verify `12.1` through `12.4` are also accepted. Stop if any Stage `12.x` gate is blocked, pending, skipped, or superseded unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `github:yeet`/`publish-ci-deploy` discipline for scoped publish, but do not leave the stage on a per-stage branch or draft PR. Temporary branches are allowed only when useful; before marking the stage `accepted`, deliver the changes to `main`, push `origin main`, verify main contains the changes, delete any temporary local/remote branch, and record main-branch delivery plus branch-cleanup evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Credential redaction rule: never write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output.
- If authenticated browser/API smoke is needed for notification surfaces, use Keycloak username `smoke_e2e_keycloak`. On `macstudio`, read the password only from `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; outside `macstudio`, use securely exported local `ROEHUB_SMOKE_E2E_PASSWORD`. Do not ask for or print the password.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Produce outbox events for rejected signal/order, fill, manual exit, kill switch, unknown/reconciliation pending, strategy stopped/restarted, 6h soak failure/success, and resource-threshold breach during soak/load.
- Keep event payloads redacted and compatible with future Telegram/email delivery.
- Add/verify Prometheus alert rules with severity, owner, escalation, and runbook action.
- Run at least one runbook drill or dry-run for a non-destructive alert path.

## Acceptance Criteria

- API/SQL proves outbox rows for representative terminal and incident states.
- Prometheus rule check passes and rule names/severities are documented.
- Runbook drill evidence is recorded.
- Browser/API shows user-visible notification/status where applicable.

## Quality Gates

- `uv run ruff check src/trading/contexts/live_execution apps tests`
- `uv run pyright src/trading/contexts/live_execution apps tests`
- `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps`
- Prometheus rule validation command available in repo/runtime.
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with outbox evidence, alert/runbook matrix, delivery-scope note, contract impact, and handoff.
