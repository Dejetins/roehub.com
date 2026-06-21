---
prompt_name: 12-5-closure
repo: roehub.com
branch: main
scope: "Close Stage 12 after all sub-gates pass: browser proof, cleanup, ledger, docs, delivery, and pass/fail decision."
language:
  implementation: markdown/shell
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
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports
      why: "Stage 12 evidence reports"
    - path: apps/web/dist/js/pages/strategies.js
      why: "browser-visible strategy state if UI proof is needed"
skill_routing:
  - skill: browser-qa-evidence
    use_when: "capturing final /strategies browser proof"
    timing: during verification
    reason: "Stage 12 closure requires user-visible status proof"
  - skill: pre-ship-gate
    use_when: "checking scoped docs/report readiness before publish"
    timing: before ship
    reason: "closure is a release-readiness handoff"
  - skill: github:yeet
    use_when: "accepted Stage 12 docs/report changes need GitHub publish"
    timing: before ship
    reason: "successful stage must be delivered to main"
  - skill: publish-ci-deploy
    use_when: "accepted changes need shipping and Mac Studio/runtime status notation"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["browser", "docs", "ledger", "cleanup", "delivery"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-5-closure.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "12.5"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
readiness_anchors:
  previous_stage_ledger_gate: "Previous stage prerequisite: before implementation, read the stage ledger and verify Stage 12.1, Stage 12.2, Stage 12.3, and Stage 12.4 are accepted in the ledger; record evidence in the Stage 12.5 report."
  file_manifest_required: true
  smoke_keycloak_username: smoke_e2e_keycloak
  host_local_smoke_password_env_var_source: "/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD"
  credential_redaction_rule: "Do not write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output."
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-5-closure.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  - docs/architecture/README.md
possible_secondary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
safety_notes:
  - "Do not mark Stage 12 accepted unless 12.1, 12.2, 12.3, and 12.4 are all accepted."
  - "Old monolithic 12-supervised-6h-soak evidence remains historical/superseded and must not be counted as acceptance."
---

# Task

Run Stage `12.5` closure.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...` and record it in the stage report.
- Previous stage ledger gate: read `stage_execution_ledger.path` before implementation and verify Stage `12.1`, `12.2`, `12.3`, and `12.4` are all accepted in the ledger; record the ledger evidence in the Stage `12.5` report.
- Verify `12.1`, `12.2`, `12.3`, and `12.4` are all `accepted`; stop if any are blocked/pending/superseded.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Credential redaction rule: never write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output.
- Collect final browser proof for `/strategies` with Keycloak username `smoke_e2e_keycloak`. On `macstudio`, read the password only from `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; outside `macstudio`, use securely exported local `ROEHUB_SMOKE_E2E_PASSWORD`. Do not ask for or print passwords/cookies/tokens.
- Confirm cleanup: no stale collector, no unexpected active temp process, Redis pending/retry/DLQ deltas within accepted thresholds, no new unknown/reconciliation debt beyond accepted thresholds, no mainnet attempt, no secret leakage.
- Summarize all Stage `12` sub-gate evidence and produce a final Stage `12` pass/fail decision.
- Update the ledger so Stage `12.5` is accepted or blocked and Stage `13` is allowed only if `12.5 accepted`.
- Run docs index check and publish scoped report/ledger/docs changes through `github:yeet`/`publish-ci-deploy` discipline. Do not stage unrelated dirty files.

## Acceptance Criteria

- Final report references each accepted sub-gate report and states the overall Stage `12` decision.
- Browser proof, cleanup proof, docs index, and delivery evidence are recorded.
- Ledger unblocks Stage `13` only when Stage `12.5` is accepted.

## Quality Gates

- `python -m tools.docs.generate_docs_index --check`
- `git status --short --branch` and explicit staged path list before any commit/push.

## Final Output

Russian closure report with overall Stage `12` decision, evidence index, cleanup status, delivery status, and handoff to Stage `13`.
