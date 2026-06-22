---
prompt_name: 14-final-readiness-docs-closure
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Close the paper/testnet strategy producer cycle with docs, ledger, prompt-pack audit, CI/deploy evidence, and next-plan handoff."
language:
  implementation: markdown/python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
      why: "plan"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "stage acceptance state"
  task_entrypoints:
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports
      why: "all stage reports"
    - path: .codex/agents/generated/strategy-producer-paper-testnet-trading-v1
      why: "prompt pack to audit"
    - path: docs/architecture/README.md
      why: "docs index"
skill_routing:
  - skill: architecture-review
    use_when: "auditing plan/stage docs for gaps before closure"
    timing: before final acceptance
    reason: "closure is an architecture/documentation consistency review"
  - skill: prompt-manager
    use_when: "auditing or updating prompt pack files"
    timing: during implementation
    reason: "prompt pack completeness is part of closure"
  - skill: publish-ci-deploy
    use_when: "accepted closure changes need publishing/deploy verification"
    timing: before ship
    reason: "final closure must record CI/deploy/runtime status if changes ship"
validation_strategy:
  depth: ci_deploy
  e2e_required: true
  acceptance_surfaces: ["docs", "ledger", "prompt-pack", "ci-deploy", "runtime-smoke-if-code-changed"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/14-final-readiness-docs-closure.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "14"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
readiness_anchors:
  previous_stage_ledger_gate: "Previous stage prerequisite: before implementation, read the stage ledger and verify Stage 13 is accepted in the ledger and Stage 12.1 through Stage 12.5 are accepted in the ledger; record evidence in the Stage 14 report."
  file_manifest_required: true
  smoke_keycloak_username: smoke_e2e_keycloak
  host_local_smoke_password_env_var_source: "/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD"
  credential_redaction_rule: "Do not write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output."
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/14-final-readiness-docs-closure.md
  - .codex/agents/generated/strategy-producer-paper-testnet-trading-v1
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "Do not mark the cycle complete unless all dependent stages are accepted or explicitly superseded."
---

# Task

Close the `strategy-producer-paper-testnet-trading-v1` cycle. Audit plan, stage reports, ledger, and prompt pack for gaps; update docs; record final go/no-go and handoff to the future mainnet real-money plan.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Previous stage ledger gate: read `stage_execution_ledger.path` before implementation and verify Stage `13` is accepted in the ledger; also verify Stage `12.1`, `12.2`, `12.3`, `12.4`, and `12.5` are all accepted in the ledger; record the ledger evidence in the Stage `14` report.
- Before implementation, verify Stage `13` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Also verify Stage `12.1`, `12.2`, `12.3`, `12.4`, and `12.5` are all `accepted`; the old monolithic Stage `12` row may be `superseded` only and must not be counted as Stage `12` acceptance.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `publish-ci-deploy` direct-main discipline for scoped publish on `main`; do not create a branch, draft PR, worktree, temporary checkout, local folder, stash, or auxiliary workflow artifact unless the user explicitly requests that exact workflow. Before marking the stage `accepted`, verify `origin/main` contains the changes and record main-branch delivery evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Credential redaction rule: never write secrets, passwords, cookies, tokens, DSNs, exchange keys, raw credentials, or session values to reports, logs, screenshots, ledger, commits, or final output.
- If authenticated browser/API smoke is needed for final proof, use Keycloak username `smoke_e2e_keycloak`. On `macstudio`, read the password only from `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; outside `macstudio`, use securely exported local `ROEHUB_SMOKE_E2E_PASSWORD`. Do not ask for or print the password.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Verify every stage has an accepted/blocker/superseded status with concrete evidence.
- Ensure no prompt or report relies on tests-only acceptance for non-trivial runtime/browser/exchange stages.
- Ensure docs state mainnet remains out of scope and future real-money plan is separate.
- Regenerate/check docs index.
- If changes are delivered, record commit, CI, deploy, and runtime smoke status.

## Acceptance Criteria

- Stage ledger is complete and internally consistent.
- All stage reports exist or have explicit blocked/superseded explanation.
- Prompt pack contains no stale paths, ambiguous acceptance, or missing stage ledger update rules.
- Final report contains go/no-go for a separate mainnet plan.

## Quality Gates

- `python -m tools.docs.generate_docs_index --check`
- Focused lint/type/tests only if code or executable scripts changed.
- CI/deploy/runtime smoke evidence if shipped.

## Final Output

Russian final closure report: accepted stages, unresolved blockers, evidence index, contract impact, delivery status, and next-plan handoff for mainnet real-money trading.
