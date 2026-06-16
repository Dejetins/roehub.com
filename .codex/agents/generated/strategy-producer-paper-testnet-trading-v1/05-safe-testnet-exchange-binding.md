---
prompt_name: 05-safe-testnet-exchange-binding
repo: roehub.com
branch: main
scope: "Bind strategies to owned testnet exchange connections and verify safe isolated futures 1x without auto-config."
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
    - path: src/trading/contexts/exchange_control
      why: "exchange connection, validation, account-state, custody"
    - path: src/trading/contexts/live_execution/domain/account_state.py
      why: "account projection and risk input"
    - path: apps/api/routes/strategies.py
      why: "strategy binding/profile readiness"
    - path: apps/exchange_execution
      why: "testnet adapter guard integration"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "current exchange-connection /settings and readiness contract"
skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing profile/exchange readiness DTOs or persistence"
    timing: during implementation
    reason: "exchange binding is a public and persistence contract"
  - skill: root-cause-debugging
    use_when: "testnet account reads fail or mismatch unexpectedly"
    timing: if blocker
    reason: "separate credential/config/environment failures from code defects"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth and scoped staging; branches/PRs are temporary and must be delivered to main and cleaned up before acceptance"
  - skill: publish-ci-deploy
    use_when: "accepted changes need shipping"
    timing: before ship
    reason: "record CI/deploy/runtime handoff"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["exchange-testnet-read", "api", "database", "metrics"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/05-safe-testnet-exchange-binding.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "05"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - src/trading/contexts/exchange_control
  - src/trading/contexts/live_execution
  - apps/api/routes/strategies.py
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/05-safe-testnet-exchange-binding.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - apps/exchange_execution
  - tests/unit/contexts/exchange_control
  - tests/unit/contexts/live_execution
  - docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
  - docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
  - docs/architecture/README.md
safety_notes:
  - "Do not auto-configure leverage, margin mode, or position mode on exchange accounts."
  - "User provides testnet keys via /settings; never ask for or store secrets in repo/docs."
---

# Task

Implement and prove safe testnet exchange binding for strategy launch. Futures shorts are allowed only when read-only account/config evidence proves isolated margin and leverage `1x`; otherwise fail closed.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Before implementation, verify Stage `04` is `accepted` in the stage ledger; stop if it is blocked or pending unless this task is explicitly converted into an unblock/repair task.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `github:yeet`/`publish-ci-deploy` discipline for scoped publish, but do not leave the stage on a per-stage branch or draft PR. Temporary branches are allowed only when useful; before marking the stage `accepted`, deliver the changes to `main`, push `origin main`, verify main contains the changes, delete any temporary local/remote branch, and record main-branch delivery plus branch-cleanup evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.
- Support owned Binance and Bybit testnet connections already added through `/settings`.
- Verify account/config state with real testnet read calls where credentials exist.
- Do not call exchange APIs that mutate leverage, margin mode, or position mode.
- Record stable blocked reasons for missing key, inactive connection, readonly mismatch, stale projection, config mismatch, missing balance, min-notional issue, or unsafe futures short.
- Preserve credential custody: only `exchange-execution` or exchange-control scope may resolve secrets as already designed.
- If `/settings`, exchange connection readiness, trading capability, validation, or strategy-binding semantics change, update the identity exchange-connections plan/ledger docs in the same stage; if unchanged, state that explicitly in the stage report.

## Acceptance Criteria

- Real Binance/Bybit testnet account/config read evidence is recorded for available keys, or exact missing-key blocker is recorded.
- SQL/API readiness shows safe/blocked states without secrets.
- A futures short `1x isolated` scenario is accepted only when verified; mismatch is blocked.
- Metrics/audit include bounded labels/reasons.

## Quality Gates

- `uv run ruff check src/trading/contexts/exchange_control src/trading/contexts/live_execution apps tests`
- `uv run pyright src/trading/contexts/exchange_control src/trading/contexts/live_execution apps tests`
- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/contexts/live_execution tests/unit/apps`
- `python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with testnet exchange-read evidence, fail-closed proof, contract impact, blockers, delivery status, and next-stage handoff.
