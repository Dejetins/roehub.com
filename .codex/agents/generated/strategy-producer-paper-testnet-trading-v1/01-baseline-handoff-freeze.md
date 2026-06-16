---
prompt_name: 01-baseline-handoff-freeze
repo: roehub.com
branch: main
scope: "Freeze the current live-execution foundation and Mac Studio runtime baseline for the new paper/testnet strategy producer cycle."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repository engineering contract and Mac Studio path rules"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
      why: "source-of-truth architecture plan"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "stage execution handoff"
  task_entrypoints:
    - path: docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md
      why: "accepted foundation state through Stage 17"
    - path: apps/api/routes/ui_backtests.py
      why: "current backtest UI API surface"
    - path: apps/api/routes/ui_strategies_dashboard.py
      why: "current strategies dashboard API surface"
    - path: infra/scripts/monit
      why: "current Mac Studio service supervision assets"
skill_routing:
  - skill: architecture-review
    use_when: "checking docs/code/runtime drift before accepting the baseline"
    timing: before implementation
    reason: "baseline is a review and evidence task"
  - skill: browser-qa-evidence
    use_when: "capturing browser-visible /settings, /backtests, or /strategies state"
    timing: during verification
    reason: "browser-visible baseline must be observed, not inferred"
  - skill: github:yeet
    use_when: "accepted changes need GitHub publish after validation"
    timing: before ship
    reason: "explicit user-required publish flow; verifies gh auth and scoped staging; branches/PRs are temporary and must be delivered to main and cleaned up before acceptance"
  - skill: publish-ci-deploy
    use_when: "publishing accepted docs or drift repairs"
    timing: before ship
    reason: "Roehub delivery must include CI/deploy/runtime handoff when changes are shipped"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["macstudio-ssh", "api", "database", "redis", "monit", "prometheus", "browser"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/01-baseline-handoff-freeze.md
stage_execution_ledger:
  path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  current_stage: "01"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/01-baseline-handoff-freeze.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
  - docs/architecture/README.md
safety_notes:
  - "Do not write credentials, cookies, tokens, raw exchange responses, or signed payloads into reports."
  - "Mainnet submit remains out of scope and must stay blocked."
---

# Task

Create Stage 01 baseline evidence for the new `strategy-producer-paper-testnet-trading-v1` plan. Do not implement feature code unless you find a blocking docs/runtime drift that must be corrected before later stages.

Done means:

- the current accepted Stage 17 foundation is reconciled with the new plan;
- Mac Studio runtime inventory covers API, Postgres, Redis, Monit, Prometheus, `/settings`, `/backtests`, and `/strategies`;
- a Russian stage report exists and the ledger is updated;
- any accepted changes are delivered only after evidence is recorded.

## Requirements (Must)

- Before implementation, explicitly state `User required before start: ...`; if nothing is required, state `User required before start: nothing`. If user-provided keys, artifacts, or access are needed, stop before implementation and list the exact requirement; do not ask for secrets in chat. Record this pre-start line in the stage report.
- Verify git state locally and, if using Mac Studio, use `git -C /Users/daniildegtyarev/Projects/roehub.com` for repository commands.
- Collect real evidence with concrete commands/calls: authenticated API where needed, SQL inventory, Redis `XINFO`/stream checks, Monit summary, Prometheus target/metric probes, and browser screenshots.
- Do not publish/deploy if acceptance is blocked. If accepted and files changed, use `github:yeet`/`publish-ci-deploy` discipline for scoped publish, but do not leave the stage on a per-stage branch or draft PR. Temporary branches are allowed only when useful; before marking the stage `accepted`, deliver the changes to `main`, push `origin main`, verify main contains the changes, delete any temporary local/remote branch, and record main-branch delivery plus branch-cleanup evidence in the stage report and ledger. For runtime/code stages, also record Mac Studio host sync/deploy smoke.
- Explicitly mark stale docs/code/runtime drift as `drift`, not as implemented fact.
- Confirm no mainnet submit path is enabled for this cycle.
- Update `01-baseline-handoff-freeze.md` and the stage ledger before final report.
- The stage report must include a file manifest table: `Created / Modified / Deleted / Reason / Contract impact`; justify any touched file outside expected paths.
- Before editing, narrow any broad expected directory path to a concrete file list or planned new files and record that list in the stage report.

## Work Plan

1. Read the plan, ledger, and accepted gateway Stage 17 handoff.
2. Inventory current local and Mac Studio runtime surfaces relevant to this plan.
3. Inspect current browser-visible flows for `/settings`, `/backtests`, and `/strategies`.
4. Record facts, blockers, and next-stage handoff in the stage report.
5. Run docs index verification.
6. If changes were made and accepted, publish/deliver to `main`, clean up any temporary branch/PR, and sync Mac Studio as required; otherwise report no publish.

## Acceptance Criteria

- Stage report contains command-level evidence for API, DB, Redis, Monit, Prometheus, and browser runtime.
- Ledger row `01` is `accepted` or `blocked` with specific blocker and next action.
- Tests-only acceptance is not used.

## Quality Gates

- `python -m tools.docs.generate_docs_index --check`
- Runtime commands/calls listed in the stage report must have concrete outputs summarized.

## Final Output

Report in Russian: status, changed files, evidence summary, blockers, contract impact, delivery/deploy status, next-stage handoff.
