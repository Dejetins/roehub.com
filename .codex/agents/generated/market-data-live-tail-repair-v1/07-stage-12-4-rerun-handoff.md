---
prompt_name: 07-stage-12-4-rerun-handoff
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
scope: "Rerun or explicitly reopen Strategy Producer Stage 12.4 after accepted live-tail repair proof."
language:
  implementation: shell/markdown
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
      why: "repair-cycle accepted proof"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "Stage 12.4 current state and 12.5 gate"
    - path: .codex/agents/generated/strategy-producer-paper-testnet-trading-v1/12-4-sustained-6h-soak.md
      why: "canonical 12.4 rerun prompt"
  task_entrypoints:
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
      why: "existing blocked report to supersede or append rerun evidence"
skill_routing:
  - skill: backend-performance-evidence
    use_when: "summarizing sustained soak latency/resource evidence"
    timing: "during verification"
    reason: "12.4 acceptance depends on comparable latency and resource measurement"
  - skill: browser-qa-evidence
    use_when: "capturing final /strategies state"
    timing: "during verification"
    reason: "browser-visible strategy status must be proven"
  - skill: publish-ci-deploy
    use_when: "accepted report/ledger changes need delivery"
    timing: "before final report"
    reason: "direct-main delivery required"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["6h-runtime", "signal-latency", "signal-dedup", "repair-metrics", "redis", "postgres", "prometheus", "browser"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
proof_boundary:
  required_when: "Stage 07 observes deployed runtime after accepted repair proof"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
stage_execution_ledger:
  path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  plan_doc: docs/architecture/market_data/market-data-live-tail-repair-v1.md
  current_stage: "07"
  required_update: true
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/07-stage-12-4-rerun-handoff.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "Do not open Strategy Producer Stage 12.5 unless Stage 12.4 is accepted."
  - "Do not reinterpret Stage 06 repair proof as a 6h soak."
---

# Task

Execute Stage `07` by rerunning or explicitly reopening Strategy Producer Stage `12.4` after accepted Market Data live-tail repair proof.

## Requirements (Must)

- Verify repair ledger marks Stage `06 accepted`.
- Verify strategy-producer ledger still marks `12.4 blocked` and `12.5 pending/blocked`.
- Use the existing Strategy Producer prompt `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/12-4-sustained-6h-soak.md` as the canonical `12.4` acceptance contract.
- Before starting any 6h timer, prove repair metrics/audit surfaces from Stage `06` are still available.
- Run `12.4` according to its current prompt:
  - active strategy runtime;
  - 6 elapsed hours;
  - signal-path latency/dedup;
  - Redis/DB/Prometheus/Monit/resource snapshots;
  - final browser/API state.
- If a short candle gap appears during the soak, record whether the new repair path recovered it and include audit/metrics evidence.
- If `12.4` passes, update both ledgers: repair Stage `07 accepted` and strategy-producer `12.4 accepted`; only then `12.5` may open.
- If `12.4` blocks again, classify whether blocker is the same live-tail repair issue or a new unrelated issue; update both ledgers.
- Deliver report/ledger docs through direct-main discipline if files changed.

## Non-Goals

- Do not add new repair implementation in this stage unless the rerun finds a narrow blocker and the report marks Stage `07 blocked`.
- Do not start Stage `12.5` in the same execution.

## Acceptance Criteria

- `12.4` accepted rerun has full 6h evidence and repair path remains healthy; or Stage `07` records a clear blocker with source and next action.
- If accepted, strategy-producer ledger allows `12.5`.
- Repair ledger records whether this plan closed the original problem.

## Quality Gates

- `uv run python -m tools.docs.generate_docs_index --check`
- Runtime collector evidence required by `12-4-sustained-6h-soak.md`
- Browser/API proof required by `12-4-sustained-6h-soak.md`

## Final Output

Russian report with `12.4` rerun result, repair-path observations, signal latency/dedup evidence, ledger updates, delivery status, and explicit statement whether `12.5` is allowed.
