---
prompt_name: 05-metrics-alerts-runbook
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
scope: "Add metrics, alerts, and runbook coverage for Market Data live-tail repair."
language:
  implementation: python/yaml/markdown
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1.md
      why: "source plan"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
      why: "stage gate"
  task_entrypoints:
    - path: apps/worker/market_data_ws/wiring/modules/market_data_ws.py
      why: "Market Data metrics wiring"
    - path: apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      why: "Strategy runner metrics wiring"
    - path: infra
      why: "Prometheus/Monit/alert assets if present"
skill_routing:
  - skill: backend-quality-gates
    use_when: "running focused checks"
    timing: "after implementation"
    reason: "Stage 05 touches metrics wiring and tests"
  - skill: publish-ci-deploy
    use_when: "accepted changes need delivery"
    timing: "before final report"
    reason: "direct-main delivery required"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces: ["metrics-endpoint", "prometheus-rules", "runbook", "docs-index"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/05-metrics-alerts-runbook.md
proof_boundary:
  required_when: "Production runtime proof deferred to Stage 06"
  label: none
  changed_code_production_claim_allowed: false
stage_execution_ledger:
  path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  plan_doc: docs/architecture/market_data/market-data-live-tail-repair-v1.md
  current_stage: "05"
  required_update: true
expected_primary_touches:
  - apps/worker/market_data_ws/wiring/modules/market_data_ws.py
  - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
  - infra
  - configs/prod
  - docs/architecture/market_data
  - tests
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/05-metrics-alerts-runbook.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "Prometheus labels must stay bounded; do not add user_id, run_id, order_id, raw provider error, or secret-bearing labels."
---

# Task

Implement Stage `05` metrics, alerts, and runbook.

## Requirements (Must)

- Verify Stage `04 accepted` in the repair ledger.
- Add or wire metrics for:
  - gap count;
  - repair source/status;
  - repair latency;
  - Redis hot cache hit/miss/write/error;
  - ClickHouse repair circuit state;
  - checkpoint stall;
  - deferred ACK or backlog retry.
- Add alert rules or documented Monit/Prometheus alert config for:
  - unrepaired gap beyond policy;
  - ClickHouse circuit open too long;
  - REST tail repair errors/rate-limit;
  - hot cache short-tail miss;
  - active run without `StrategySignal` growth.
- Add a Russian operator runbook with exact safe checks and no secret output.
- Prove metrics through a synthetic repair/cache call against the local metrics endpoint or direct registry scrape.
- Update report and ledger.
- If accepted and files changed, deliver through `publish-ci-deploy`.

## Non-Goals

- Do not run Mac Studio production proof in this stage.
- Do not change repair behavior except narrow metric wiring needed for visibility.

## Acceptance Criteria

- Metrics endpoint or registry output contains the required metric names after synthetic calls.
- Alert rules parse or pass the repository's existing alert validation method.
- Runbook explains what to check in Redis, Postgres audit, ClickHouse circuit, REST tail, and strategy checkpoint without printing secrets.
- Ledger marks `05 accepted`.

## Quality Gates

- `uv run ruff check apps src tests`
- `uv run pyright apps src tests`
- `uv run pytest -q tests/unit tests/integration`
- `uv run python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with metric names, alert evidence, runbook path, synthetic metrics proof, delivery status, and Stage `06` handoff.
