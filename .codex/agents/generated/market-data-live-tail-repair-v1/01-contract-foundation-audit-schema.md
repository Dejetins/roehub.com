---
prompt_name: 01-contract-foundation-audit-schema
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
scope: "Implement the Market Data live-tail repair contract foundation: ClosedCandleTailProvider port, result DTOs, config primitives, and durable repair audit schema."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, branch policy, Mac Studio and redaction rules"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1.md
      why: "source plan"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
      why: "repair-cycle stage ledger"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
      why: "runtime blocker evidence"
  task_entrypoints:
    - path: src/trading/contexts/strategy/application/services/live_runner.py
      why: "current ClickHouse-only repair consumer"
      inspect_symbols: ["StrategyLiveRunner", "_repair_gap"]
    - path: src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
      why: "existing canonical candle reader contract"
    - path: src/trading/contexts/market_data/application/dto
      why: "DTO location for Market Data application objects"
    - path: alembic/versions
      why: "Postgres migration pattern"
skill_routing:
  - skill: backend-quality-gates
    use_when: "running Python lint, type, and focused tests"
    timing: "after implementation"
    reason: "Stage 01 changes Python contracts and migration code"
  - skill: publish-ci-deploy
    use_when: "stage is accepted and changes must be delivered"
    timing: "before final report"
    reason: "accepted Roehub stages require direct-main delivery"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces: ["postgres-migration", "application-port-contract", "docs-index"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/01-contract-foundation-audit-schema.md
proof_boundary:
  required_when: "Mac Studio or production runtime proof is not required in Stage 01"
  label: none
  changed_code_production_claim_allowed: false
stage_execution_ledger:
  path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  plan_doc: docs/architecture/market_data/market-data-live-tail-repair-v1.md
  current_stage: "01"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - src/trading/contexts/strategy/application/ports/closed_candle_tail_provider.py
  - src/trading/contexts/market_data/application/dto
  - src/trading/contexts/market_data/application/ports
  - src/trading/contexts/market_data/adapters/outbound/persistence/postgres
  - alembic/versions
  - tests/unit/contexts/market_data
  - tests/unit/contexts/strategy
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/01-contract-foundation-audit-schema.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
possible_secondary_touches:
  - src/trading/contexts/market_data/application/__init__.py
  - src/trading/contexts/strategy/application/ports/__init__.py
  - docs/architecture/market_data/market-data-live-tail-repair-v1.md
  - docs/architecture/README.md
safety_notes:
  - "Do not modify StrategyLiveRunner behavior in Stage 01."
  - "Do not call Binance/Bybit REST or ClickHouse in Stage 01 acceptance; this stage defines contracts and audit persistence only."
---

# Task

Implement Stage `01` for `market-data-live-tail-repair-v1`.

Done means:

- `ClosedCandleTailProvider` exists as a Strategy application port.
- Market Data repair result DTOs and audit event persistence contract exist.
- Postgres additive migration for repair audit exists and is locally validated.
- Stage report and ledger record accepted or blocked status.

## Requirements (Must)

- State `User required before start: nothing` unless you discover a real missing prerequisite; do not ask for secrets in chat.
- Verify this is Stage `01` in the repair ledger before implementation.
- Add a Strategy-side port but keep the implementation owned by Market Data adapters/use cases in later stages.
- Add DTOs that can represent:
  - continuous vs missing range;
  - source per candle: `redis_hot_cache`, `clickhouse`, `rest`;
  - sources attempted;
  - restored and missing `ts_open` values;
  - stable redacted error codes.
- Add additive Postgres audit schema for `market_data_candle_repair_events` or an explicitly equivalent table.
- Add repository/port tests that prove an audit event can be inserted and read without ClickHouse.
- Add fake provider contract tests that prove continuous and missing results are represented deterministically.
- Update `market-data-live-tail-repair-v1.md` only if implementation finds a necessary contract correction.
- Update stage report and ledger after validation.
- If accepted and files changed, deliver through `publish-ci-deploy` direct-main discipline.

## Non-Goals

- Do not implement Redis hot cache.
- Do not implement REST fallback.
- Do not change `_repair_gap`.
- Do not run Stage `12.4`.

## Acceptance Criteria

- Focused tests pass for DTO/port/audit repository.
- Migration upgrade/downgrade or project migration check passes according to repo pattern.
- A direct local DB/repository call proves audit insert/read.
- No runtime behavior changed yet.
- Ledger marks `01 accepted`; Stage `02` remains closed until delivery evidence is recorded.

## Quality Gates

- `uv run ruff check src/trading/contexts/market_data src/trading/contexts/strategy tests/unit/contexts/market_data tests/unit/contexts/strategy`
- `uv run pyright src/trading/contexts/market_data src/trading/contexts/strategy tests/unit/contexts/market_data tests/unit/contexts/strategy`
- `uv run pytest -q tests/unit/contexts/market_data tests/unit/contexts/strategy`
- `uv run python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with file manifest, migration name, focused gate output, audit DB proof, ledger status, delivery status, and next-stage handoff.
