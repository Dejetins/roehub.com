---
prompt_name: 03-tail-provider-source-chain
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
scope: "Implement the ClosedCandleTailProvider source chain: Redis hot cache, ClickHouse circuit-bounded read, REST tail fallback, and repair audit."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1.md
      why: "source plan"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
      why: "stage gate"
    - path: docs/architecture/market_data/market-data-rest-historical-catchup-1m-v2.md
      why: "existing REST catch-up/fill semantics"
  task_entrypoints:
    - path: src/trading/contexts/market_data/application/use_cases/rest_fill_range_1m.py
      why: "existing REST source/writer use case"
    - path: src/trading/contexts/market_data/application/ports/sources/candle_ingest_source.py
      why: "REST candle source port"
    - path: src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py
      why: "ClickHouse canonical reader"
    - path: src/trading/contexts/market_data/adapters/outbound/messaging/redis
      why: "hot cache adapter from Stage 02"
skill_routing:
  - skill: backend-quality-gates
    use_when: "running focused Python checks"
    timing: "after implementation"
    reason: "provider chain touches adapters and ports"
  - skill: root-cause-debugging
    use_when: "provider integration does not reproduce the ClickHouse-failure fallback"
    timing: "if blocker"
    reason: "must prove the original failure mode is addressed"
  - skill: publish-ci-deploy
    use_when: "accepted changes need delivery"
    timing: "before final report"
    reason: "direct-main delivery required"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces: ["provider-chain", "redis", "postgres-audit", "rest-adapter-boundary"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/03-tail-provider-source-chain.md
proof_boundary:
  required_when: "Production runtime proof deferred to Stage 06"
  label: none
  changed_code_production_claim_allowed: false
stage_execution_ledger:
  path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  plan_doc: docs/architecture/market_data/market-data-live-tail-repair-v1.md
  current_stage: "03"
  required_update: true
expected_primary_touches:
  - src/trading/contexts/market_data/application
  - src/trading/contexts/market_data/adapters/outbound
  - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
  - configs/prod/strategy.yaml
  - configs/prod/market_data.yaml
  - tests/unit/contexts/market_data
  - tests/integration
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/03-tail-provider-source-chain.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/market_data/market-data-live-tail-repair-v1.md
  - docs/architecture/README.md
safety_notes:
  - "REST tail may fetch only strictly closed candles inside configured short-tail window."
  - "Do not log raw provider payloads or credentials."
---

# Task

Implement Stage `03` provider source chain.

## Requirements (Must)

- Verify Stage `02 accepted` in the repair ledger.
- Implement a concrete `ClosedCandleTailProvider` adapter owned by Market Data.
- Source order must be Redis hot cache -> ClickHouse with short timeout/circuit breaker -> REST tail -> audit miss/failure.
- REST tail must use existing Market Data REST source boundary; Strategy must not call Binance/Bybit REST directly.
- REST tail must reject current open minute and ranges older than configured live-tail limit.
- Repaired REST candles must be written to Redis hot cache before the provider returns success.
- Every provider attempt must write redacted repair audit rows.
- ClickHouse failure must not fail the provider before Redis/REST fallback is attempted.
- Update docs if implementation chooses exact timeout/circuit config names.
- Update report and ledger after validation.
- If accepted and files changed, deliver through `publish-ci-deploy`.

## Non-Goals

- Do not modify Strategy runner processing/ACK behavior in this stage.
- Do not perform live exchange/testnet trading.
- Do not run the 6h soak.

## Acceptance Criteria

- A direct provider integration call with:
  - Redis cache miss for one minute;
  - ClickHouse reader forced to fail or circuit-open;
  - fake or safe REST source returning the missing closed candle;
  proves `continuous=true`, sorted rows, hot cache write, and audit event.
- A second provider call proves Redis hot cache hit without calling REST again.
- Missing REST tail or non-closed candle returns `continuous=false` and writes audit miss without checkpoint side effects.
- No raw provider payload or secret appears in logs/report.
- Ledger marks `03 accepted`.

## Quality Gates

- `uv run ruff check src/trading/contexts/market_data apps/worker/strategy_live_runner tests`
- `uv run pyright src/trading/contexts/market_data apps/worker/strategy_live_runner tests`
- `uv run pytest -q tests/unit/contexts/market_data tests/integration`
- `uv run python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with provider source-order proof, ClickHouse-failure fallback proof, audit DB proof, hot-cache proof, redaction statement, delivery status, and Stage `04` handoff.
