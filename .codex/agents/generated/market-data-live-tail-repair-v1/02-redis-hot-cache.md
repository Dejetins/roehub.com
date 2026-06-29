---
prompt_name: 02-redis-hot-cache
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
scope: "Implement Redis hot cache for closed 1m candles in Market Data and prove range reads on real Redis."
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
      why: "previous stage gate and handoff"
    - path: docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md
      why: "existing Redis stream publisher contract"
  task_entrypoints:
    - path: src/trading/contexts/market_data/adapters/outbound/messaging/redis/redis_streams_live_candle_publisher.py
      why: "current WS closed candle Redis publisher"
    - path: apps/worker/market_data_ws/wiring/modules/market_data_ws.py
      why: "market-data worker wiring and metrics"
    - path: configs/prod/market_data.yaml
      why: "production Market Data runtime config"
skill_routing:
  - skill: backend-quality-gates
    use_when: "running focused Python checks"
    timing: "after implementation"
    reason: "Stage 02 changes Redis adapters and config"
  - skill: publish-ci-deploy
    use_when: "accepted changes need delivery"
    timing: "before final report"
    reason: "direct-main delivery required"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces: ["redis", "metrics", "docs-index"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/02-redis-hot-cache.md
proof_boundary:
  required_when: "Redis integration can be local; production proof deferred to Stage 06"
  label: none
  changed_code_production_claim_allowed: false
stage_execution_ledger:
  path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  plan_doc: docs/architecture/market_data/market-data-live-tail-repair-v1.md
  current_stage: "02"
  required_update: true
expected_primary_touches:
  - src/trading/contexts/market_data/adapters/outbound/messaging/redis
  - src/trading/contexts/market_data/adapters/outbound/config
  - apps/worker/market_data_ws/wiring/modules/market_data_ws.py
  - configs/prod/market_data.yaml
  - tests/unit/contexts/market_data
  - tests/integration
  - docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/02-redis-hot-cache.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/market_data/market-data-live-tail-repair-v1.md
  - docs/architecture/README.md
safety_notes:
  - "Redis hot cache is a range-store, not a replacement for Redis Streams."
  - "Cache write failure must not stop WS ingestion, but must emit metrics."
---

# Task

Implement Stage `02` Redis hot cache.

## Requirements (Must)

- Verify Stage `01 accepted` in the repair ledger; stop if it is not accepted.
- Implement Redis hot cache keys:
  - `md:hot:1m:<instrument_key>:z`
  - `md:hot:1m:<instrument_key>:h`
- Store normalized candle JSON by `ts_open_epoch_ms` and zset score by the same timestamp.
- Add bounded retention configuration, defaulting to production-safe `24h` unless implementation evidence requires lower default.
- Write every WS closed 1m candle to hot cache from Market Data before or alongside stream publish.
- Duplicate writes must be deterministic and not create ambiguous rows.
- Implement range read `[start,end)` returning strictly sorted candles.
- Expose hit/miss/write/error/duration metrics.
- Update `market-data-live-feed-redis-streams-v1.md` to distinguish stream transport from hot cache range-store.
- Update stage report and ledger after validation.
- If accepted and files changed, deliver through `publish-ci-deploy`.

## Non-Goals

- Do not implement REST tail fallback.
- Do not integrate `StrategyLiveRunner` with hot cache yet.
- Do not change Stage `12.4`.

## Acceptance Criteria

- Unit tests prove serialization, duplicate write, sorted range read, and retention behavior.
- A real Redis call writes at least three synthetic closed candles, rewrites one duplicate, reads a `[start,end)` range in order, and proves no ambiguity.
- Metrics are emitted or directly observable through the worker metric hooks/tests.
- Ledger marks `02 accepted`.

## Quality Gates

- `uv run ruff check src/trading/contexts/market_data apps/worker/market_data_ws tests`
- `uv run pyright src/trading/contexts/market_data apps/worker/market_data_ws tests`
- `uv run pytest -q tests/unit/contexts/market_data tests/integration`
- `uv run python -m tools.docs.generate_docs_index --check`

## Final Output

Russian report with file manifest, Redis key examples, real Redis command/call evidence, metrics evidence, delivery status, and Stage `03` handoff.
