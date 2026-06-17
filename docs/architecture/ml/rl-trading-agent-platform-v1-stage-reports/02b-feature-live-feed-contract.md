---
doc: rl-trading-agent-platform-v1-stage-02b-feature-live-feed-contract
stage: "02B"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-18"
---

# Stage 02B: Feature And Live-Feed Contract

Статус: `accepted`.

Stage `02B` freezes the article-compatible feature/live-feed contract for the RL Trading Agent Platform v1. It does not train models, create datasets, add cloud/model hosting, open paper/testnet/live execution, or change exchange submission/secrets custody.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

## Scope

Included:

- verify prerequisite Stage `02A`;
- define the `binance:futures` v1 feature order, dtype, normalization inputs, missing-field policy, `vwap` derivation and deterministic feature-contract hash;
- record the v1 training-source matrix with `binance:futures` as the only training branch and Binance spot, Bybit spot and Bybit futures as `blocked_not_training_source_v1`;
- define the Binance Futures metadata gate for funding, mark/index, filters, leverage tiers, fee/slippage/liquidation assumptions;
- make the Redis live-feed `trades_count` decision explicit without adding full ClickHouse scans to the live hot path;
- record prompt path/hash, file manifest, evidence, contract impact, delivery state and next-stage handoff.

Not included:

- training user-owned custom models or adding PyTorch runtime dependencies;
- importing or materializing full datasets/checkpoints in git or `/opt/roehub/state/rl_trading/`;
- Bybit `trades_count` enrich, feature-mask training branches, or research-only Bybit training;
- changing public API, persisted schemas, browser-visible behavior, exchange execution, or secrets handling;
- mainnet, testnet or paper exchange side effects.

## File Manifest

Planned concrete file list before edits:

- `src/trading/contexts/rl_trading/__init__.py`
- `src/trading/contexts/rl_trading/domain/__init__.py`
- `src/trading/contexts/rl_trading/domain/feature_contract.py`
- `tests/unit/contexts/rl_trading/domain/test_feature_contract.py`
- `src/trading/contexts/market_data/adapters/outbound/messaging/redis/redis_streams_live_candle_publisher.py`
- `tests/unit/contexts/market_data/adapters/test_redis_streams_live_candle_publisher.py`
- `src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_streams_live_candle_stream.py`
- `tests/unit/contexts/strategy/adapters/test_redis_strategy_live_candle_stream.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02b-feature-live-feed-contract.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if `python -m tools.docs.generate_docs_index --check` requires index regeneration

Outside expected paths: `src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_streams_live_candle_stream.py` and its focused test are outside the prompt's primary/secondary touch list but are an explicit task entrypoint and are required to prove Redis live-feed consumer parity for `trades_count`. `docs/architecture/README.md` is justified only if the docs index generator updates it after adding this report.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/02b-feature-live-feed-contract.md` |
| Prompt sha256 | `c61f070e31631ff60edbf82b3c08ea57ddb4a12de293a5e606eb3daab9a1e70b` |
| Ledger state before implementation | Stage `02A` accepted; `current_stage=02B`; Stage `02B` pending |
| Required prerequisite | Stage `02A` accepted |
| Delivery state | `local-only`; no branch, PR, main delivery, deploy, runtime service, schema, API, UI, exchange, or ML artifact change |
| Large artifacts | No datasets, checkpoints, raw provider payloads, or runtime ML artifacts were written to git or `/opt/roehub/state/rl_trading/`. |

## Observed State

| Area | Evidence summary |
|---|---|
| Stage prerequisite | Ledger records Stage `02A` as `accepted`; Stage `02B` is the current stage. |
| Feature source | Stage `02A` records article-compatible channel order as `open`, `high`, `volume_weighted_average`, `low`, `close`, `volume`, `num_trades`. |
| Training source | Stage `02A` and the plan restrict v1 training to `binance:futures`; Binance spot, Bybit spot and Bybit futures must remain non-training branches. |
| Candle metadata | `CandleMeta` already contains nullable `trades_count`; Binance WS/REST can populate it, Bybit currently leaves it `None`. |
| Redis live feed | Current publisher emits OHLCV and `volume_quote` but not `trades_count`; current Redis consumer reconstructs `CandleMeta` with `trades_count=None`. |
| Futures metadata | Current `market_data` inventory lacks funding, mark/index price, leverage-tier and point-in-time filter/lifecycle tables; `ref_instruments` has current filters only. |

Implemented file manifest:

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/__init__.py`; `src/trading/contexts/rl_trading/domain/__init__.py`; `src/trading/contexts/rl_trading/domain/feature_contract.py`; `tests/unit/contexts/rl_trading/domain/test_feature_contract.py`; this report | - | - | First bounded `rl_trading` feature-contract surface and deterministic tests. | `compatible-change` new internal/domain contract and docs/report |
| - | `src/trading/contexts/market_data/adapters/outbound/messaging/redis/redis_streams_live_candle_publisher.py`; `tests/unit/contexts/market_data/adapters/test_redis_streams_live_candle_publisher.py` | - | Publish `CandleMeta.trades_count` as an additive Redis live-feed field and test null/string behavior. | `compatible-change` additive Redis wire field |
| - | `src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_streams_live_candle_stream.py`; `tests/unit/contexts/strategy/adapters/test_redis_strategy_live_candle_stream.py` | - | Preserve additive `trades_count` when present and keep old schema-v1 payloads backward compatible. | `compatible-change` additive Redis consumer behavior |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark Stage `02B` accepted, record evidence and open Stage `02C`. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Docs index update after adding Stage `02B` report. | `compatible-change` docs index only |

Outside expected paths:

- `src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_streams_live_candle_stream.py` and `tests/unit/contexts/strategy/adapters/test_redis_strategy_live_candle_stream.py`: explicit task entrypoint needed for Redis live-feed consumer parity.
- `tests/unit/contexts/market_data/adapters/test_redis_streams_live_candle_publisher.py`: focused regression for the allowed market-data secondary code touch.
- `docs/architecture/README.md`: generated architecture docs index.

## Feature Contract

| Field | Value |
|---|---|
| Contract id | `rl_trading.article_compatible.binance_futures` |
| Schema version | `1` |
| Feature dtype | `float32` for materialized slabs/model inputs; HF import may read original arrays before conversion. |
| Feature order | `open`, `high`, `volume_weighted_average`, `low`, `close`, `volume`, `num_trades` |
| Feature contract hash | `d2e99786b68482d730494c6aeec72a1e9f40ac225729019fac5c82f96f900be9` |
| Normalization inputs | `z_score` using train-split-only per `exchange + market_type + symbol` stats: `mean`, `std`, `min_std=1e-12`. Live inference must use an accepted training-stats manifest and must not refit on the hot path. |
| Missing-field behavior | Fail closed. No feature-mask training branch is opened in Stage `02B`. |
| `vwap` derivation | `volume_quote / volume_base` when `volume_base > 0`; when both volumes are zero, use `close`; missing `volume_quote` or positive quote with zero base volume fails closed. |
| `num_trades` source | `canonical_candles_1m.trades_count` / Redis `trades_count`; missing value fails closed for RL vector construction. |
| Live-feed hot path | Full ClickHouse scan is forbidden. Redis must carry `volume_quote` and `trades_count`; repair is gap/degraded path only. |

Implementation:

- `RlFeatureCandle` and `build_article_feature_vector_v1` produce the frozen article-compatible order.
- `FeatureContractViolation.reason` records deterministic fail-closed reasons such as `missing_volume_quote`, `missing_trades_count`, and `inconsistent_zero_base_positive_quote_volume`.
- `feature_contract_canonical_payload_v1()` and `feature_contract_hash_v1()` provide a deterministic hash for Stage `05` dataset builder and Stage `13` train/live parity.

## Training-Source Matrix

| Exchange | Market type | Status | Decision |
|---|---|---|---|
| `binance` | `futures` | `trainable` | Only v1 training branch. Article-compatible candle fields are available; futures metadata gate still blocks production-grade evaluation/activation until resolved. |
| `binance` | `spot` | `blocked_not_training_source_v1` | Product/execution inventory only for this cycle. |
| `bybit` | `spot` | `blocked_not_training_source_v1` | No Bybit `trades_count` enrich, feature-mask branch, or research-only Bybit training in this stage. |
| `bybit` | `futures` | `blocked_not_training_source_v1` | No Bybit `trades_count` enrich, feature-mask branch, or research-only Bybit training in this stage. |

## Binance Futures Metadata Gate

Gate id: `binance_futures_metadata_gate_v1`.

Activation behavior: `fail_closed_for_production_grade_futures_evaluation_until_resolved`.

| Requirement | Status | Gate behavior |
|---|---|---|
| Funding-rate history | `missing_required_source` | Blocks production-grade futures backtest/evaluation until sourced or explicitly approximated. |
| Mark-price history | `missing_required_source` | Blocks liquidation/mark-to-market claims until sourced or explicitly approximated. |
| Index-price history | `missing_required_source` | Blocks basis/mark-index assumptions until sourced or explicitly approximated. |
| Point-in-time filters | `available_current_snapshot_only` | Current `ref_instruments` filters are not enough for historical survivorship-bias proof. |
| Leverage tiers | `missing_required_source` | Blocks leverage/liquidation-sensitive evaluation until sourced or explicitly approximated. |
| Fee policy | `assumption_required` | Stage `08` scorecard must declare maker/taker fee assumptions before candidate acceptance. |
| Slippage policy | `assumption_required` | Stage `08` scorecard must declare slippage assumptions before candidate acceptance. |
| Liquidation policy | `assumption_required` | Stage `08` scorecard must declare liquidation assumptions before candidate acceptance. |

This metadata gate does not block Stage `02C`; it blocks future production-grade futures evaluation/activation claims until later stages source the data or record an accepted approximation.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API routes or response payloads changed. |
| Port contract | `none` | No Python port/protocol signature changed. |
| DTO schema | `compatible-change` | Redis live candle wire payload gains additive optional `trades_count`; old schema-v1 messages still parse with `trades_count=None`. New RL feature dataclasses are internal/additive. |
| Persisted schema | `none` | No migration or storage schema changed. |
| Config schema/defaults | `none` | No env/YAML/default changed. |
| Request hash / cache key / persistence identity | `none` | Existing request/cache/persistence identity is unchanged. |
| Service-call auth/timeout/retry/error semantics | `none` | No service call behavior changed. |
| External side effects / idempotency / unknown-state semantics | `none` | No exchange, paper, testnet, mainnet, provider or durable side effect was added. |
| Logs/metrics/traces/audit/ledger/report/redaction | `compatible-change` | Adds sanitized Stage `02B` report/ledger evidence; no secrets/provider payloads. |
| Alert/runbook semantics | `none` | No alerting or runbook trigger changed. |
| Benchmark / rollout gate | `compatible-change` | Stage `02B` acceptance opens Stage `02C`; futures metadata gate remains fail-closed for later evaluation/activation. |
| Performance hot path | `compatible-change` | Redis publisher adds one small string field; RL live policy forbids hot-path ClickHouse scans and requires degraded/gap repair only. |
| Browser-visible behavior | `none` | No UI/browser behavior changed; prompt disabled browser runtime verification. |
| Docs/runbooks | `compatible-change` | Stage report, ledger and docs index updated only. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/02b-feature-live-feed-contract.md` | passed; `c61f070e31631ff60edbf82b3c08ea57ddb4a12de293a5e606eb3daab9a1e70b` |
| Focused `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_feature_contract.py tests/unit/contexts/market_data/adapters/test_redis_streams_live_candle_publisher.py tests/unit/contexts/strategy/adapters/test_redis_strategy_live_candle_stream.py` | passed; `16 passed` |
| Focused `uv run ruff check ...` on touched Python paths | passed |
| Focused `uv run pyright ...` on touched Python paths | passed; `0 errors` |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `326 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | initially failed before index regeneration because this new report was not indexed; final check passed after `python -m tools.docs.generate_docs_index` |

## Evidence

| Acceptance surface | Evidence |
|---|---|
| Feature contract tests | `test_feature_contract_hash_and_channel_order_are_stable`, vector order, VWAP zero-volume policy and fail-closed missing-field tests passed. |
| Redis/live-feed parity decision | Publisher now emits `trades_count` from `CandleMeta`; consumer parses it when present and preserves old schema-v1 compatibility when absent. RL feature builder blocks missing `trades_count`; no ClickHouse scan is introduced on live candle publish/read. |
| Activation matrix | `training_source_matrix_payload_v1()` is test-covered: only `binance:futures` is `trainable`; Binance spot, Bybit spot and Bybit futures are `blocked_not_training_source_v1`. |
| Futures metadata gate | `futures_metadata_gate_payload_v1()` is test-covered for funding, mark/index, filters, leverage tiers, fee, slippage and liquidation statuses. |
| Secrets/artifacts | No secrets, tokens, cookies, ciphertext, raw provider payloads, raw signed requests, checkpoint tensors, datasets or runtime ML artifacts were written. |

## Blockers And Residual Risks

| Item | Status | Next action |
|---|---|---|
| Stage `02B` acceptance | No blocker | Stage accepted locally with tests, Redis boundary parity tests, docs report and ledger update. |
| Futures metadata | Residual fail-closed gate | Later stages must source funding, mark/index, leverage tiers, point-in-time filters, and explicit fee/slippage/liquidation assumptions before production-grade futures evaluation/activation. |
| Live Redis legacy messages | Compatible but degraded for RL | Existing schema-v1 messages without `trades_count` still parse for strategy consumers; RL feature construction fails closed until a fresh enriched live candle is available. |
| Delivery | `local-only` | No branch, PR, main delivery, CI, Mac Studio deploy or runtime service proof was performed because this stage did not request publishing and changed no deployed runtime state. |

## Next-Stage Handoff

Stage `02C` is allowed to start from this local accepted contract.

It must know:

- Feature contract hash: `d2e99786b68482d730494c6aeec72a1e9f40ac225729019fac5c82f96f900be9`.
- Stage `05` dataset builder must use channel order `open`, `high`, `volume_weighted_average`, `low`, `close`, `volume`, `num_trades` and materialize `float32` feature slabs.
- `volume_weighted_average` derives from `volume_quote / volume_base`; zero base and zero quote uses `close`; missing or inconsistent inputs fail closed.
- Redis/live inference must use the carried `trades_count`; hot-path full ClickHouse scans are forbidden, and repair belongs only to gap/degraded paths.
- Binance spot, Bybit spot and Bybit futures remain `blocked_not_training_source_v1`; no Bybit enrich, feature mask, or research-only Bybit branch was opened.
- Binance Futures metadata gate remains fail-closed for production-grade futures evaluation/activation until funding, mark/index, filter history, leverage tiers, fee, slippage and liquidation assumptions are resolved by later accepted stages.
