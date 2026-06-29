# Stage 04: Strategy Runner Integration And ACK Policy

Статус: `accepted`.

Дата: `2026-06-30`.

## Pre-Start

Ledger gate: `market-data-live-tail-repair-v1-stage-ledger.md` указывал `current_stage=04`; Stage `03` принят, доставлен в `origin/main` commit `40a7d3d40917f160a01afba06726fb6332477a20`, CI run `28408445703` завершился `success`, downstream `Deploy Backend` `28408492983`, `Publish App Image` `28408492981`, `Deploy Web` `28408492997` и `Deploy Web` `28408501208` завершились `success`.

Stage `05` до Stage `04 accepted` был закрыт.

## Выбранная ACK Policy

Stage `04` выбрал pending reclaim with no ACK until checkpoint accepts current candle.

Правило:

| Сценарий | ACK решение |
|---|---|
| `ts_open <= strategy_runs.checkpoint_ts_open` | Message stale/idempotent; ACK allowed. |
| `ts_open == checkpoint + 1m` | ACK только после successful checkpoint/result persistence. |
| `ts_open > checkpoint + 1m`, repair success | Repaired candles process in strict `ts_open` order, then triggering candle; ACK only after checkpoint reaches triggering `ts_open`. |
| `ts_open > checkpoint + 1m`, repair miss/failure | No checkpoint advance and no ACK; Redis consumer-group pending list keeps the message for retry/reclaim. |

Durable backlog was not selected because existing Redis Streams consumer-group semantics already provide a bounded pending-list retry surface. Adding a new persistence backlog would expand Stage `04` schema/scope without being required for the current blocker.

## Что Реализовано

| Область | Итог |
|---|---|
| Runner repair dependency | `StrategyLiveRunner` now depends on `ClosedCandleTailProvider` instead of `CanonicalCandleReader`. |
| Runner ACK gating | `_process_candle` returns per-run ACK readiness; `run_once` ACKs a Redis message only when all relevant active run contexts accepted or safely ignored it. |
| Gap repair | `_repair_gap` calls `ClosedCandleTailProvider.get_closed_1m_tail(...)`, processes repaired rows in sorted order, then lets the triggering candle advance checkpoint. |
| Pending retry | Failed repair returns `ACK readiness=false`; triggering message remains pending for the next iteration. |
| Redis adapter | `RedisStrategyLiveCandleStream` first attempts `XAUTOCLAIM`, then current-consumer pending replay (`XREADGROUP ... 0`), then new messages (`>`). |
| Runtime wiring | Strategy live-runner worker wires `MarketDataClosedCandleTailProvider` using Redis hot cache, ClickHouse canonical reader, REST source, and Postgres repair audit repository. |
| Config | Added `pending_claim_min_idle_ms` with default `0`; prod config sets it explicitly. |
| Docs | Updated Strategy runner contract and Market Data plan with selected ACK policy and proof boundary. |

## Runner Proof

Focused runner tests proved:

| Test / call | Evidence |
|---|---|
| Normal contiguous candle | Existing direct runner tests still advance checkpoint and ACK after persistence. |
| Gap repair success | `test_live_runner_gap_repair_retries_and_advances_only_after_full_continuity` processed repaired `10:01`, `10:02` before triggering `10:03`; checkpoint reached `10:03`; one ACK was emitted for `m-gap`. |
| Failed repair no-loss | `test_live_runner_failed_repair_leaves_message_pending_and_later_retry_succeeds` first run left checkpoint at `10:00` and emitted `0` ACKs; second run replayed the same `m-gap`, repaired `10:01`/`10:02`, processed `10:03`, and ACKed once. |
| Dedupe | The later-retry test asserted unique `signal_id` count equals row count and unique `(strategy_run_id, bar_ts_open)` count equals row count after retry success. |

## Redis Pending / Reclaim Proof

Real Redis proof used approved isolated synthetic keys on `macstudio` through an SSH tunnel to local port `16379`.

| Step | Evidence |
|---|---|
| Synthetic stream | `md.candles.1m.stage04-proof.20260629231857-c0ada0c7.binance:spot:BTCUSDT`. |
| Consumer group | `stage04.proof.20260629231857-c0ada0c7`. |
| First consumer | `stage04-consumer-a` read one valid candle message and did not ACK it. |
| Pending check | `pending_after_no_ack=1`. |
| Reclaim consumer | `stage04-consumer-b` read the same `message_id=1782775137184-0` through adapter pending reclaim path. |
| ACK check | After `consumer_b.ack(...)`, `pending_after_ack=0`. |
| Cleanup | `cleanup_deleted=1`, `cleanup_remaining_keys=0`. |

An earlier invalid-payload probe used a synthetic `source` value outside the domain allowlist; the adapter correctly dropped/ACKed the invalid message and cleanup deleted the temporary stream. The accepted proof above used valid `source='ws'`.

## Service Calls / Ops Coverage

| Surface | Stage `04` decision |
|---|---|
| Strategy -> Market Data | Strategy calls only `ClosedCandleTailProvider`; no direct Binance/Bybit REST import or provider secret access. |
| Strategy -> Redis stream | ACK is delayed until checkpoint accepts the triggering candle; failed repair leaves message pending. |
| Redis pending reclaim | Adapter uses bounded `XAUTOCLAIM` and pending replay before new reads. |
| Strategy -> Postgres | `strategy_runs.checkpoint_ts_open` remains the source of truth; no new Strategy persistence table. |
| StrategySignal dedupe | Existing deterministic `signal_id` and repository conflict semantics remain unchanged and were covered by retry proof. |
| Runtime deployment | Stage `04` does not claim production changed-code runtime repair proof; Stage `06` owns `post_main_production_runtime_proof`. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/04-strategy-runner-integration-ack-policy.md` | none | none | Stage report and validation evidence. | `none`; docs/evidence only. |
| none | `src/trading/contexts/strategy/application/services/live_runner.py` | none | Replace ClickHouse-only repair dependency with `ClosedCandleTailProvider` and gate ACK on checkpoint acceptance. | `compatible-change`; stricter ACK semantics for failed repair. |
| none | `src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_streams_live_candle_stream.py` | none | Add pending reclaim/replay before reading new stream messages. | `compatible-change`; retriable pending messages are now surfaced before new messages. |
| none | `src/trading/contexts/strategy/adapters/outbound/config/strategy_runtime_config.py` | none | Add `pending_claim_min_idle_ms` config with default. | `compatible-change`; additive optional config. |
| none | `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py` | none | Carry pending-claim config into worker runtime shape. | `compatible-change`; additive optional config. |
| none | `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py` | none | Wire Market Data provider chain into Strategy runner worker. | `compatible-change`; new runtime dependency on accepted Market Data repair components. |
| none | `configs/prod/strategy.yaml` | none | Make pending reclaim min-idle policy explicit in production config. | `compatible-change`; explicit value equals default. |
| none | `tests/unit/contexts/strategy/application/test_strategy_live_runner.py` | none | Runner proof for gap success, failed repair pending retry, and dedupe. | `none`; tests only. |
| none | `tests/unit/contexts/strategy/adapters/test_redis_strategy_live_candle_stream.py` | none | Adapter proof for pending reclaim before new reads. | `none`; tests only. |
| none | `tests/unit/contexts/strategy/adapters/test_strategy_runtime_config.py` | none | Config loader proof for new field. | `none`; tests only. |
| none | `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_runtime_config.py` | none | Runtime config shim/legacy proof for new field. | `none`; tests only. |
| none | `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md` | none | Update Strategy repair/ACK contract and real Redis proof note. | `none`; docs sync. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1.md` | none | Record selected Stage `04` ACK policy. | `none`; docs sync. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md` | none | Mark Stage `04 accepted` and open Stage `05`. | `none`; staged workflow state only. |
| none | `docs/architecture/README.md` | none | Docs index refreshed after adding Stage `04` report. | `none`; generated docs index only. |

Files outside prompt expected paths: none. The pre-existing foreign change in `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` remains out of scope.

## Validation Evidence

| Gate | Result | Evidence |
|---|---:|---|
| Focused Stage `04` tests | passed | `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_live_runner.py tests/unit/contexts/strategy/adapters/test_redis_strategy_live_candle_stream.py tests/unit/contexts/strategy/adapters/test_strategy_runtime_config.py tests/unit/contexts/strategy/adapters/test_strategy_live_runner_runtime_config.py` -> `35 passed in 0.32s`. |
| Focused ruff | passed | `uv run ruff check ...` on modified Python files/tests -> `All checks passed!`. |
| Focused pyright | passed | `uv run pyright ...` on modified Python files/tests -> `0 errors, 0 warnings, 0 informations`. |
| Prompt `ruff` gate | passed | `uv run ruff check src/trading/contexts/strategy src/trading/contexts/market_data apps/worker/strategy_live_runner tests` -> `All checks passed!`. |
| Prompt `pyright` gate | passed | `uv run pyright src/trading/contexts/strategy src/trading/contexts/market_data apps/worker/strategy_live_runner tests` -> `0 errors, 0 warnings, 0 informations`. |
| Prompt pytest gate adaptation | passed | `tests/integration` is absent; `uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/market_data` -> `244 passed in 1.96s`. |
| Real Redis pending reclaim proof | passed | `stage04_redis_pending_reclaim_proof=ok`; `pending_after_no_ack=1`; `reclaimed_consumer_messages=1`; `pending_after_ack=0`; `cleanup_remaining_keys=0`. |

Prompt gate `uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/market_data tests/integration` is adapted because `tests/integration` does not exist in this repository snapshot. The equivalent Stage `04` evidence is `tests/unit/contexts/strategy`, `tests/unit/contexts/market_data`, focused runner/adapter tests, and real Redis pending reclaim proof above.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No HTTP/API payload changed. |
| Port contract | `compatible-change` | `StrategyLiveRunner` now consumes the existing `ClosedCandleTailProvider` port; external port shape unchanged. |
| DTO schema | `none` | Reuses Stage `01`/`03` repair DTOs. |
| Persisted schema | `none` | No migration; no durable backlog table. |
| Config schema | `compatible-change` | Adds optional `pending_claim_min_idle_ms` with default `0`; prod explicitly sets default. |
| Redis stream semantics | `compatible-change` | Valid pending messages are now reclaimed/replayed before new messages. Invalid payload drop/ACK behavior is preserved. |
| Runtime behavior | `compatible-change` | Failed repair no longer ACKs the triggering message, preventing future candle loss. |
| Logs / redaction | `none` | No raw provider payloads/secrets are logged in Stage `04` changes. |
| Browser-visible behavior | `none` | No UI/browser behavior changed. |
| Performance benchmark / latency claim | `N/A` | Stage `04` makes no speed/latency claim; proof is functional ACK/retry evidence. |

## Delivery Status

Accepted delivery evidence for this stage is reviewed scoped staging, direct-main commit/push, local publish gates and GitHub Actions/CI after push. The exact commit hash is fixed in the final executor report because the hash cannot be written into the commit that creates it.

## Next Stage Handoff

Stage `05` can start after Stage `04` scoped direct-main delivery and CI are green. The next executor can rely on:

- Strategy runner calls `ClosedCandleTailProvider` for gaps and does not directly call exchange REST;
- repaired candles are processed before the triggering candle;
- failed repair leaves the Redis message pending and does not advance checkpoint;
- Redis adapter reclaims/replays pending messages before new entries;
- duplicate `signal_id` and duplicate `(strategy_run_id, bar_ts_open)` remain guarded by deterministic IDs/repository idempotency;
- metrics/alerts/runbook are not complete yet and belong to Stage `05`.
