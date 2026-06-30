# Stage 06: Mac Studio Repair Proof

Статус: `accepted`.

Дата: `2026-06-30`.

## Pre-Start

Ledger gate: `market-data-live-tail-repair-v1-stage-ledger.md` указывал `current_stage=06`; Stage `05` принят и доставлен в `origin/main` commit `449389a0483482ef2ba1bde59fdf5b9d43c4fda2`.

GitHub Actions для commit `449389a0483482ef2ba1bde59fdf5b9d43c4fda2` в `Dejetins/roehub.com` завершились успешно:

| Workflow | Run | Status |
|---|---:|---|
| `CI` | `28410519230` | `success` |
| `Deploy Backend` | `28410596203` | `success` |
| `Publish App Image` | `28410596218` | `success` |
| `Deploy Web` | `28410596178` | `success` |
| `Deploy Web` | `28410602457` | `success` |

Mac Studio checkout `/Users/daniildegtyarev/Projects/roehub.com` был синхронизирован на `449389a0483482ef2ba1bde59fdf5b9d43c4fda2`, status `## main...origin/main`. Runtime `/opt/roehub/app` содержит Stage `05` live-tail repair metrics wiring and Prometheus rule file.

Proof boundary: `post_main_production_runtime_proof`.

## Controlled Proof

Запуск выполнен на `macstudio` из `/opt/roehub/app` against production Postgres and Redis, but with isolated synthetic Redis keys:

| Field | Value |
|---|---|
| `proof_id` | `20260630002251-4050a989` |
| `run_id` | `e6f07400-43dc-45b1-a9e7-1bdb173a0be7` |
| `owner_user_id` | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| `strategy_id` | `ee15e181-309f-478e-8726-04a299f1292f` |
| `live_profile_id` | `5103b2db-5211-4f62-9e0e-a23605de9b41` |
| `profile_mode` | `testnet` |
| `instrument_key` | `binance:spot:BTCUSDT` |
| `stream_name` | `md.candles.1m.stage06-proof.20260630002251-4050a989.binance:spot:BTCUSDT` |
| `consumer_group` | `stage06.proof.20260630002251-4050a989` |

Timeline:

| Minute | Role |
|---|---|
| `2026-06-30T00:17:00Z` | Initial checkpoint and duplicate stream candle. |
| `2026-06-30T00:18:00Z` | Missing closed minute, absent from stream on first pass. |
| `2026-06-30T00:19:00Z` | Trigger stream candle exposing the gap. |

The proof used a scoped one-shot `StrategyLiveRunner` repository wrapper so only proof run `e6f07400-43dc-45b1-a9e7-1bdb173a0be7` was visible to that process. It did not mutate LaunchAgent producer allowlists, exchange account config, provider credentials, or mainnet state.

## Runtime Result

| Step | Result |
|---|---|
| First iteration | `read_messages=2`, `acked_messages=1`, `failed_runs=0`; target candle stayed pending because repair was incomplete. |
| Checkpoint after first iteration | `2026-06-30T00:17:00Z`; no unsafe checkpoint advance. |
| Redis pending after first iteration | `1`. |
| Second iteration | `read_messages=1`, `acked_messages=1`, `failed_runs=0`; pending target was reclaimed and processed. |
| Checkpoint after second iteration | `2026-06-30T00:19:00Z`; checkpoint advanced across the repaired missing minute. |
| Redis pending after second iteration | `0`. |
| Proof run final state | `stopped`; proof cleanup stopped the temporary run. |

Latency evidence from the one-shot proof process:

| Iteration | Duration |
|---|---:|
| First stalled pass | `0.012584s` |
| Second recovery pass | `0.047030s` |

This is acceptance evidence for bounded proof latency only. It is not a performance benchmark or throughput claim.

## DB Evidence

Postgres rows for proof run:

| Surface | Evidence |
|---|---|
| `strategy_runs` | `metadata_json.stage06_proof_id=20260630002251-4050a989`, `proof_boundary=post_main_production_runtime_proof`, `target_sha=449389a0483482ef2ba1bde59fdf5b9d43c4fda2`, final state `stopped`, checkpoint `2026-06-30T03:19:00+03:00`. |
| `strategy_signals` | `2` rows: outcomes `no_signal`, `signal`; reason codes `ma_cross_no_change`, `ma_fast_crossed_above_slow_testnet_no_order_stage05`. |
| `execution_source_events` | `2` linked rows; source-event outcomes `no_intent`, `recorded`. |
| `execution_intents` | `0` rows for proof run; expected for current `testnet` signal producer behavior. |
| Global execution intent delta | `0`. |

The proof validates `ExecutionSourceEvent` continuation after repair. It intentionally does not create a testnet order or execution intent because the current producer only creates intents for `paper` signals and keeps `testnet` signals at source-event level in this path.

## Repair Audit

Two audit rows were written for the same runner correlation range:

| Status | Sources attempted | Restored / missing |
|---|---|---|
| `miss` | `redis_hot_cache/miss`, `clickhouse/failed`, `rest/miss` | restored `[]`, missing `2026-06-30T00:18:00.000Z` |
| `succeeded` | `redis_hot_cache/miss`, `clickhouse/circuit_open`, `rest/succeeded` | restored `2026-06-30T00:18:00.000Z`, missing `[]` |

The second row proves the intended Stage `06` path: ClickHouse was unavailable or circuit-open, the safe synthetic REST source restored the missing closed minute, and the provider returned a continuous range.

## Redis Hot Cache

Isolated hot-cache keys used prefix `md:hot:1m:stage06-proof:20260630002251-4050a989`.

Before cleanup:

| Check | Value |
|---|---:|
| `zcard` | `1` |
| `hlen` | `1` |
| read-back rows for missing minute | `1` |
| missing minute zset member present | `true` |
| missing minute hash field present | `true` |

Cleanup:

| Check | Value |
|---|---:|
| Redis keys deleted | `3` |
| remaining `stage06-proof` keys | `[]` |

## Metrics / Monitoring Evidence

The proof process used `StrategyLiveRunnerMetrics(CollectorRegistry())` with real provider/runner hooks and asserted these metric surfaces were present after the two iterations:

| Metric surface | Result |
|---|---|
| `market_data_live_tail_gap_total{source_stage="strategy_runner"}` | present |
| `market_data_live_tail_repair_total{source="rest",status="miss"}` | present |
| `market_data_live_tail_repair_total{source="rest",status="succeeded"}` | present |
| `market_data_live_tail_repair_total{source="clickhouse",status="failed"}` | present |
| `market_data_live_tail_repair_total{source="clickhouse",status="circuit_open"}` | present |
| `market_data_hot_cache_miss_total` | present |
| `market_data_hot_cache_hit_total` | present |
| `market_data_hot_cache_write_total` | present |
| `strategy_live_runner_checkpoint_stall_total{reason="repair_incomplete"}` | present |
| `strategy_live_runner_deferred_ack_total{reason="repair_incomplete"}` | present |
| `strategy_producer_source_events_total{mode="testnet",outcome="signal"}` | present |
| `strategy_signal_total{action="open",mode="testnet",outcome="signal"}` | present |
| `market_data_clickhouse_repair_circuit_state` | present |

Mac Studio runtime endpoint and Prometheus checks after deploy:

| Surface | Evidence |
|---|---|
| `http://127.0.0.1:9207/health/ready` | `ready=true`; producer runtime config remains fail-closed with `enabled=false`, `allow_all=false`, allowlist counts `0/0`. |
| `http://127.0.0.1:9207/metrics` | Exposes `market_data_live_tail_gap_total`, `market_data_hot_cache_hit_total`, and `strategy_live_runner_checkpoint_stall_total`. |
| `http://127.0.0.1:9090/api/v1/rules` | Loads market-data live-tail repair rule group and includes `MarketDataLiveTailUnrepairedGapBeyondPolicy` plus `StrategyProducerNoSignalGrowth`. |

## Safety / Redaction

No env values, DSNs, tokens, cookies, credentials, Redis auth values, raw provider payloads, Authorization headers, or exchange account/order payloads were printed into this report. The proof did not attempt mainnet submit and did not mutate exchange account config.

## Strategy-Producer Handoff

The original Stage `12.4` blocker is repaired for the specific live-tail failure mode: a missing closed minute with ClickHouse unavailable no longer forces the runner to fail or stop signal/source-event progress when the short REST tail can restore continuity.

Stage `12.4` itself is still `blocked`, not retroactively accepted: the 6h soak must be rerun from a fresh active-run baseline. Stage `07` of this repair plan owns that rerun/handoff decision.

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/06-macstudio-repair-proof.md` | none | none | Stage `06` runtime proof report. | `none`; docs/evidence only. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md` | none | Mark Stage `06 accepted` and open Stage `07`. | `none`; staged workflow state only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Record that the Market Data live-tail blocker is repaired and `12.4` may be rerun through Stage `07`. | `none`; handoff docs only. |
| none | `docs/architecture/README.md` | none | Docs index refreshed after adding Stage `06` report. | `none`; generated docs index only. |

## Validation Evidence

| Gate | Result | Evidence |
|---|---:|---|
| Stage `05` delivery | passed | `origin/main` commit `449389a0483482ef2ba1bde59fdf5b9d43c4fda2`. |
| GitHub Actions / CI | passed | `CI` run `28410519230` -> `success`. |
| Deploy evidence | passed | `Deploy Backend` `28410596203`, `Publish App Image` `28410596218`, `Deploy Web` `28410596178`, `Deploy Web` `28410602457` -> `success`. |
| Mac Studio checkout sync | passed | `/Users/daniildegtyarev/Projects/roehub.com` head `449389a0483482ef2ba1bde59fdf5b9d43c4fda2`, status `## main...origin/main`. |
| Runtime source/rules presence | passed | `/opt/roehub/app` contains Stage `05` metrics code; `/opt/roehub/config/prometheus.rules/market-data-live-tail-repair.rules.yml` exists. |
| Controlled missing-minute proof | passed | `proof_id=20260630002251-4050a989`; checkpoint advanced from `00:17` to `00:19` only after REST restoration; Redis pending `1 -> 0`. |
| Repair audit proof | passed | One `miss` row and one `succeeded` row for the repaired range. |
| Redis hot-cache proof | passed | Missing minute present in zset/hash before cleanup; read-back rows `1`; cleanup remaining keys `[]`. |
| DB signal/source-event proof | passed | `2` `StrategySignal` rows and `2` linked `ExecutionSourceEvent` rows for proof run. |
| Runtime metrics endpoint | passed | `strategy-producer` metrics endpoint exposes Stage `05` metric names. |
| Prometheus rules endpoint | passed | Prometheus API exposes Stage `05` live-tail repair alerts. |

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API behavior changed in Stage `06`. |
| Port contract | `none` | Stage `06` proves existing Stage `01`-`05` contracts. |
| Persisted schema | `none` | No migration; proof writes expected audit/run/signal/source-event rows. |
| Config / ops | `none` | No LaunchAgent or exchange config mutation; Prometheus rules already delivered by Stage `05`. |
| Redis behavior | `none` | Temporary isolated proof keys were deleted. |
| Runtime behavior | `none` | Stage `06` is proof only; changed runtime behavior came from prior accepted stages. |
| Performance claim | `N/A` | Reported proof iteration durations are bounded proof evidence, not benchmark acceptance. |
| Mainnet / trading risk | `none` | No mainnet submit and no exchange account config mutation. |

## Next Stage Handoff

Stage `07` may start. It must not treat Stage `12.4` as accepted; it should either rerun/open `12.4` with a fresh active selected run and the same signal-path latency/dedup method, or document a new unrelated blocker. Stage `12.5` remains closed until `12.4 accepted`.
