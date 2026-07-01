# Stage 12.4: Sustained 6h Soak

Статус: `accepted`.

Дата старта preflight: `2026-06-27` Moscow / `2026-06-26` UTC.

## Pre-Start

User required before start: nothing. Stage uses existing Mac Studio SSH/runtime access, existing host-local runtime env source `/Users/daniildegtyarev/.config/roehub/roehub.env`, and no secrets from chat.

Previous stage ledger gate: `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` marks Stage `12.3` as `accepted`, `current_stage=12.4`, and records the accepted `2026-06-26` burst/resource gate with controlled `180` `testnet` strategies, Redis pending `0`, no retry/DLQ growth, no production intent/order/mainnet deltas, and active canary run continuity. Stage `12.4` may start only after this accepted gate.

Selected active subject carried from Stage `12.2`/`12.3`:

| Field | Value |
|---|---|
| owner `user_id` | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| strategy `id` | `ee15e181-309f-478e-8726-04a299f1292f` |
| run `id` | `d87917a1-1d72-49a8-b5c5-e40290bd3096` |
| mode / exchange / market | `testnet` / `binance` / `spot` |
| instrument | `binance:spot:BTCUSDT` |
| Redis candle stream / group | `md.candles.1m.binance:spot:BTCUSDT` / `strategy.live_runner.v1` |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md` | none | none | Stage `12.4` measurement method, runtime evidence, decision, and Stage `12.5` handoff. | `none`; documentation/evidence only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Mark Stage `12.4` accepted or blocked and update next-stage handoff after validation. | `compatible-change`; staged workflow status only. |
| none | `docs/architecture/README.md` if docs index changes | none | Required docs index regeneration/check after adding a stage report. | `none`; generated documentation index only. |
| none | `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist` | none | Pre-start repair restored the already accepted scoped producer override after runtime drift reset producer enablement/allowlists. | `compatible-change`; host runtime config stayed scoped to one `paper,testnet` owner/strategy and `allow_all=false`. |
| `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist.stage12-4-20260626T234036Z.bak` | none | none | Recovery copy before the host-runtime LaunchAgent update. | `none`; host-local backup. |
| `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/<run-id>/` | none | none | Durable sanitized snapshot JSONL/status artifacts for the 6h soak. | `none`; host-local evidence artifacts only, no secrets. |

Files outside prompt expected paths: the LaunchAgent and host-local state artifacts are intentionally outside the repo because Stage `12.4` is a target-runtime evidence gate. The LaunchAgent repair used only non-secret scoped producer variables already accepted in Stage `12.1`/`12.2`; the state directory stores sanitized counters, timestamps, Monit summaries, Prometheus query values, Redis counters, and DB aggregates only.

## Preflight Runtime Drift And Repair

Initial preflight artifact: `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260626T233827Z/preflight.json`.

The first preflight blocked the timer:

| Surface | Observed before repair | Decision |
|---|---|---|
| Strategy producer health | `/health/ready` returned `ready=true`, but `producer.enabled=false`. | block timer |
| Producer scope | `allow_all=false`, allowed modes `paper,testnet`, allowed user count `0`, allowed strategy count `0`. | block timer |
| Selected run | DB inventory showed run `d87917a1-...` still `running`, checkpoint current, and signals still being written. | active run exists |
| Source-event path | Last 30m had current `StrategySignal` rows but no linked `ExecutionSourceEvent` rows; latest source event was about two hours old. | block timer |
| Redis / execution | `XPENDING execution.requests.v1=0`, `XPENDING` candle group `0`, retry stream length `1`, DLQ stream length `2`. | baseline risk to track; no timer yet |
| Monit | Default `monit -c /opt/homebrew/etc/monitrc summary` worked. | usable for snapshots |

Repair applied before starting the timer: restored only the accepted scoped host-runtime override in `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist`, then reloaded `com.roehub.strategy-live-runner`.

| Variable | Value |
|---|---|
| `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED` | `true` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOW_ALL` | `false` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOWED_MODES` | `paper,testnet` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOWED_USER_IDS` | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOWED_STRATEGY_IDS` | `ee15e181-309f-478e-8726-04a299f1292f` |

Post-repair readiness before timer:

| Surface | Result |
|---|---|
| `/health/ready` producer payload | `enabled=true`, `allow_all=false`, modes `paper,testnet`, allowed user count `1`, allowed strategy count `1`. |
| Active run | `d87917a1-...` remained `running`; checkpoint advanced to current candles. |
| Fresh source-event proof | Post-repair polling observed linked source events at `2026-06-27 02:41:03+03`, `02:42:02+03`, and `02:43:02+03`. |
| Fresh source-event latency probe | Last 10m post-repair max `StrategySignal.created_at -> ExecutionSourceEvent.received_at` was `0.055015s`. |

## Measurement Method Declared Before Timer

The 6h timer starts only after this method is recorded and after the collector records a baseline snapshot with active run count `>0`, producer enabled, non-empty allowlists, and fresh source-event telemetry.

### SQL Timestamp Sources

| Segment / counter | Source |
|---|---|
| `candle.bar_ts_close -> StrategySignal.created_at` | `strategy_signals.bar_ts_close`, `strategy_signals.created_at`, filtered by selected `strategy_run_id`. |
| `StrategySignal.created_at -> ExecutionSourceEvent.received_at` | `strategy_signals.created_at` joined to `execution_source_events.received_at` by `execution_source_events.strategy_signal_id = strategy_signals.signal_id`, filtered to `source_type='strategy_signal'`. |
| Processed candle count | `count(distinct strategy_signals.bar_ts_open)` for the selected run/window. |
| Unique `StrategySignal` count | `count(distinct strategy_signals.signal_id)` for the selected run/window. |
| Unique `ExecutionSourceEvent` count | `count(distinct execution_source_events.source_event_id)` for joined `source_type='strategy_signal'` rows. |
| Duplicate `signal_id` rows | groups in the selected run/window where `count(*) > 1` by `signal_id`; must be `0`. |
| Duplicate `(strategy_run_id, bar_ts_open)` rows | groups in the selected run/window where `count(*) > 1`; must be `0`. |
| Duplicate source-event idempotency rows | groups in the selected run/window where `count(*) > 1` by `(owner_user_id, source_type, idempotency_key_hash)`; must be `0`, or any duplicate replay must prove it returned an existing row without creating a new source event. |
| Execution side-effect deltas | `execution_intents`, `execution_orders`, `execution_reconciliation_runs`, and `execution_notification_outbox` counts relative to Stage `12.4` baseline. |

Latency quantiles use PostgreSQL `percentile_cont(0.50/0.95/0.99) within group (order by seconds)` and `max(seconds)` over `extract(epoch from timestamp_delta)`. DB windows are UTC-instants even when report tables also show Moscow time.

### Snapshot Cadence

| Snapshot | Window |
|---|---|
| `start` | Baseline at Stage `12.4` timer start; establishes absolute counts and includes a short pre-start telemetry probe. |
| `+1h` through `+5h` | Interval window from previous snapshot to this snapshot, plus cumulative deltas from the `start` baseline. |
| `final +6h` | Final interval window and full Stage `12.4` window from start to final. |

Idle time is not counted toward acceptance. If a snapshot sees `running_strategy_runs=0`, producer disabled, or empty allowlists, the collector records `blocked` and the elapsed timer must not be accepted.

### Prometheus / PromQL

Prometheus provides operational monitoring continuity; DB evidence is the durable per-window acceptance source.

```promql
max(up{job="strategy-producer"})
max(up{job="exchange-execution"})
strategy_producer_signal_lag_seconds{job="strategy-producer"}
histogram_quantile(0.50, sum(increase(strategy_producer_source_event_latency_seconds_bucket{job="strategy-producer"}[$WINDOW])) by (le))
histogram_quantile(0.95, sum(increase(strategy_producer_source_event_latency_seconds_bucket{job="strategy-producer"}[$WINDOW])) by (le))
histogram_quantile(0.99, sum(increase(strategy_producer_source_event_latency_seconds_bucket{job="strategy-producer"}[$WINDOW])) by (le))
100 * (1 - avg(rate(node_cpu_seconds_total{job="node-exporter",mode="idle"}[5m])))
node_load1{job="node-exporter"}
node_memory_free_bytes{job="node-exporter"}
redis_up{job="redis-exporter"}
redis_memory_used_bytes{job="redis-exporter"}
redis_connected_clients{job="redis-exporter"}
pg_up{job="postgres-exporter"}
pg_stat_database_numbackends{job="postgres-exporter",datname="roehub"}
exchange_execution_redis_pending{job="exchange-execution"}
exchange_execution_dlq_total{job="exchange-execution"}
```

`$WINDOW` is the same elapsed interval represented by the DB snapshot: `15m` for the pre-start probe, about `1h` for hourly intervals, and `6h` for the final full-window summary.

### Redis / Monit / Process Evidence

Each snapshot records:

| Surface | Evidence |
|---|---|
| Monit | `monit -c /opt/homebrew/etc/monitrc summary`; default control path is usable at preflight. |
| Process RSS/CPU | `ps` rows for `strategy_live_runner`, `exchange_execution`, Redis, and Postgres. |
| Redis execution streams | `XPENDING execution.requests.v1 exchange-execution.v1`, `XLEN execution.requests.v1`, `XLEN execution.requests.retry.v1`, `XLEN execution.requests.dlq.v1`. |
| Redis candle stream | `XPENDING md.candles.1m.binance:spot:BTCUSDT strategy.live_runner.v1`, `XINFO GROUPS` for the same stream. |
| Health endpoints | `http://127.0.0.1:9207/health/ready`, `http://127.0.0.1:9206/health/ready`. |

## Soak Window

Collector artifact directory: `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260626T234757Z-6h`.

| Field | Value |
|---|---|
| timer start UTC | `2026-06-26T23:47:57.553851+00:00` |
| timer start Moscow | `2026-06-27 02:47:57+03` |
| blocked snapshot UTC | `2026-06-27T02:47:57.772468+00:00` |
| blocked snapshot Moscow | `2026-06-27 05:47:57+03` |
| elapsed before block | `10800s` / `3h00m00s` |
| required final UTC | `2026-06-27T05:47:57.553851+00:00` |
| required final Moscow | `2026-06-27 08:47:57+03` |
| collector PID | `27209` |
| collector final process state | `not_running` after writing `status=blocked` |
| durable status path | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260626T234757Z-6h/latest_status.json` |
| durable snapshots path | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260626T234757Z-6h/snapshots.jsonl` |

Start snapshot result: passed, no `snapshot_errors`. The start snapshot uses a 15-minute pre-start telemetry probe and is not counted as accepted soak elapsed time before `timer_start_utc`.

Collector final status:

```json
{
  "status": "blocked",
  "latest_snapshot_label": "hour_3",
  "reason": "snapshot_failed",
  "errors": [
    "interval has no StrategySignal rows",
    "interval has no linked ExecutionSourceEvent rows"
  ]
}
```

## Snapshot Summary

| Snapshot | UTC | Running runs | Signals / source events | Redis exec pending / retry / DLQ | Dedupe | Decision |
|---|---|---:|---:|---:|---:|---|
| `start` | `2026-06-26T23:47:57.553858+00:00` | `1` | pre-start probe `15 / 7` | `0 / 1 / 2` | `0 / 0 / 0` | pass; mismatch expected because part of the 15-minute probe preceded the scoped producer repair. |
| `hour_1` | `2026-06-27T00:47:57.791908+00:00` | `1` | `60 / 60` | `0 / 1 / 2` | `0 / 0 / 0` | pass |
| `hour_2` | `2026-06-27T01:47:57.831191+00:00` | `1` | `51 / 51` | `0 / 1 / 2` | `0 / 0 / 0` | pass |
| `hour_3` | `2026-06-27T02:47:57.772468+00:00` | `1` at snapshot, but checkpoint stayed at `2026-06-27 04:37:00+03` | `0 / 0` | `0 / 1 / 2` | `0 / 0 / 0` | block: no new signal/source-event rows in the interval |

The current post-run inventory at `2026-06-27 12:50:36+03` shows the selected run later reached `state=failed`, `active_runs_selected_strategy=0`, `checkpoint_ts_open=2026-06-27 04:37:00+03`, and `last_error` reports a ClickHouse HTTP request failure to `http://127.0.0.1:8123` after `ConnectionResetError(54, 'Connection reset by peer')`. This explains why no rerun/closure can treat the blocked collector as an accepted 6h soak.

Resource and queue surfaces stayed stable through the blocked snapshot:

| Surface | Start | Hour 1 | Hour 2 | Hour 3 |
|---|---:|---:|---:|---:|
| Prometheus `strategy_up` / `exchange_up` | `1 / 1` | `1 / 1` | `1 / 1` | `1 / 1` |
| CPU busy 5m | `8.327961888674896%` | `9.687937115817203%` | `8.45064771365025%` | `7.952436860876777%` |
| `node_load1` | `1.53955078125` | `1.6806640625` | `1.80810546875` | `1.4052734375` |
| Redis memory bytes | `1289622048` | `1294542256` | `1299494384` | `1304337968` |
| Postgres backends | `4` | `4` | `4` | `4` |
| Redis candle pending / lag | `0 / 0` | `0 / 0` | `0 / 0` | `0 / 0` |

## Signal-Path Latency And Dedupe

Start pre-start probe:

| Metric | Value |
|---|---:|
| processed candles | `15` |
| unique `StrategySignal` | `15` |
| unique `ExecutionSourceEvent` | `7` |
| duplicate `signal_id` groups | `0` |
| duplicate `(strategy_run_id, bar_ts_open)` groups | `0` |
| duplicate source-event idempotency groups | `0` |
| DB `candle.bar_ts_close -> StrategySignal.created_at` p50/p95/p99/max | `1.995s / 2.9685s / 3.0889s / 3.119s` |
| DB `StrategySignal.created_at -> ExecutionSourceEvent.received_at` p50/p95/p99/max | `0.042971s / 0.0535405s / 0.0547201s / 0.055015s` |

Accepted evidence before the block, cumulative from `timer_start_utc` through `hour_2`:

| Metric | Value |
|---|---:|
| processed candles | `111` |
| unique `StrategySignal` | `111` |
| unique `ExecutionSourceEvent` | `111` |
| source-event outcomes | `no_intent/ma_cross_no_change=85`, `recorded/source_event_recorded=26` |
| duplicate `signal_id` groups | `0` |
| duplicate `(strategy_run_id, bar_ts_open)` groups | `0` |
| duplicate source-event idempotency groups | `0` |
| DB `candle.bar_ts_close -> StrategySignal.created_at` p50/p95/p99/max | `1.515s / 3.0655s / 3.1333s / 3.163s` |
| DB `StrategySignal.created_at -> ExecutionSourceEvent.received_at` p50/p95/p99/max | `0.037689s / 0.054768s / 0.0573362s / 0.058313s` |

Blocked interval `hour_3`:

| Metric | Value |
|---|---:|
| processed candles | `0` |
| unique `StrategySignal` | `0` |
| unique `ExecutionSourceEvent` | `0` |
| duplicate `signal_id` groups | `0` |
| duplicate `(strategy_run_id, bar_ts_open)` groups | `0` |
| duplicate source-event idempotency groups | `0` |
| DB latency quantiles | not measurable; no rows in the interval |

## Safety Boundaries

| Boundary | Status |
|---|---|
| Mainnet submit | Passed for observed window: `mainnet_orders_total=2` at baseline and stayed `2`; no growth. |
| Unknown orders | Passed for observed window: `unknown_orders_total=0` throughout. |
| Reconciliation pending | No growth during observed window: `reconciliation_pending_total=30` at baseline and stayed `30`; these rows are pre-existing June 2026 testnet residue. |
| Notification outbox pending | No growth during observed window: `84` at baseline and stayed `84`; pre-existing baseline. |
| Redis retry/DLQ | No growth during observed window: retry stream length `1`, DLQ stream length `2` at all snapshots. |
| Redis pending | Passed: execution pending `0`, candle pending `0`, candle lag `0` at all snapshots. |
| Resource saturation | No sustained pressure observed before block; CPU, load, Redis, and Postgres values stayed within the Stage `12.3` resource band. |
| Secrets/redaction | Passed by report/artifact inspection scope: no secrets, cookies, DSNs, exchange keys, raw credentials, provider payloads, or session values are recorded. |

## Quality Gates And Review

| Gate | Result | Evidence |
|---|---:|---|
| Runtime collector | blocked | `latest_status.json` ended with `status=blocked`, `latest_snapshot_label=hour_3`, errors `interval has no StrategySignal rows` and `interval has no linked ExecutionSourceEvent rows`. |
| DB signal-path evidence | partial pass before block | `hour_1` and `hour_2` produced measurable DB p50/p95/p99/max latency, `111` cumulative signals/source events, and duplicate counters `0/0/0`. |
| Redis/DB side-effect guard | passed for observed window | Redis retry/DLQ stayed `1/2`; execution pending stayed `0`; unknown orders stayed `0`; mainnet orders stayed unchanged at historical `2`. |
| Focused local code gates | not run | No repo code/config/schema changed. |
| `python -m tools.docs.generate_docs_index --check` | passed | `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.` |
| Cold-head artifact review | passed for historical blocked handoff | Cold self-review fallback completed because independent subagent spawning is disallowed without explicit user request in the active tool policy. |

Cold-head review: completed
Mode: cold self-review fallback
Review scope: historical Stage `12.4` blocked report, stage ledger updates, docs index, proof boundary, file manifest, handoff to rerun.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed for that historical handoff: docs index regenerated; report and ledger both marked `12.4 blocked`; `12.5` remained closed at that point; accepted-stage publish/deploy was not claimed; durable artifact paths and blocker reason were recorded.
Local follow-up check: completed
Residual risks: root cause is only localized to observed ClickHouse HTTP reset and stalled selected run; a future repair/rerun must prove ClickHouse/market-data stability before restarting the 6h timer.

## Historical Delivery Status For 20260626 Attempt

| Surface | Status |
|---|---|
| Repository docs | Updated locally: Stage `12.4` report, stage ledger, and docs index. |
| Runtime code deploy | `N/A`; no repo runtime code changed. |
| Host runtime config | Scoped LaunchAgent override remains applied from pre-start repair; it is non-secret and limited to one owner/strategy, `paper,testnet`, `allow_all=false`. |
| GitHub/main publish | Not claimed as accepted-stage delivery because the `20260626T234757Z-6h` attempt was blocked. |
| Stage advancement | At this historical point, `12.5` remained blocked. |

## Historical Decision For 20260626 Attempt

Final decision for the `20260626T234757Z-6h` attempt: `blocked`.

The `20260626T234757Z-6h` attempt was not accepted because the collector did not cover 6 elapsed hours with active signal/source-event processing. The run produced valid, measurable signal-path evidence for the first two hours after timer start, but `hour_3` had `0` `StrategySignal` rows and `0` linked `ExecutionSourceEvent` rows. That idle hour cannot count toward acceptance, and the selected run later became `failed`.

Contract classification:

| Dimension | Impact | Notes |
|---|---:|---|
| Public API contract | `none` | No API changed. |
| Port / DTO / schema contract | `none` | No code or schema changed. |
| Host runtime config | `compatible-change` | Re-applied the existing scoped producer LaunchAgent override for one `paper,testnet` owner/strategy; repo defaults stayed fail-closed. |
| Runtime side effects | `compatible-change` | The soak observed the existing selected Testnet strategy and wrote normal strategy signal/source-event rows before block; no new order test was submitted for this gate. |
| Browser-visible behavior | `unknown` | Final browser proof was not collected because Stage `12.4` blocked before final acceptance. |
| Delivery | `blocked` for this historical attempt | Report/ledger recorded the blocked evidence; Stage `12.5` remained closed at that point. |

## Historical Handoff Before Fixed Rerun

At this historical point, Stage `12.5` was blocked until Stage `12.4` could be accepted by a later fixed rerun.

Next `12.4` repair/rerun should start from these requirements:

| Required before rerun | Reason |
|---|---|
| Create or restart a fresh selected active run if `d87917a1-...` remains `failed`. | Current post-run inventory shows `active_runs_selected_strategy=0`. |
| Prove ClickHouse/market-data read stability before the 6h timer. | Current run failed after a local ClickHouse HTTP connection reset to `127.0.0.1:8123`. |
| Keep the scoped producer override active: enabled `true`, `allow_all=false`, allowed user/strategy counts `1/1`. | Initial Stage `12.4` preflight found runtime drift back to producer disabled/empty allowlists. |
| Keep the same SQL/Prometheus latency/dedup method. | It successfully produced durable p50/p95/p99/max evidence before the run stalled. |
| Do not count idle intervals toward the 6h timer. | `hour_3` was correctly rejected with no signal/source-event rows. |
| Full candle-to-order-ack latency remains follow-up after `12.4`/`12.5`. | This gate intentionally did not submit extra testnet orders. |

## Candidate Rerun После Market Data Repair

Дата анализа: `2026-06-30`.

После принятого Stage `06` в `market-data-live-tail-repair-v1` на `macstudio` найден новый candidate artifact для Stage `12.4`:

| Поле | Значение |
|---|---|
| artifact root | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T012705Z-stage07-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757` |
| `latest_status.json` | `status=passed`, `reason=completed_6h`, `elapsed_seconds=21600`, `snapshot_count=7` |
| `snapshots.jsonl` | `7` snapshots: `start`, `hour_1`, `hour_2`, `hour_3`, `hour_4`, `hour_5`, `final` |
| selected run | `c2138129-a14a-40b3-bcf0-9ff4cf5a5757` |
| selected strategy | `ee15e181-309f-478e-8726-04a299f1292f` |
| selected profile | `5103b2db-5211-4f62-9e0e-a23605de9b41` |
| instrument | `binance:spot:BTCUSDT` |

Результат signal path из final snapshot:

| Метрика | Значение |
|---|---:|
| cumulative processed candles | `359` |
| cumulative unique `StrategySignal` | `359` |
| cumulative unique `ExecutionSourceEvent` | `359` |
| unlinked signal rows | `0` |
| duplicate `signal_id` groups | `0` |
| duplicate `(strategy_run_id, bar_ts_open)` groups | `0` |
| duplicate source-event idempotency groups | `0` |
| cumulative DB p99 `candle.bar_ts_close -> StrategySignal.created_at` | `3.123s` |
| cumulative DB p99 `StrategySignal.created_at -> ExecutionSourceEvent.received_at` | `0.06462636000000001s` |
| Redis candle pending / lag | `0 / 0` |
| execution pending / retry / DLQ | `0 / 1 / 2`, unchanged from baseline |
| unknown orders | `0` |
| mainnet order count | `2`, unchanged from baseline |

Каждое часовое окно от `hour_1` до `final` записало `60` обработанных свечей, `60` уникальных сигналов и `60` уникальных source events с duplicate counters `0/0/0`. Это сильное candidate evidence, что исходный live-tail blocker не повторился на 6-часовом signal path после Market Data repair.

Бизнес-смысл: стратегия снова выглядит как непрерывный источник сигналов и `ExecutionSourceEvent` в течение 6 часов. Но stage нельзя принимать только по этому признаку, потому что операционная приемка требует еще доказать same-window resource/process и final browser/API proof. Repair observability была проверена отдельно Stage `07`: endpoints и audit table доступны, repair events в candidate window не было.

Service-call coverage для этого report update: `N/A` для новых service calls. Эта секция только классифицирует runtime evidence; она не добавляет код, не вызывает Binance/Bybit, не отправляет orders и не меняет retry/unknown-state behavior. Следующий executor может выполнять только read-only runtime/API/browser checks, если не будет принят отдельный rerun.

Однако этот old candidate artifact оставался `blocked`, пока не был завершен evidence closure:

| Missing / incomplete surface | Почему это блокирует acceptance |
|---|---|
| `processes=[]` in every snapshot | Prompt требует CPU/RAM/process RSS evidence. Пустые process rows не считаются process-resource proof. Повторная Stage `07` проверка показала, что collector обрезал broad `ps` output до парсинга, а same-window Prometheus не содержит process CPU/RSS для `strategy-producer` и `exchange-execution`; reliable same-window resource evidence не восстановлено. Требуется rerun 6h collector с точечным per-process collection и fail-closed validation. |
| Нет final browser/API proof artifact в candidate directory | Финальное user-visible состояние `/strategies` не доказано для accepted state. |
| Repair metrics/audit отсутствовали в candidate artifact, но проверены отдельно | `strategy-producer` `/metrics` экспонирует `market_data_live_tail_*`, `market_data_clickhouse_repair_circuit_state`, `strategy_live_runner_checkpoint_stall_total`, `strategy_live_runner_deferred_ack_total`; `market-data-ws-worker` экспонирует `market_data_hot_cache_*`; `public.market_data_candle_repair_events` доступна, aggregate `241` rows, `miss=240`, `succeeded=1`, candidate-window repair events `0`. В следующем accepted rerun эти checks должны быть частью collector/report, а не ручной post-fact проверкой. |
| Repo reports/ledgers требуют синхронизации | Durable docs должны быть обновлены до старта `12.5`; chat или raw JSON не являются source of truth для stage. |

Решение по этому old candidate artifact оставалось `blocked`; `12.5` тогда оставался закрытым. Следующее действие на тот момент — rerun Stage `12.4` через repair Stage `07` с исправленным process collector, встроенными repair metric/audit checks и финальным `/strategies` browser/API proof.

## Fixed Process Collector Rerun

Стартовая запись: `2026-06-30T16:20:58Z`.

User required before start: `nothing`. Используются существующие `macstudio` SSH/runtime access и host-local env sources; секреты, cookies, DSN, exchange keys и raw provider payloads не записываются в отчеты.

Previous stage ledger gate: Strategy Producer ledger маркирует Stage `12.3` как `accepted`, Stage `12.4` как `blocked`, Stage `12.5` как `pending` / закрытый до `12.4 accepted`. Market Data repair ledger маркирует Stage `06` как `accepted` и Stage `07` как rerun с fixed collector.

Preflight `2026-06-30T16:20:20Z` на `macstudio`:

| Surface | Result |
|---|---|
| selected run | `c2138129-a14a-40b3-bcf0-9ff4cf5a5757`, state `running`, `last_error_present=false`, checkpoint `2026-06-30T19:19:00+03:00` |
| producer scope | ready; `enabled=true`, `allow_all=false`, `allowed_user_count=1`, `allowed_strategy_count=1`, modes `paper,testnet` |
| exchange-execution | ready; `adapter_mode=testnet` |
| Redis | candle pending `0`; execution pending `0` |
| Prometheus | `strategy-producer`, `exchange-execution`, Redis, Postgres and node exporter targets up |
| process rows | non-empty exact `pgrep -f` / `ps -p` rows for `strategy_live_runner`, `exchange_execution`, Redis, Postgres and Prometheus |
| repair metrics/audit | `strategy-producer` exposes live-tail repair/stall/deferred-ACK families, `market-data-ws-worker` exposes hot-cache metrics, `public.market_data_candle_repair_events` exists with `miss=240`, `succeeded=1` |

Measurement method:

| Evidence | Method |
|---|---|
| signal latency | DB timestamp deltas from `strategy_signals.bar_ts_close -> strategy_signals.created_at` and `strategy_signals.created_at -> execution_source_events.received_at`; p50/p95/p99/max per window and cumulative |
| dedupe | DB duplicate groups for `signal_id`, `(strategy_run_id, bar_ts_open)`, and source-event idempotency groups |
| resource/process | exact per-process `pgrep -f` / `ps -p` snapshots; collector fails closed when any required tag is empty |
| Redis | `XPENDING` / `XINFO GROUPS` for candle and execution streams, plus retry/DLQ lengths |
| Prometheus/Monit | instant Prometheus queries and `monit -c /opt/homebrew/etc/monitrc summary` |
| repair observability | runtime metrics endpoint family presence plus `market_data_candle_repair_events` aggregate/window counts |

Artifact root: `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757`.

Collector launch: `2026-06-30T16:30:17Z`; planned final snapshot: `2026-06-30T22:30:17Z`.

## Fixed Process Collector Final Result

Final decision for Stage `12.4`: `accepted`.

The fixed collector rerun completed the required 6-hour active strategy window and closed the evidence gaps that blocked the earlier candidate. The old `20260630T012705Z-stage07-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757` artifact remains valid historical signal-path evidence, but acceptance is based on the new fixed-collector artifact below.

| Field | Value |
|---|---|
| artifact root | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757` |
| `latest_status.json` | `status=passed`, `phase=completed_6h`, `elapsed_seconds=21600`, `snapshot_count=7` |
| timer start UTC | `2026-06-30T16:30:17.687519Z` |
| final snapshot UTC | `2026-06-30T22:30:17.968239Z` |
| selected run | `c2138129-a14a-40b3-bcf0-9ff4cf5a5757` |
| selected strategy | `ee15e181-309f-478e-8726-04a299f1292f` |
| browser/API proof | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/browser_api_proof.json` |
| browser screenshot | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/strategies-final-selected-run.png` |

Snapshot summary:

| Snapshot | Cumulative candles / signals / source events | Window candles / signals / source events | Required process rows | Redis retry / DLQ delta | Decision |
|---|---:|---:|---:|---:|---|
| `start` | pre-window `60 / 60 / 60` | pre-window `60 / 60 / 60` | `strategy_live_runner=1`, `exchange_execution=1`, Redis `1`, Postgres `10`, Prometheus `1` | `0 / 0` | pass |
| `hour_1` | `60 / 60 / 60` | `60 / 60 / 60` | non-empty for all required tags | `0 / 0` | pass |
| `hour_2` | `120 / 120 / 120` | `60 / 60 / 60` | non-empty for all required tags | `0 / 0` | pass |
| `hour_3` | `180 / 180 / 180` | `60 / 60 / 60` | non-empty for all required tags | `0 / 0` | pass |
| `hour_4` | `240 / 240 / 240` | `60 / 60 / 60` | non-empty for all required tags | `0 / 0` | pass |
| `hour_5` | `300 / 300 / 300` | `60 / 60 / 60` | non-empty for all required tags | `0 / 0` | pass |
| `final` | `360 / 360 / 360` | `60 / 60 / 60` | `strategy_live_runner=1`, `exchange_execution=1`, Redis `1`, Postgres `10`, Prometheus `1` | `0 / 0` | pass |

Signal-path final cumulative evidence:

| Metric | Value |
|---|---:|
| processed candles | `360` |
| unique `StrategySignal` | `360` |
| unique `ExecutionSourceEvent` | `360` |
| unlinked signal rows | `0` |
| duplicate `signal_id` groups | `0` |
| duplicate `(strategy_run_id, bar_ts_open)` groups | `0` |
| duplicate source-event idempotency groups | `0` |
| DB `candle.bar_ts_close -> StrategySignal.created_at` p50/p95/p99/max | `1.5145s / 3.01615s / 3.14769s / 3.234s` |
| DB `StrategySignal.created_at -> ExecutionSourceEvent.received_at` p50/p95/p99/max | `0.05146s / 0.060296s / 0.064028s / 0.066287s` |

Safety and observability evidence:

| Surface | Result |
|---|---|
| Redis candle pending / lag | final `0 / 0` |
| execution pending / retry / DLQ | pending `0`; retry and DLQ no growth from baseline |
| unknown orders | delta `0` |
| mainnet orders | delta `0` |
| repair metrics | `strategy-producer` exposes live-tail repair/stall/deferred-ACK families; `market-data-ws-worker` exposes hot-cache families |
| repair audit | `public.market_data_candle_repair_events` exists; aggregate `miss=240`, `succeeded=1`; fixed-rerun window repair events `0` |
| Monit/Prometheus | collected in every snapshot with required targets up |
| secrets/redaction | no secrets, cookies, DSNs, exchange keys, raw credentials, provider payloads, or session values recorded |

Final `/strategies` browser/API proof passed at `2026-06-30T22:39:28.187Z`:

| Surface | Result |
|---|---|
| page | `https://roehub.com/strategies?strategy_id=ee15e181-309f-478e-8726-04a299f1292f` |
| title | `Strategies | Roehub` |
| selected strategy status | `live` |
| UI runtime producer | `running: running` |
| UI selected run state | `running` |
| dashboard API | `200`, `ok=true` |
| selected run id | `c2138129-a14a-40b3-bcf0-9ff4cf5a5757` |
| checkpoint | `2026-06-30T22:38:00Z` |
| latest signal | `2026-06-30T22:39:00.885000Z` |
| latest source event | `2026-07-01T01:39:00.915014+03:00` |
| observed latency gap | `0s` |
| browser console errors / request failures | `0 / 0` |
| local proof JSON | `/Users/daniildegtyarev/Projects/roehub.com/output/playwright/stage12-4-fixed-process-rerun/strategies-browser-api-proof.json` |
| local screenshot | `/Users/daniildegtyarev/Projects/roehub.com/output/playwright/stage12-4-fixed-process-rerun/strategies-final-selected-run.png` |

Known non-blocking dashboard residuals: the final API payload still reports some unrelated dashboard panels as unavailable or not migrated, including `strategy_paper_accounting` unavailable, chart/stat/fills/events panels not migrated, and a stale exchange account projection. These do not block Stage `12.4` because this gate accepts sustained active strategy runtime, signal/source continuity, process/resource evidence, Redis/DB safety counters, repair observability, and `/strategies` runtime/browser/API state. Full `candle -> signal -> source event -> intent -> Redis -> exchange-execution -> testnet order ack` latency remains a later explicit gate after `12.4` / `12.5`.

## Accepted Delivery Status

| Surface | Status |
|---|---|
| Repository docs | Updated locally: Stage `12.4` report, Strategy Producer ledger, Market Data Stage `07` report/ledger, prompt handoff artifacts, and docs index check. |
| Runtime code deploy | `N/A`; no repo runtime code changed during the fixed rerun. |
| Host runtime artifacts | Fixed collector artifacts and browser/API proof copied under the accepted artifact root on `macstudio`. |
| GitHub/main publish | Not staged, committed, or pushed in this executor turn. |
| Stage advancement | Stage `12.4` is `accepted`; Stage `12.5` may start. |

## Handoff To Stage 12.5

Stage `12.5` is now open.

Next prompt:

```text
.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/12-5-closure.md
```

Carry forward these constraints:

| Required in `12.5` | Reason |
|---|---|
| Use the fixed-rerun artifact root as the accepted Stage `12.4` evidence source. | It is the only 6h artifact with non-empty process rows and final browser/API proof. |
| Preserve the old candidate as historical signal-path evidence only. | It still has `processes=[]` and cannot independently open downstream stages. |
| Keep no-mainnet/no-chat-secrets boundaries. | Stage `12.4` accepted without mainnet order growth and without recording credentials. |
| Do not reinterpret Stage `12.4` as full testnet-order latency proof. | This gate intentionally proved signal/source continuity, not full order ACK attribution. |
