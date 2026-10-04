# Persistent streaming writer — repair and latency verification

## Outcome and scope

User-authorized optimization of the repeated process/import overhead, interpreted from context as approximately 0.5–0.6 **seconds** total latency (500–600 ms). No system/Docker memory changes, schema migration, public API/UI changes, commits or publication. `apps/platform-web` and persistent user selections were not changed.

The production WS composition now owns one persistent, prewarmed raw writer process and ClickHouse client. Startup/connection work occurs before WS intake; subsequent batches reuse that process/client. REST recovery still uses its existing bounded subprocesses and configured concurrency. The 250 ms dev batching interval is unchanged, keeping the main comparison focused on writer reuse.

Private socketpair IPC is frame-bounded (8 MiB), asynchronous on the parent, and serialized to one in-flight request. Cancellation, timeout or uncertain outcome terminates/joins the process before any retry. Parent death also terminates the child while its INSERT is blocked. Durable receipts/backoff remain in the existing buffer/store; a failed generation cannot acknowledge a later request. Owner shutdown drains first and then closes the idle writer. Stop also interrupts prewarming.

Owned product changes: `apps/worker/market_data_ws/wiring/persistent_writer.py`, lifecycle/writer wiring in `apps/worker/market_data_ws/wiring/modules/market_data_ws.py`, associated process/lifecycle tests and the existing WS operational contract. Generated documentation inventory is refreshed from the current shared workspace; previous unrelated changes remain preserved.

## Live before/after results

Same machine, Binance Spot AAVEUSDT/ADAUSDT/ALGOUSDT/ANKRUSDT, five consecutive closes and 20 candles in each sample, same 250 ms timer and instrumentation. Baseline: [preceding latency report](../latency/README.md), 12:09–12:13 UTC. Candidate: **12:27–12:31 UTC** (15:27–15:31 Moscow). These are sequential live samples, not identical provider/network traffic. In particular, the baseline's later ALGO/ANKR arrivals split one minute into two batches. The empty-write startup profile and actual same-PID reuse support attribution of the removed fixed overhead; no equal-distribution production performance claim is made.

| Metric | Before | Persistent writer |
|---|---:|---:|
| Full process-backed write call | 596–672 ms | **14–23 ms** |
| Receive → canonical, mean | 885 ms | **235 ms** |
| Receive → canonical, sample p95 | 1,134 ms | **291 ms** |
| Close → canonical, mean, UTC estimate | 1,151 ms | **479 ms** |
| Close → canonical, sample p95, UTC estimate | 1,713 ms | **496 ms** |
| Close → canonical, observed estimated range | 1,073–1,713 ms | **467–496 ms** |

Per-instrument means (ms):

| Instrument | Close → receive, estimated UTC | Receive → canonical | Total, estimated UTC |
|---|---:|---:|---:|
| AAVEUSDT | 213.9 | 265.5 | 479.4 |
| ADAUSDT | 194.9 | 284.5 | 479.4 |
| ALGOUSDT | 264.2 | 215.2 | 479.4 |
| ANKRUSDT | 304.8 | 174.6 | 479.4 |

All five batches had four candles and used the same writer PID 37096. Five successful initial INSERTs were uniquely correlated with the measured write intervals in ClickHouse query_log. All 20 candles were independently visible in canonical after acknowledgment. WS, INSERT and REST-task error counters stayed zero. No deliberate fault was injected into these live subscriptions.

Clock handling follows the baseline: read-only NTP probes during/after the candidate bound UTC minus local time to approximately +126–253 ms; reported estimates use **+189.7 ms**, uncertainty approximately **±63.3 ms**, under stable offset and responding servers' UTC accuracy. No clock was changed. This is not a confidence interval or exact one-way network timing. The total p95 upper clock-bound estimate is approximately 559 ms, within the contextual 600 ms objective for this short sample. Receive→write measurements do not need the external UTC correction. Database completion is estimated from query start + duration (ms resolution), with raw→canonical synchronous materialized views and async INSERT disabled. Canonical verification overhead occurred after acknowledgment, as in the baseline.

`candles.csv` contains all 20 raw timestamps, intervals and clock-adjusted estimates; `measurements.json` retains instrumentation, actual PID reuse and source hashes; `summary.json` retains correlated query timings and aggregates. `ntp-check.json`/`ntp-after.json` retain clock evidence. p95 is nearest rank over 20 samples; this does not establish a long-duration SLO or other-provider performance. The live probe used the new hot path; the later stop-during-warmup guard/type adjustment was independently regression-tested and does not change the measured write path.

## Failure safety and checks

- **230 unit tests passed** across Market Data, old/new I/O process ownership, storage lifecycle and relevant API routes.
- Focused Ruff and Pyright passed (0 errors); `git diff --check` passed.
- New tests use real spawned processes: reuse/serialization, blocked write cancellation, timeout, concurrent close, idle process death/restart, startup timeout, temporary/permanent classification, parent death, blocked 2 MB transport cancellation without executor threads, and application stop during writer warmup.
- `fault-recovery.json`: actual ClickHouse code **241** on an owned disposable table. Four synthetic rows/receipts existed only in a disposable ClickHouse table and isolated PostgreSQL schema. All four receipts survived the failed generation, a new writer PID confirmed four rows, and the journal returned to empty. Children/tables/schema were cleaned. Test backoff was 0.05 s for a bounded fault proof; production backoff remains unchanged at 15/30/60/120/240/300 s. No user candle or job was overwritten by this test.
- Independent review was required for concurrency/data safety. It found a stop-during-warmup lifecycle issue; that was fixed and re-reviewed with 10 passing process tests. No material findings remained. Focused self-review covered scope and measurements.

Commands (repository root):

```text
.venv/bin/python -m pytest -q tests/unit/contexts/market_data tests/unit/apps/worker/test_market_data_io_process.py tests/unit/apps/worker/test_persistent_writer.py tests/unit/apps/migrations/test_storage_lifecycle.py tests/unit/apps/api/test_market_data_reference_routes.py
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_latency_warm_probe.py
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_latency_warm_summary.py
.venv/bin/python .local_artifacts/navigator-preview-sep26/persistent_writer_fault_probe.py
```

## Compatibility and limits

Public API/DTO, database schema, settings shape and user selection semantics: **none**. Internal lifecycle/operational behavior: **compatible-change**; optional startup/close hooks preserve existing callers, one idle writer now remains resident between batches, with explicit ownership/cleanup. Existing durable recovery, cap, retry classification and stale-ack safeguards remain in force. Process restart still incurs cold-start cost after failure; that recovery latency is not hidden inside the healthy warm result.

Final collector exited normally, owned writer children count zero, recovery journal empty, selections empty before/after (UI untouched). Redis is unavailable locally and disabled only in the owned test YAML; this proves exchange→WS/REST→ClickHouse, not live Redis delivery or strategy consumption. No browser-visible behavior changed, so a new browser flow was unnecessary. No persistent collector service was installed. Long soak, arbitrary PostgreSQL TCP blackholes, other providers and production deployment remain outside this proof. Resource limits were not increased, and short-run latency cannot promise a universal exchange/network upper bound.
