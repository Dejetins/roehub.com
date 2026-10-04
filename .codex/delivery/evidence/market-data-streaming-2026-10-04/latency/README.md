# Four-instrument streaming latency — 2026-10-04

## Scope and measurement boundary

Observed five consecutive 1m closes per Binance Spot instrument: AAVEUSDT, ADAUSDT, ALGOUSDT, ANKRUSDT. Twenty candles; closes 12:09–12:13 UTC (15:09–15:13 Moscow). Owned collector ran 12:08:21–12:13:03 UTC. No implementation/configuration-source changes, no clock/resource changes, no publication. Compatibility impact: `none`; focused measurement self-review.

The collector used the current production WS composition root and existing REST/ClickHouse adapters with an owned instrument reader fixed to those four symbols. Persistent user selections were empty before and after; no UI toggles were changed. Redis is unavailable locally and was disabled only in the owned test YAML; this measures exchange → worker → canonical, not Redis/strategy consumption or production hosting.

1. Close: exchange candle open + one minute, the normalized exclusive minute boundary.
2. Receive: local timestamp immediately after the WebSocket payload is delivered to the application, before parsing. This is application receipt, not a NIC/kernel packet timestamp.
3. Write: estimated successful INSERT completion from ClickHouse `system.query_log` (`query_start_time_microseconds + query_duration_ms`, millisecond precision). Each of six measured write calls matched exactly one successful initial raw INSERT. `async_insert=0`, `materialized_views_ignore_errors=0`, and the raw→canonical materialized view was present. Each resulting candle was independently read in canonical after acknowledgment.

CSV additionally contains monotonic receive→write-call/ack intervals, local/raw timestamps, exchange event timestamps, UTC-adjusted estimates, and canonical verification timestamps. Database and host clocks were within approximately 1–3 ms in the observed narrow read probes. Query-log completion is an upper boundary for row visibility within a batch, not a per-row physical disk timestamp.

## Clock correction

Binance `/api/v3/time` and read-only NTP probes against Cloudflare/Apple revealed host clock uncertainty. Eight NTP probes before/after bounded external UTC minus host time to approximately **+110 to +231 ms** under the responding servers’ clock accuracy and stable-offset assumption. Reported close-relative estimates use the interval midpoint **+170.7 ms**, with approximately **±60.8 ms** uncertainty. This is not a statistical confidence interval or exact network one-way latency. No clock was changed. Host-internal durations need no external-clock correction. Uncorrected close→receive averages only 94.9 ms, so presenting that alone as actual delivery latency would be misleading.

## Results

Per instrument means, milliseconds; first/last columns include the approximate UTC correction.

| Instrument | Close → receive | Receive → canonical write | Total close → canonical write |
|---|---:|---:|---:|
| AAVEUSDT | 209.7 | 878.7 | 1088.4 |
| ADAUSDT | 190.4 | 897.9 | 1088.4 |
| ALGOUSDT | 320.0 | 892.9 | 1212.9 |
| ANKRUSDT | 342.1 | 870.8 | 1212.9 |

| Interval, ms | Mean | Median | Sample p95 | Min | Max |
|---|---:|---:|---:|---:|---:|
| Close → receive, estimated UTC | 265.5 | 233.4 | 565.4 | 186.1 | 579.2 |
| Receive → canonical | 885.1 | 874.0 | 1133.7 | 782.7 | 1147.5 |
| Total, estimated UTC | 1150.6 | 1094.0 | 1712.9 | 1072.8 | 1712.9 |

p95 uses nearest rank over 20 samples. This is a short observed sample, not a sustained production percentile. The existing local receive→insert p95 ≤1s target was **not met in this sample**: two candles took approximately 1.134/1.148s to database completion (1.151/1.165s to parent acknowledgment).

## Where the time went

- Six INSERT calls for batches of 4, 4, 2, 2, 4, 4 candles. ClickHouse INSERT execution: **7–11 ms** per query, including the synchronous view work; tracked query memory about 6.1 MiB.
- Mean receipt→write-call wait: **261 ms** (range 161–569 ms). Existing batching delay is 250 ms; arrivals within a batch wait different amounts.
- Full process-backed write call: **596–672 ms**. This includes process startup, adapter/client setup, connection/transport, INSERT, polling and child cleanup. The evidence isolates the short database query, but does not separately profile every remaining component.
- At 12:11, ALGO/ANKR arrived later than ADA/AAVE, after the first batch had already begun. They waited for the single writer and formed a second batch. Their total corrected gap reached approximately **1.713s**. No retry or storage error explains this outlier.
- Post-ack canonical verification took **7.9–10.8 ms**. This observation overhead is outside the primary completion measurement but can postpone the next batch by that amount. No overhead was silently subtracted.

## Evidence, reproducibility and cleanup

`measurements.json`: sanitized timestamps, current metrics/config, code hashes. `summary.json`: aggregates and correlated query timing. `candles.csv`: all 20 candles and all measured/derived intervals. `clock-check.json`, `clickhouse-clock.json`, `ntp-check.json`, `ntp-after.json`: clock uncertainty evidence.

Commands (repository root):

```text
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_latency_probe.py
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_latency_summary.py
```

An initial four-candle preflight was interrupted with SIGINT after a probe bug compared naive UTC values returned by ClickHouse to aware UTC values. Independent reads confirmed all four rows; the false presence flags and interruption are preserved in `preflight.json`, excluded from the final sample. The corrected five-minute observation completed normally. No product failure was attributed to that probe bug.

Final run: 20/20 canonical presence checks passed, zero observed WS/INSERT/REST-task errors, journal empty, persistent selections unchanged, test collector/child processes stopped. Startup REST catch-up is outside the measured WS sample. No retries were deliberately injected for this latency measurement. No credentials, provider OHLC payloads, cookies, process tokens or environment dumps were saved.
