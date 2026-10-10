# Quick streaming latency check after full-history changes

## Workload and comparison

2026-10-04, closes 19:15–19:17 UTC. Four real Binance Spot subscriptions:
AAVEUSDT, ADAUSDT, ALGOUSDT, ANKRUSDT. Three consecutive closes, 12 candles,
three four-row INSERT batches. Same local host, dev 250 ms buffer and persistent
writer as the previous [warm-writer measurement](../latency-warm/README.md).
No product code or resource settings changed for this measurement.

Reused the existing warm probe/summary with a new output directory, new test YAML,
and three-minute target instead of five. The fixed four-instrument test reader is
the same measurement boundary as the baseline: already-admitted healthy WS write
path. It deliberately does not time full-history startup admission or turn on
persistent user selections. Redis remains disabled only in the test YAML because
it is unavailable locally; this is exchange→canonical ClickHouse, not Redis or
strategy consumer latency.

## Results (milliseconds)

| Metric | Previous sample | Current sample |
|---|---:|---:|
| Close→receive, estimated UTC, mean | 244.5 | 246.5 |
| Receive→canonical completion, mean | 234.9 | 245.3 |
| Receive→canonical completion, sample p95 | 290.7 | 297.5 |
| Close→canonical completion, estimated UTC, mean | 479.4 | 491.8 |
| Close→canonical completion, estimated UTC, sample p95 | 495.9 | 492.8 |
| Full writer call observed range | 14–23 | 24.8–27.5 |

The writer call averaged 26.3 ms in this sample; actual ClickHouse initial INSERT
durations were 15/19/19 ms. All three batches reused PID 64195. The repeated
process/import overhead did not return. The parser, insert buffer, persistent
writer and I/O-process source hashes equal the previous measurement; composition
wiring changed for full-history admission. This supports absence of a large
regression in the observed write path, not proof that its latency is identical:
receive→canonical mean increased by about 10 ms and writer calls were several
milliseconds slower. Three batches cannot isolate noise from a small regression.

All 12 candles were independently observed in canonical after acknowledgement;
all three INSERTs were uniquely correlated in query_log. WS, INSERT and REST
error counters were zero. Four startup tail-repair tasks finished before the
first measured close. The owned probe stopped normally, with zero remaining
writer children, empty recovery journal and unchanged empty user selections.
No fault injection or system/Docker limit changes.

## Clock and proof limits

Read-only NTP probes before/after bound UTC minus local wall time to
+87.5…+214.3 ms. Reported UTC estimates use midpoint +150.9 ms, uncertainty
approximately ±63.4 ms (not a confidence interval). Two post-run Apple NTP
requests timed out; valid Cloudflare and pre-run Apple samples remain in the
clock evidence. Host clocks were not changed. Receive→ack uses monotonic elapsed
time; canonical completion uses query start + duration and synchronous materialized
views with async INSERT disabled, matching the baseline method.

Sample p95 is nearest rank across 12 candles, only three independent close/batch
times. Sequential live traffic is not identical to the prior 20-candle/five-close
sample. No long-run SLO, multi-provider, concurrent-history-load or full admission
performance claim. Against the previously discussed 500–600 ms overall range,
this short sample is within 600 ms even at the upper clock-offset bound (~556 ms),
but does not prove a sustained guarantee.

## Reproduction and evidence

```
PYTHONPATH=src:. .venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_latency_after_history_probe.py
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_clock_probe.py <output>/ntp-check.json
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_clock_probe.py <output>/ntp-after.json
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_latency_after_history_summary.py
```

The pre-clock probe ran before the collector and post-clock probe after it.
`identity.json` records current Git base, dirty-candidate designation, source/harness
hashes and preflight. `measurements.json`, `summary.json`, `candles.csv` and NTP
files hold sanitized timings and aggregates only. No credentials, cookies,
provider payloads or environment dumps retained. Review: focused measurement
self-review; no production change or publication.
