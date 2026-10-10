# 0GUSDT five-minute streaming observation

Read-only live observation of Binance Spot 0GUSDT, 2026-10-07 00:51:05–00:56:05
Europe/Moscow (300.02 seconds). Existing ACTUSDT stream also active. No restart,
selection change, injected faults or application edits during measurement.

Five consecutive new 0GUSDT closed minutes observed in canonical_candles_1m,
source ws, one observed version each. Five shared insert batches / ten raw rows
across both active instruments. No WS/insert errors, reconnects, duplicate or
out-of-order counters; no new collector warning/error or ClickHouse exceptions.
No retry occurred, so recovery under failure was not exercised.

For each 0GUSDT candle, correlated its WS ingestion timestamp (set on receive,
NOT DB completion) with the unique subsequent raw INSERT in the next two seconds.
All five matches unique. Canonical presence verified independently by 2-second
polling. QueryFinish start + duration provides estimated server completion;
correlation is temporal, not an instrument-tagged tracing span.

- Receive to server INSERT completion: 270.816–274.374 ms, mean 272.875 ms.
- INSERT itself including view processing: 15–17 ms, mean 15.8 ms.
- Close to receive, local-clock raw: 115–136 ms, mean 126.6 ms.
- Close to INSERT completion, local-clock raw: 388.489–408.636 ms.
- Read-only NTP offset bounds: +27.65 to +159.40 ms; midpoint +93.53 ms.
- Clock-adjusted close-to-receive estimate: 209–230 ms, mean 220 ms.
- Clock-adjusted close-to-DB estimate: 482–502 ms, mean 493 ms; clock uncertainty
  approximately ±66 ms, separate from inter-sample variability.

Global collector receive-to-write-ACK mean 298.9 ms for ten candles (both symbols),
not a per-0GUSDT value. Raw rows 10 and query_log written_rows 30 are different
boundaries because materialized views also write; do not label as duplicates.
Five points cannot establish robust long-term P95 or availability. No comparative
performance win/regression claimed; no acceptance threshold was specified.

Evidence: observations.json, summary.json, clock.json. Harnesses:
.local_artifacts/navigator-preview-sep26/monitor_0g_readonly.py and
summarize_0g_monitor.py. Focused evidence self-review; secrets/payloads excluded.
