# Streaming slowdown attribution — 2026-10-04

Scope: explain the current measured latency and compare relevant existing implementation/evidence. No product changes or resource changes. Compatibility impact: none. Review: focused measurement self-review.

## Causal evidence

The recent resilience repair replaced the prior reusable in-process/thread writer with `MarketDataIoProcess.write`: every batch starts a fresh spawned Python interpreter, imports the application, builds configuration/adapters/clients, executes one write and tears down the process. The preceding live sample measured full write calls at 596–672 ms, while successful ClickHouse INSERTs took only 7–11 ms.

Control experiment, same local machine/current code and entrypoint-equivalent application imports: invoke the production child body seven times with an empty write. `ClickHouseRawKlineWriter.write_1m([])` returns before constructing a ClickHouse connection or sending any database query. No candles, journal entries or user settings are mutated.

| Component | Median | Observed range |
|---|---:|---:|
| Whole empty process-backed call | 573.77 ms | 567.17–594.47 ms |
| Launch + imports before target entry | 553.12 ms | 543.30–569.82 ms |
| Configuration + empty write body | 2.85 ms | 2.79–3.03 ms |
| Parent polling + cleanup | 18.74 ms | 9.48–27.69 ms |
| Child CPU already consumed at target entry | 549.11 ms | 539.26–565.67 ms |

CPU time being close to startup wall time identifies repeated initialization as actual CPU work, not a wait for ClickHouse or predominantly scheduling starvation. This bounded empty-write probe isolates fixed overhead; it is not a substitute for a full before/after live benchmark. `spawn-profile.json` preserves all seven samples, including the first. No children remained.

The existing configuration additionally waits 250 ms to coalesce a batch. The recent first-arrival timer change makes that interval start at the first row; the older periodic timer's residual wait depended on timer phase. Four simultaneous instruments do not require four independent live writers: candles can share a batch. Adding writers does not eliminate per-process imports or the configured coalescing wait, and increases resource consumption.

A read-only container snapshot during diagnosis was 3.78% CPU and 1.983 GiB / 3.822 GiB for ClickHouse. This is not retrospective telemetry for the earlier live run. There is no evidence that raising Docker memory would solve the measured fixed delay; the empty-write experiment performs no ClickHouse I/O.

## Historical comparison

The repository predecessor buffer used `asyncio.to_thread(self._writer.write_1m, batch)` with a reusable thread-local ClickHouse gateway/client. The new subprocess is the principal newly introduced fixed cost. The 250 ms configuration already existed and must not be described as entirely introduced by this repair.

A focused search found historical 53.54/54.77 ms p95 in `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md`, but that measures `StrategySignal.created_at → ExecutionSourceEvent.received_at`. It is not WS receipt→canonical or close→canonical. A matching historical 40–50 ms candle-ingestion end-to-end baseline was not established. The last reported 1.15 seconds is the mean total close→canonical estimate; current sample p95 is 1.13 seconds receipt→canonical and 1.71 seconds total. Those statistics/boundaries should not be mixed.

## Proposed correction, not implemented in this diagnosis

Keep a supervised, persistent, warmed writer process and reuse its ClickHouse connection; bound/kill/restart that process on timeout or cancellation, preserving the new durable recovery receipts, bounded memory and fencing. Shorten or adapt the 250 ms batching interval for low-volume streaming, with load/part-count measurements before broadening to many symbols. Preserve controlled REST recovery concurrency separately. Re-run the same live four-instrument measurement plus the failure/restart cases after implementation. A 40–50 ms internal processing budget is a target to verify, not a measured promise; total latency also includes exchange/network delivery.

Command:

```text
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_spawn_profile.py
```
