# Navigator streaming verification — 2026-10-04

## Scope and verdict

User-selected scope: Binance Spot AAVEUSDT, ADAUSDT, ALGOUSDT, ANKRUSDT from the previous August download test. Verification only in this turn; no product source/config/schema changes, publication, container resource changes, or platform-client edits. Existing foreign changes were preserved. Review mode: focused source/behavior self-review; no release-readiness verdict.

**Verdict: normal streaming and bounded recovery passed; sustained-failure resilience did not pass.** Compatibility impact of this turn: `none` for product API, persistence, config/defaults and UI contracts. Test-only runtime/config changes were removed or stopped; acquired candles remain stored.

## Real runtime and IAB observations

- At baseline the request scheduler was running, but no standalone WebSocket collector was running. A UI selection alone does not start that process.
- Enabled all four through Codex IAB at `http://localhost:20120/data`; PostgreSQL and worker subscription observations agreed.
- Ran the actual `build_market_data_ws_app`, real Binance WebSocket/REST adapters and real ClickHouse writer/index. The test wrapper observed safe metadata and controlled only its own socket/writer boundary.
- Redis service was unavailable locally. Only the owned test config disabled Redis streams/hotcache. This proves exchange → worker → ClickHouse, **not** live Redis publication or strategy consumption.
- Received/stored 31 closed WS candles. Non-closed updates were ignored. No spontaneous WS transport failures occurred during the bounded window; eight transport errors corresponded to the controlled disconnect/unavailable reconnect attempts.
- Controlled disconnect at approximately 11:09:31 UTC; next successful connection 11:10:22 UTC. Observed retry intervals rose through 0.964, 1.562, 2.458, 4.365, 7.172, 12.111 seconds, with a later successful attempt after approximately 20.6 seconds.
- Disabled ANKR in IAB: worker replaced the four-symbol plan with three symbols at 11:10:39 UTC. The 11:10 minute was received for AAVE/ADA/ALGO only. Re-enabled ANKR: four-symbol plan returned at 11:11:49 UTC and subsequent live candles arrived for all four.
- Subscription replacement reconnects the whole connection group; observed close waits were about 7–10 seconds, in addition to the 15-second selection refresh interval. This is not an instantaneous toggle.
- REST eventually supplied the missed 11:09 minute for all four and 11:10 for ANKR. Final canonical query for `[11:05,11:14)` returned exactly nine rows/nine distinct minutes per instrument: 36 total. Raw metadata showed 31 WS + five REST rows for that window.
- All four August histories remained at 44,640 distinct minutes each.
- One real `TemporarySourceError` occurred in REST catch-up at 11:13:27 UTC. The safe log retained its type only, so the underlying provider/transport cause and instrument of that failure were not established. Do not infer provider rate limiting from this evidence.
- IAB switches returned to OFF; PostgreSQL selections returned to the initial empty set. Browser console check returned zero warning/error entries. Pending commands visibly disabled switches during updates.

## Confirmed findings

### F1 — REST fill queue loses automatic recovery after exhausted HTTP retries

`src/trading/contexts/market_data/application/services/rest_fill_queue.py:223` catches every task exception, emits failure, then removes the pending key. There is no delayed requeue. HTTP-level bounded retries exist, but once exhausted (or when Retry-After is handed upward), this queue does not schedule another attempt.

Isolated probe: executor raised `TemporarySourceError(retry_after_s=0.01)`; after failure the attempt count stayed at one, pending set and queue were empty. Later reconnect/gap events may enqueue other repair work, but this is not guaranteed recovery of the failed task. The real run also observed one task-level `TemporarySourceError`.

Recommended repair: persist or safely retain a bounded repair task with classified transient retry/backoff, preserve Retry-After, make cancellation/shutdown explicit, and test exhaustion followed by recovery without an external reconnect.

### F2 — storage retry has no capacity/backoff or safe failed-shutdown boundary

`insert_buffer.py:98` uses an unbounded queue. `max_buffer_rows` is a flush threshold, not capacity. At `:298`, any exception returns the batch to memory, including non-retryable errors. Timer/size flushes keep retrying without an error-specific backoff. `close()` at `:174` returns normally even when its final flush fails and rows remain only in memory.

Isolated persistent ConnectionError and ValueError probes: threshold two, ten submitted rows, largest batch ten, 18 attempts during a short test with a 20 ms test timer, zero saved rows, ten rows still buffered after normal close. This demonstrates the mechanism, not production throughput or a memory-capacity estimate. Current development timer is 250 ms.

A separate real-ClickHouse disposable-table test produced `DatabaseError`, code **241**, `MEMORY_LIMIT_EXCEEDED`, followed by successful automatic buffer retry: two attempts, one stored row, zero remaining buffer. No server/container memory setting was changed; the memory limit applied only to the owned test query.

Recommended repair: classified backoff, bounded intake/buffer with explicit overload behavior, and a shutdown/restart recovery contract that does not report success while rows are unsaved.

### F3 — graceful stop waits on REST work without a bounded drain

`rest_fill_queue.py:144` waits for workers to drain all queued tasks. Synchronous work invoked through `asyncio.to_thread` has no cooperative stop/deadline contract here.

Real worker stopped its WS intake at 11:14:21 UTC and removed the subscription by 11:14:30. At 11:17:36 (194 seconds after intake stopped), it still had three REST tasks active; ten of 13 tasks had finished, including one failure. The owned test process was force-stopped to end the bounded test. This proves lack of timely shutdown in this run, **not** a permanent deadlock; the precise cause of the long-running REST calls remains undetermined. A separate bounded read-only source diagnostic completed 9,415 rows across ten HTTP pages, with one 11.21-second page, so a general REST outage was not established.

Recommended repair: bounded/cooperative task drain, cancel pending repair work safely, retain restartable repair intent, and expose unresolved shutdown work.

### Additional existing behavior to preserve or decide explicitly

- Empty global selections resolve to `binance:futures:BTCUSDT` in `instrument_selection_repository.py:175`. This is the documented bootstrap default, not a newly introduced regression. “All four toggles OFF” therefore does not mean an otherwise running collector has zero subscriptions globally. The test collector was stopped before the final deselection, so it never subscribed to this out-of-scope instrument.
- Redis publishing failures are covered by existing stub-based tests only; no live Redis claim is made.
- Recovery tests are bounded; they do not prove uninterrupted operation over days, successful completion of all seven-day catch-up tasks, all possible race timings, or production readiness.

## Verification commands/results

Backend: **35 passed** in the focused stream/selection suite:

```sh
.venv/bin/python -m pytest -q \
  tests/unit/contexts/market_data/adapters/test_ws_binance_client.py \
  tests/unit/contexts/market_data/adapters/test_ws_bybit_client.py \
  tests/unit/contexts/market_data/application/services/test_insert_buffer.py \
  tests/unit/contexts/market_data/application/services/test_rest_fill_queue.py \
  tests/unit/contexts/market_data/application/services/test_gap_tracker.py \
  tests/unit/contexts/market_data/application/services/test_reconnect_tail_fill.py \
  tests/unit/contexts/market_data/application/services/test_ws_worker_publishes_redis.py \
  tests/unit/apps/api/test_market_data_reference_routes.py \
  tests/unit/contexts/market_data/adapters/test_postgres_instrument_selection_repository.py
```

Additional HTTP retry suite: **8 passed**:

```sh
.venv/bin/python -m pytest -q tests/unit/contexts/market_data/adapters/test_http_source_retry.py
```

Frontend: **34 passed** (including strategy-pinned controls, stale instrument reads and unknown mutation outcome fencing):

```sh
pnpm -C apps/navigator-web exec vitest run src/data-page.test.tsx
```

These pre-existing passing tests do not cover away F1–F3. Isolated diagnostic probes reproduced failures, rather than treating current broken behavior as successful regression expectations.

Real-infrastructure verification was performed by the owned local probes below, not by a claimed all-green real-infra pytest suite:

```sh
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_probe.py
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_edge_probe.py
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_storage_probe.py
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_rest_probe.py
.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_coverage.py
```

The first live storage fault attempt did not trigger ClickHouse's memory tracker: its test guard raised `AssertionError`, then the real candle batch retried successfully. This remains visible in `runtime.json`; it is **not** claimed as code-241 proof. `storage-fault.json` contains the subsequent separate actual code-241 test using a bounded GROUP BY allocation and a disposable table.

## Evidence and cleanup

- `runtime.json`: safe connection/subscription metadata, metrics and successful WS batch counts; ends at forced test cleanup, not a successful `finished_at`.
- `storage-fault.json`: actual code-241 → buffer retry → stored row.
- `edge-cases.json`: isolated queue/buffer failures and empty-selection behavior.
- `rest-pagination.json`: aggregate-only bounded REST pagination diagnostic.
- `coverage.json`, `summary.json`: final selections, candle counts, retry timings, source hashes.
- `shutdown.json`: active task count, elapsed shutdown bound and forced cleanup outcome.
- `streaming-enabled-iab.png`, `streaming-off-iab.png`: UI evidence.

Owned process PID 24877 stopped; metrics port 20122 no longer served. All owned fault tables were removed. Existing request scheduler PID 22265 was preserved. No secrets, cookies, tokens, raw provider payloads or environment dumps were saved. No fixes to F1–F3 are claimed by this verification turn.
