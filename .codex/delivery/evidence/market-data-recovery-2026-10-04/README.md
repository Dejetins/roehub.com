# Market Data recovery, queue and local ClickHouse evidence

Date: 2026-10-04. Target: local Navigator candidate `http://localhost:20120`,
API `20111`; real PostgreSQL and ClickHouse `25.8.33.6` in local Docker.
No publication, production rollout, platform-client cutover or Docker memory
increase is included. Unrelated dirty-tree work is preserved.

## Causal diagnosis

Historical rejected INSERTs at 2026-10-03T21:46:43Z and 22:13:07Z had exception
241 and approximately 6.65 MiB query memory. The wider investigation correlated
background merge executor IDs with `system.metric_log`: the 1,435-column table's
merge reached approximately 2.41 GiB and repeatedly failed under the 3.44 GiB
server budget. Twenty-eight merge failures occurred in each incident's five-second
window. This resolves the earlier uncertainty about the dominant concurrent
allocation; it does not establish that no other allocations contributed.

The trace log contained 1,129,552,708 rows / 19,158,778,553 bytes before cleanup.
Continuous allocation profiling at 4 MiB steps plus trace-level logging was
inappropriate for this small local runtime. `clickhouse-before.json` records only
sanitized aggregates. INSERT size alone did not explain either failure.

## Owned implementation

- Existing MarketDataWorkRequestRunner/REST fill retained. Allowlisted ClickHouse
  241/159/209/210 and connection failures enter durable backoff. Unknown schema,
  query and authorization errors remain terminal. Phase/reason codes are sanitized.
- Manual continuation preserves checkpoint, counters and original start time.
  Pause/cancel races retain user intent. One 1,440-minute window per claim bounds
  writes and provides fair turns through the existing global executor lock.
- Migration 0026 replaces the one-unfinished-job index with bounded serialized
  admission, overlapping-work rejection, atomic groups of up to 50 single-instrument
  jobs, a shared storage cooldown and bounded event storage. Cap: 200 unfinished
  jobs per organization. Checkpoints are not rewritten.
- Navigator central row and inspector share command fencing and unknown-outcome
  state. Continue/Pause/Resume/Cancel have accessible names. Error codes distinguish
  stale control, duplicates, full queue and unavailable continuation. Queue waiting
  is distinct from internal single-job yield; queue time remains in ETA samples.
- Events show saved progress, time, reason and next retry; technical phase/code is
  collapsed. Pagination and optional export of loaded events remain inside Navigator.
  Limit: 512 events/job, 30 days; no per-candle events or worker tokens.
- Local ClickHouse XML profiles reduce metric frequency and wide merge buffers,
  bound log retention, and disable continuous allocation/CPU/wall-clock traces.
  Trace truncation was explicitly authorized; candle tables were untouched.
  Current and rotated `metric_log_0` tables both have TTL. The local recreation
  helper now mounts the versioned profiles; it was inspected/edited, not executed.
  See `docs/runbooks/navigator-local-market-data-recovery.md` for lifecycle limits.

## Migration and real provider exercise

The consumer and API were stopped with verified process owner/PID. Migration 0026
was applied under the existing storage migration lock and its manifest marker was
recorded in the same transaction. All 15 existing jobs matched their pre-migration
projection (ID/state/attempt/start/checkpoint/read/written) exactly. In particular,
Bybit was queued at 252,600 and Binance failed at 180,120 confirmed minutes.

The API restarted with work-request routes enabled. A single requests-only worker
continued Bybit automatically. Through Codex IAB, two atomic batches of 50 Binance
Spot instruments were submitted for 2026-09-01T00:00Z–2026-10-01T00:00Z:

- `7d216d9b-a64c-4b9b-946d-2f20930850a1`
- `af2585a0-42f2-4438-b617-9605bfb3bcb5`

All 100 separate jobs were accepted while the previous Bybit job remained active.
ASTSBUSDT was paused from its central row and resumed from the inspector; both
surfaces agreed and the event list showed queued/paused/resumed. A separate 60-minute
ETHUSDT test request was cancelled from the central row with confirmation, and the
inspector/event journal reported cancelled. The 100 monthly jobs were not cancelled.

Binance's original failed job `85e21d9a-ca42-42b8-91df-8fd1dd88d0cf` was continued
from the central row. Attempt became 2 while 180,120 units / 14.97% remained intact.
No manual counter update was performed. Screenshots `final-queue-iab.png`, `queue-events-iab.png` and
`binance-continue-iab.png` capture the actual local UI.

## Verification

- Unit: `.venv/bin/python -m pytest -q tests/unit/contexts/market_data
  tests/unit/apps/api/test_market_data_workspace.py
  tests/unit/apps/api/test_market_data_catalog.py
  tests/unit/apps/migrations/test_storage_lifecycle.py` — **212 passed**.
- Real infrastructure: `tests/real_infra/market_data/test_work_requests_postgres.py`
  with explicitly selected local PostgreSQL/ClickHouse targets — **15 passed**.
  Each test uses an isolated disposable PostgreSQL schema. ClickHouse fault uses
  only an owned disposable table and per-query `max_memory_usage=1`; no global
  memory reduction or production/user download failure was injected.
- Actual driver DatabaseError 241 → retry_wait at 60/180 confirmed units → due retry
  → succeeded at 180/180. Visits `[0,60,60,120]`; table count and distinct minute
  count both 180. Event sequence queued/started/retry_scheduled/recovered/succeeded.
  This proves real storage rejection/classification/runner/PG checkpoint recovery.
  Its generated test rows are not proof of exchange behavior; the separate monthly
  batches exercise actual exchange adapters and candle storage.
- Real PostgreSQL also covers atomic/concurrent idempotency, overlap/capacity
  rejection and rollback, membership/isolation, stale commands, single session
  executor fencing, pause/provider cooldown skipping, shared storage cooldown,
  continuation counters and bounded/paginated events.
- Navigator Vitest: **283 passed** across 28 files, including shared row/inspector
  unknown-outcome fencing, no command replay, stale GET cancellation, queue ETA,
  monthly batch dates and paginated memory-failure diagnostics. Typecheck passed.
- Focused Ruff and Pyright passed. Navigator production build passed with the
  existing large JavaScript-chunk warning. No claim of full-repository CI.
- IAB used exclusively. Real queue admission, row pause/inspector resume, cancel,
  original failed continuation, retained progress and embedded events were observed.
  No console warnings/errors in the inspected view. Network-failure/unknown-command
  edge cases have unit evidence, not an artificially disrupted IAB network.

## Review and compatibility

One independent source review (required by repository risk/security policy) found
three P2 issues: active summaries filtered by coverage period, permanent failure
winning over a requested pause, and fair queue waiting clearing ETA samples. All
were corrected and rechecked. A follow-up noted a duplicate attempt-history entry
for paused-on-error; finish now omits that terminal entry and a real-PG assertion
covers it. No recursive review. Review itself did not execute infrastructure/browser
checks; those are reported separately above.

Compatibility: HTTP commands/reads and Navigator behavior **compatible-change**
(additive endpoints/fields, new typed conflicts; existing states remain). Storage
admission/schema rollout **breaking-change** for uncoordinated old runtimes that
assume a single unfinished job or lack migration 0026. Worker checkpoint/write
contracts **compatible-change**. Organization/same-origin/token protections **none**
(no relaxation). Local observability settings **compatible-change**, verified only
on the stated ClickHouse version/runtime.

## Proof boundary

This is local candidate evidence. The short load observation does not prove all
100 monthly histories completed, sustained production capacity, or absence of all
future memory pressure. An empty pre-listing window counts as checked work, not
invented candles. Automatic recovery mitigates transient pressure; recurring pressure
still requires diagnosis. Queue length does not multiply writer concurrency: this
run keeps one worker. RSS, server-accounted memory and per-query peaks are distinct.
Final aggregate measurements and current job state are stored alongside this report.


## Session outage discovered during IAB verification

A real API restart exposed an existing session-read failure path that permanently
closed the Navigator after a transient 503. The local client now retries only
transient session GETs, with bounded backoff and Retry-After. A previously verified
same-user workspace remains mounted but hidden/inert during the outage, retaining
unsent drafts and unknown-command fences. Initial unverified sessions never mount
private content; 401/403, malformed replies, changed identity and sign-out still
unmount and purge. No command is automatically replayed.

Independent focused security review identified default reconnect refetch as a way
to bypass terminal denial/Retry-After. Session `refetchOnReconnect` is now explicitly
false; scheduled reads own recovery. Tests cover real-timer recovery, unchanged
input identity/value, preserved unknown-command state, 403 purge and no online-event
retry. This is an additional compatible client behavior change, not relaxed access.


## Final observation: 08:39:33 UTC

`final-load.json` covers approximately 21 minutes of real load:

- 100/100 monthly jobs have confirmed progress (at least 2,880 minutes each).
  Total checked: 331,200 minutes; newly written: 316,800 candles. No monthly job
  failed or entered retry_wait. Full 30-day completion is not claimed; jobs remain
  active and the requests consumer is running.
- Original Binance: 181,560 confirmed / 181,500 written, attempt 2. It passed the
  old 180,120 checkpoint; 60 minutes already stored by the failed window were
  deduplicated, so its first new 1,440-minute window wrote only 1,380 new candles.
  Original Bybit: 265,560 confirmed / 260,280 written, attempt 2.
- Server-accounted memory sampled by metric_log peaked at 676,246,601 bytes
  (~645 MiB). A separate 90-sample, two-second observation recorded max one running
  request and peak RSS 1,467,699,200 bytes (~1.37 GiB). These are observed peaks,
  not a guarantee against shorter unsampled spikes.
- 225 successful candle INSERT queries; max query memory 8,495,871 bytes (~8.1 MiB),
  server-side p95 duration 14 ms (not end-to-end exchange latency). Query-log written
  row counts include materialized views; median 2,881 is not 2,881 new candles.
- No non-test query exceptions in this load interval. All three code-241 INSERT
  exceptions target owned disposable fault-test tables. The fault tests never
  lowered server/container limits.
- First-day samples for 0GUSDT, AAVEUSDT and BTCUSDT each have exactly 1,440 raw rows
  and 1,440 canonical rows, with 1,440 distinct minute keys in both layers.
- trace_log remains at 4,442 rows / 104,385 bytes, versus 19,158,778,553 bytes before
  cleanup. Current metric_log is under 1 MiB; retained metric_log_0 is ~109 MiB with
  TTL. Text-log historical disk use remains ~2.62 GiB under the new three-day TTL;
  asynchronous retention is not reported as immediate deletion.

IAB API-outage verification additionally confirmed automatic session and child-read
recovery without page reload. The unsent 00:05–00:25 draft remained intact and its
controls became available again (`session-restored-draft-iab.png`). Worker progress
continued while API was stopped; `api-outage-check.json` records the first interval.
No download command was submitted during this check. The final retry helper also
rechecks current error/deadline, skips recovered/denied queries, rearms an extended
Retry-After and avoids cancelling an in-flight GET; focused tests cover these races.

## Four concurrent Navigator downloads — follow-up, 2026-10-04

The user explicitly authorized changing the explicit-request consumer to four parallel
loads and completing four month-long real-provider jobs. Previous one-writer load
results above remain historical; this section records the new candidate.

Implementation: `MarketDataWorkRequestRunner.run_batch` claims up to the configured
`market_data.ingestion.rest_concurrency_instruments` (dev already 4), then dispatches
bounded windows through a persistent scheduler thread pool. The existing global
PostgreSQL coordinator lease remains held until all futures drain, including submission
and worker failures. Lease checks are serialized on that connection. Orphan recovery
and bounded event maintenance run once before dispatch, never beside live siblings.
Claims serialize overlapping ranges of the same market/instrument across organizations.
The existing REST fill, thread-local ClickHouse gateway, checkpoints, retry/cooldown,
pause/cancel and token fencing remain authoritative. Navigator's queue explanation now
says that it waits for an available download slot. No PostgreSQL migration or API
restart was needed; the verified local requests-only consumer and web were restarted.

Compatibility: API/DTO and persisted schema **none**; explicit-request capacity and
scheduler configuration use **compatible-change** for the supported local Navigator
runtime. No request/state/control-version formats changed. The existing parameter
now controls the explicit pool too. Full scheduler mode still has its independent
automatic REST pool; this is not a combined four-writer cap for that mode. The local
candidate remains requests-only. Config is read on startup; drain/restart the consumer
for changes. No Docker/ClickHouse memory limit changes, publication or platform cutover.

Verification:

- 205 unit tests passed across `tests/unit/contexts/market_data`, the scheduler funding
  tests, API workspace and catalog tests; the new actual scheduler-pool/config test
  also passed (206 total). These are scoped gates, not full repository CI.
- All **20** `tests/real_infra/market_data/test_work_requests_postgres.py` tests passed
  against disposable local PostgreSQL schemas and an owned disposable ClickHouse table.
  New tests prove four concurrent fills with a fifth queued, pause/cancel isolation,
  no live-sibling recovery, cross-org overlap serialization, lease retention while
  siblings drain after executor/worker faults, and fencing of all four writers after
  terminating only the isolated test's lease connection (zero delegated writes).
  The original actual ClickHouse-241 checkpoint recovery test still passes.
- Navigator **283 tests passed**, typecheck/build passed; build retains its existing
  large-chunk warning. Focused Ruff and Pyright passed (0 errors / warnings).
- One independent source review found no blocking defect. Its three fault-proof gaps
  were closed by the new real-PG tests described above. No recursive review.
- Codex IAB exclusively: selected AAVEUSDT, ADAUSDT, ALGOUSDT and ANKRUSDT (Binance
  Spot), set 2026-08-01 00:00 through 2026-09-01 00:00 UTC, submitted one atomic batch
  `20293bda-6f98-48ba-aa34-88c286595350`. All four were observed Downloading together,
  with changing progress and ETA; all four completed. Brief Queued states between
  bounded daily windows are still visible. No page reload was needed during the run.
  Browser click automation needed the native accessibility actions for checkboxes;
  successful UI state, database admission and resulting jobs were verified.
  The inspected final browser console contained no warnings or errors.

Real-provider result (`parallel-four-load.json`, `parallel-four-summary.json`):

- Job creation 10:46:14.630985 UTC; final completion 10:49:46.005522 UTC: **211.37 s**.
- **4/4 succeeded**, each 44,640 confirmed/read/written minutes; **178,560 new candles**.
- Before: no rows in this period for these four symbols. After: for each symbol,
  raw and canonical tables both have 44,640 rows and 44,640 distinct minute keys.
- 110 samples at approximately two-second intervals: max running **4**, monotonic
  aggregate progress, peak tracked memory **859,700,831 bytes**, peak RSS
  **1,903,546,368 bytes** (~1.77 GiB). Sampled peaks can miss shorter spikes.
- 124 successful candle INSERTs; max query memory 6,566,903 bytes; server-side p95
  duration 20.7 ms. This is not provider latency or a comparable speedup benchmark.
- No non-test query exceptions during this interval. The single logged code241 was
  the explicitly injected disposable fault test. All four job retry counts stayed 0.
- Events: four queued, four started, four succeeded. Global active queue empty after
  completion. Existing user data/history retained; no extra test downloads left running.

This proves the selected four-instrument/month workload on the local runtime. It does
not size production, prove a safe seven-writer limit, or establish a fourfold speedup.
The summary includes source hashes for the tested candidate. Screenshots:
`parallel-four-progress-iab.png` and `parallel-four-completed-iab.png`.
