# Local platform data loading contract v1

Status: accepted implementation direction, 2026-09-26. The user's decision applies
across the current local platform and all future pages. It supersedes routine
content fades and the 2026-09-24 report-data entrance animation in the Backtests
iteration log. Existing visual tokens, density and structural layout motion remain.

## Required behavior

- Keep the application shell, controls, existing report and scroll position mounted
  during reads. Never key a whole report by a data selection solely to reload it.
- Distinguish initial loading (no valid data yet), background refresh, a requested
  selection that has not arrived, empty success, recoverable failure, and lost access.
  An initial local loading state is allowed; subsequent reads must not replace a
  valid report with a spinner, skeleton, opacity fade or page transition.
- Keep the requested selection responsive. Retain the last complete view until the
  new view is ready, and label it **Showing previous data** / **Показаны предыдущие
  данные** in the reserved local status row. The displayed entity/variant label and
  its metrics, chart and table must refer to the same identity. Never relabel old data
  with the newly requested identity. A successful empty result is still a commit.
- Start independent reads for the visible panel together. Commit its coherent bundle
  together; hidden panels and all historical trades must not be eagerly fetched.
  Price & trades may fetch its bounded trade-marker data only when selected.
- Forward cancellation signals. Late responses may populate their own query key but
  must not win over a newer selection. Keep existing server Retry-After / 202
  materialization rules, validation, freshness and command eligibility checks.
- A recoverable failure keeps the last view, exposes the failure and a permitted
  local retry. Do not retry a command implicitly. A 401/403/404, changed subject,
  logout or revoked source must hide retained protected data immediately. Never
  reuse snapshots across subjects. Security takes precedence over visual retention.
- Disable commands against a retained old selection; keep navigation and selection
  controls operable. Entity remounts are allowed at an atomic entity commit to
  isolate command drafts, recovery and identity. Do not remove identity keys from
  command-owning components without implementing equivalent state isolation.
- Keep keyboard focus on the initiating control for routine data updates. Use a
  polite local status; motion must never be the sole signal. Reserve status geometry
  so starting/finishing a request does not move the surrounding controls.

## Shared implementation

`apps/platform-web/src/read-snapshot.ts` supplies `useReadSnapshot(scope, key,
candidate, blocked)`. Supply a complete validated candidate or undefined. It retains
one committed view per mounted consumer, not a history or an additional cache. Scope
must include the authorization boundary; pass blocked for access/source loss.
Do not store React elements or command closures as the snapshot payload.

`ReadStatus` in `loading-data.tsx` is the reserved update row. `LoadingData` is for
initial local loading. `result-read.ts` owns result read scheduling and 202 deadlines.
Use query keys containing subject, entity, selection, panel and pagination/timeframe
as appropriate. Do not install global cross-key placeholder data: command, session,
catalog eligibility and resource identity must never inherit another query's data.

`motion.tsx` applies content updates immediately without fading any containing
surface. Structural layout transitions remain interruptible and respect reduced
motion. `chart-instance.ts` initializes/disposes ECharts with the host element;
data effects update the same instance with `setOption`. Remove obsolete series using
`replaceMerge`, preserve applicable zoom, and reset axes/zoom when their meaning
changes. Initial chart animation is disabled. Optional line updates take at most
200 ms (0 for reduced motion); candle updates remain immediate.

## Bounded cache

`query-client.ts` is the only shared QueryClient factory:

- read freshness remains 15 seconds; inactive read GC is 120 seconds;
- inactive, idle read results are capped at 64 entries and 16 MiB of estimated
  serialized UTF-16 payload, evicting least recently observed/updated entries first;
- exceeding either bound triggers eviction after the query notification batch;
- active observers and in-flight requests are not evicted. Their working set is
  governed by visible-panel reads and existing response limits (e.g. 1,500 series
  points, 60,000 candles, 10,000 price markers). Do not call the inactive payload
  estimate a hard total-JavaScript-heap limit;
- session data and command recovery are excluded from read-budget eviction.
  Existing cancel/delete cache markers retain their previous five-minute lifetime;
  durable operation recovery retains its own existing lifecycle;
- no persistent result cache and no unlimited hover/all-variant prefetch. A future
  prefetch feature must share these budgets and specify cancellation/concurrency;
- logout/subject/access boundaries retain their existing query cancellation and
  private-data clearing behavior. Snapshot retention does not override them.

## Current surfaces and scope

Backtests: library filters/pages, selected jobs, variants, report tabs, candle
intervals, result refresh and the builder's dependent reads. Strategies: library,
selected detail/runtime bundle, polling, charts and research detail reads. Overview:
local selections, chart updates and the mock refresh state. Settings, login and
other server-rendered pages currently have no comparable client data-switching
loop; their initial document navigation and authentication forms remain server owned.
Adding asynchronous behavior to those pages must follow this contract.

The separately selected Navigator candidate now applies the same primitives to
`/settings/*`, `/connections`, `/data`, `/data/ingestion` and `/monitoring/*`.
Catalog reads use immutable snapshot IDs, 50-row cursor pages and at most five
active pages. Job and binding histories use 30-row cursor pages; account histories
use 20-row pages. All share the existing QueryClient budget. A catalogue snapshot
evicted from the latest-three projection returns 409 and requires explicit refresh.
Work-request submission stores an opaque recovery key in the URL before dispatch;
unknown outcomes are reconciled by an authenticated lookup and never replayed
automatically. Dirty navigation permits only this internal key update. A detail
401/403 closes related protected inventory immediately; an object 404 remains local
where it does not revoke the collection's authority. Navigator is not a default
cutover of `apps/platform-web`; its proof and remaining gaps are recorded in the
[work-page evidence](../../../../.codex/delivery/evidence/ROEHUB-WORKPAGES-2026-10-03.md).

### Navigator Data progress amendment (2026-10-04)

Within the selected Navigator candidate, Data job progress shares one subject/job
query and polling timer across the catalogue cell, optional journal and inspector.
Same-identity progress reads are silent and leave command controls, focus, scroll
and reserved geometry stable. Initial/retained/error/access-loss states remain
explicit. Derived catalogue, coverage and history snapshots refresh on commands,
manual refresh or job completion rather than on every progress checkpoint. A worker
yield to its internal queue does not reset the public Downloading state after the
job has started. The adjacent ETA is based on recent observed work, includes internal
scheduler yields, and becomes unavailable on read loss or shows stalled progress
instead of counting down without evidence.

### Navigator Data pause and recovery amendment (2026-10-04)

- `pause_requested` acknowledges a pause after the current bounded write/checkpoint;
  `paused` retains the same job, minute cursor and row counters in PostgreSQL.
  Resume continues this job without resetting the checkpoint. A completed final
  window can finish successfully before the pause is acknowledged.
- Pause/resume commands carry `control_version`; an older command cannot reverse a
  newer manual decision. Before publishing a successful command response to the
  shared query cache, cancel any earlier progress GET so its result cannot roll
  back the acknowledged state. Unknown command outcomes still require read
  reconciliation and are never replayed by browser polling.
- `retry_wait` persists temporary-source/storage backoff (15, 30, 60, 120, 240,
  then 300 seconds; a longer provider cooldown wins). Auto recovery keeps the job,
  attempt and checkpoint. Successful progress resets consecutive failure count.
  Manual pause/cancel wins over a concurrent failure; resuming a paused cooldown
  still respects its due time. Paused jobs retain the existing single unfinished
  request slot per organization. Cancel preserves stored candles and is terminal.
- Fenced worker updates and the existing global consumer lock remain authoritative.
  After the lock is lost and the heartbeat is at least five minutes old, recover
  from the persisted checkpoint; pending pause/cancel becomes paused/cancelled.
  Existing REST fill minute-key deduplication reconciles an uncheckpointed write.
- Only classified transient transport/storage errors recover automatically. HTTP
  request/auth failures, invalid instruments, SQL/validation failures and unknown
  exceptions remain terminal. Invalid JSON gets bounded HTTP retries. PostgreSQL
  connection errors with no SQLSTATE cannot always distinguish connectivity from
  connection-setup misconfiguration; this classification is intentionally visible
  as a limitation. Diagnostic logs contain only phase and exception class.
- Browser read loss is independent of job execution: keep the last snapshot,
  disable its commands, stop the ETA, and retry only its GET with bounded backoff
  (up to 30 seconds, honoring Retry-After). 401/403/404/protocol failures do not enter
  this retry loop. No page reload or command replay is used for recovery.
- Pause/wait replaces ETA with its state. Resume/recovery resets the rate sampling
  epoch so downtime is excluded from the new estimate.

Migration `0025_market_data_work_recovery_v1.sql` is additive to existing job data.
Apply it with the local request worker stopped, update API and worker together,
then expose the new client commands. Existing rows default to revision/epoch zero.
Old workers do not understand the new persisted states: mixed-worker rollout and
rollback to the prior worker are not compatible without draining/converting those
states. Auth/org/origin checks and source/raw-candle ownership remain unchanged.
See the work-page evidence for browser versus synthetic-provider proof boundaries.

### Navigator parallel request execution amendment (2026-10-04)

The requests-only consumer now applies `market_data.ingestion.rest_concurrency_instruments`
(dev: four) to a bounded pool under the existing single coordinator lease. Each batch
claims up to that many ready jobs before dispatch, processes one confirmed window
per job and drains every worker before releasing the lease. Orphan recovery runs
only before dispatch. Paused/cooling jobs release slots, and overlapping ranges
for the same market/instrument across organizations are serialized. Queue admission
uses the 0026 contract (up to 200 unfinished jobs/org), superseding the earlier
single unfinished slot statement above. No API/schema changes are required by this
parallelism amendment. Navigator says queued work is waiting for an available slot.
The ordinary scheduler's automatic pool remains independent; the stated cap applies
to explicit requests in the selected requests-only runtime.

## Required regression evidence

1. Cold and warm selection, delayed detail/chart replies, empty success, failed read,
   202 pending/failed materialization, rapid A→B→A and back/forward navigation.
2. Retained labels/metrics/series agree; commands cannot target the previous selection.
   Denial/subject change never keeps prior protected content visible.
3. Chart host and instance survive same-panel data updates; applicable zoom and
   expanded state survive; first drawing and reduced motion do not introduce fades.
4. Cache entry cap, oversized payload eviction, inactive expiry, active observer
   protection and unchanged command recovery behavior.
5. Browser proof on current desktop and narrow layouts, supported locale labels,
   keyboard navigation and absence of new console errors. Mocks supplement browser
   evidence and do not prove server/provider behavior.
