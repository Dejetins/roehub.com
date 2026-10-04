# Navigator local Market Data recovery

Scope: the explicitly authorized local Navigator candidate (`localhost:20120`,
API `20111`, `roehub-navigator-ch`). This does not select a production target or
replace `apps/platform-web`. The retired memory-profile runbook stays retired.

## Scheduling and recovery contract

- One request coordinator holds the existing PostgreSQL session advisory lock
  until every worker in its batch drains, including failure paths. Its persistent
  thread pool uses `market_data.ingestion.rest_concurrency_instruments` (dev: 4).
  Claims and orphan recovery run before parallel dispatch; recovery must never
  scan live siblings. Overlapping ranges for the same market/instrument across
  organizations do not execute together. No storage migration is needed for this
  concurrency change; gracefully drain/restart the consumer after a config change.
  `--requests-only` must be used for this local candidate; the ordinary scheduler
  also schedules independent automatic backfill work and is not covered by this
  concurrency guarantee.
- A claim processes at most one 1,440-minute window, then checkpoints and yields.
  Existing REST pagination, canonical minute deduplication and raw-table materialized
  views remain the write path. An incomplete window can be replayed; confirmed
  windows and accumulated progress survive automatic and manual continuation.
- Queue admission allows 200 unfinished jobs per organization. A batch contains
  1–50 distinct instruments, each a separate job; batch insertion is atomic and
  idempotent. Overlapping unfinished work for the same market/instrument is rejected.
- Ready jobs are selected by oldest update. Paused jobs and provider cooldowns
  release the worker. Storage outages also set a shared cooldown to prevent a burst
  of failing inserts across the queue. Retries use 15/30/60/120/240/300 seconds and
  preserve a longer provider Retry-After. Pause and cancellation take priority.
- ClickHouse codes 241, 159, 209 and 210 are retryable; arbitrary DatabaseError is
  not. Only sanitized reason/phase codes are persisted. Permanent errors stop and
  require explicit continuation after the cause is resolved.
- Events are capped at 512 per job and expired after 30 days by worker maintenance.
  Ordinary checkpoint/yield operations do not create events. Pagination uses event
  IDs. Existing jobs begin with an explicit history-start snapshot.

## Coordinated local upgrade

1. Identify current process owners and PIDs. Gracefully stop the requests consumer
   and API writes before applying PostgreSQL phase `market-data-queue-events-0026`.
2. Use the current storage lifecycle/manifest. Preserve job IDs, states, attempts,
   timestamps and counters; compare before/after projections. The migration drops
   the former one-active-job-per-organization index and adds admission/event triggers,
   batches and storage cooldown. Do not downgrade to the old consumer after admission
   of multiple unfinished jobs without a separately reviewed rollback.
3. Rebuild `apps/navigator-web`; restart the local API with
   `ROEHUB_MARKET_DATA_WORK_REQUESTS_ENABLED=1`, then the web candidate and the single
   requests-only consumer. Verify an existing checkpoint advances without resetting.

## Local ClickHouse observability profile

`configs/dev/clickhouse-local-observability.xml` belongs in
`/etc/clickhouse-server/config.d/roehub-local-observability.xml`;
`configs/dev/clickhouse-local-profiles.xml` belongs in
`/etc/clickhouse-server/users.d/roehub-local-profiles.xml`.
Mount these read-only when creating a new local container. A `docker cp` installation
survives a restart but not container recreation. Never run an old preview restoration
script just to install these files: it can seed data or launch unrelated surfaces.

The profile does not increase Docker or server memory limits. It reduces continuous
logging: metrics every five seconds, logs flushed every thirty seconds, warning-level
text log, allocation/CPU/real-time sampling disabled by default. Query and part logs
remain available. Detailed profiling can be temporarily enabled for a bounded
investigation and then disabled again.

Retention: metric/query/error logs seven days; text/part/asynchronous metric/query-view
logs three days; trace/processor-profile logs one day. TTL cleanup is asynchronous.
Existing log tables require matching `ALTER TABLE ... MODIFY TTL ... SETTINGS
materialize_ttl_after_modify=0`; configuration changes can rotate a system table to
`*_0`. Inspect both current and rotated tables and their TTLs. Do not force expensive
TTL materialization/OPTIMIZE under memory pressure. `system.metric_log` uses
`merge_max_block_size=1024`, `max_compress_block_size=65536` and
`min_compress_block_size=8192` to bound the wide-table merge buffers.

The 2026-10-04 trace-log truncation was explicitly authorized. It is a one-time cleanup,
not an automatic startup action. No candle table was truncated.

## Capacity proof and monitoring

Queue length is not write concurrency. One hundred queued instruments have at most
four active request windows with the current dev configuration; they primarily
increase elapsed time and retained data. The ordinary scheduler's automatic REST
pool is independent, so this is not a combined limit for its full runtime. Do not
multiply one INSERT's peak memory by 100 to size this configuration. Measure whole
server memory, background merges, concurrent reads, worker memory and headroom under
representative load before increasing concurrency. A 3.44 GiB budget is an observed
local constraint, not a recommended production capacity.

Read bounded aggregates from `system.metric_log`, `system.query_log`, `system.parts`
and PostgreSQL job/event tables. Keep credentials, raw SQL/provider payloads and
exception text out of evidence. Distinguish isolated per-query fault injection from
uncontrolled server memory pressure, and query-memory peaks from total process RSS.
Include the observation duration and error counts; short progress does not prove
completion of all requested history.
