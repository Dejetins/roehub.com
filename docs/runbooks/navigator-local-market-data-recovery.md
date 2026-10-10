# Navigator local Market Data recovery

Scope: the explicitly authorized local Navigator candidate (`localhost:20120`,
API `20111`, `roehub-navigator-ch`). This does not select a production target or
replace `apps/platform-web`. The retired memory-profile runbook stays retired.

## Scheduling and recovery contract

- S03 source uses one request coordinator through the whole rolling pool. A completed
  slot is refilled while unrelated slow siblings continue; orphan recovery runs once
  before dispatch. The persistent thread pool retains
  `market_data.ingestion.rest_concurrency_instruments` (dev: 4), one 1,440-minute
  window per claim and oldest-ready ordering. Overlapping ranges for the same
  market/instrument across organizations remain excluded.
- PostgreSQL session ownership plus the durable coordinator token prevents a rival
  from recovering workers merely because the old PG session disappeared. Stop
  admission, drain every future/owned I/O, then release the token. A stranded token
  requires the confirmed-drain procedure below; never expire it by age alone.
  S03 requires migration `market-data-provider-admission-0034` before new clients.
- History-bound probes run in a separate single-worker loop/coordinator
  (`744948345612`); request execution uses `744948345610`. Neither holds the
  history-inbox admission lock `744948345611` over HTTP. Temporary probe admission
  failures remain `queued` with `next_retry_at`, not false history `unavailable`.
- `--requests-only` remains the measured local candidate mode. The ordinary
  scheduler also schedules automatic backfill. Its REST clients participate in
  shared provider admission, but this does not extend the request queue's range
  exclusion contract to independent automatic work.
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
- ClickHouse codes 202, 241, 159, 209 and 210 are retryable; arbitrary DatabaseError is
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

The profile does not increase Docker or server memory limits. The local renderer
now emits both observability and query-profile files, so recreation preserves the
reduced configuration. Current system metrics remain readable, but continuous
`metric_log`, `asynchronous_metric_log`, `query_metric_log`, `trace_log`,
`processors_profile_log`, `query_views_log` and `query_thread_log` recording is
disabled. Detailed profiling is opt-in for a bounded investigation.

Query, part, warning-level text and error logs remain enabled with one-day TTLs,
128 reserved rows, 4096-row maximum queues and a 2048-row flush threshold. Queues
flush every ten seconds; error counters are sampled every sixty seconds. These
are bounded diagnostics, not an audit ledger: records can be dropped if a queue
fills while storage is unavailable. Existing and rotated retained log tables need
matching `ALTER TABLE ... MODIFY TTL ... SETTINGS materialize_ttl_after_modify=0`.
TTL cleanup is asynchronous; do not force OPTIMIZE under memory pressure.

ClickHouse 25.8 condition-cache accounting omits substantial key/entry overhead.
The local default disables `use_query_condition_cache`; server cache capacity is
also capped at 1 MiB for explicitly opted-in diagnostic queries. This cache changes
scan acceleration only, not candle persistence or query semantics. Do not restore
the default cache merely because its reported byte count appears small.

The 2026-10-04 trace-log cleanup and 2026-10-10 removal of accumulated disabled
metric/profiling logs were explicitly authorized one-time operations. Never put
log truncation in startup or renderer code. No candle table was truncated.

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

## Managed local installation (2026-10-09)

The user's local installation now uses two Compose projects. This replaces the
native preview launchers for this installation; do not run `start_artifacts.py`,
`start_market_data_ws.py`, `start_market_data_requests.py`, `restart_api.py`, or
`restore_runtime.py` alongside it. In particular, the fixture supervisor/main
must not be used to restore this installation: it can seed disposable data.

- `roehub-local`: PostgreSQL, ClickHouse, API, Navigator web, operational-health,
  market-data-ws, requests-only market-data-scheduler and backtest-artifact-publisher.
- `roehub-control`: independent control-agent, the sole Docker socket holder.
  Docker Desktop cannot expose a native macOS Unix socket reliably to the API;
  both services therefore share a Linux socket volume. The controller is outside
  the workload project so a full platform stop remains recoverable.
- User entrypoint: `http://localhost:20120`. Databases have no published ports;
  internal addresses are `postgresql:5432` and `clickhouse:8123`.
- Existing volumes `roehub-navigator-pg`, `roehub-navigator-ch`, installation/owner
  identities, and the existing artifact directory are adopted, without seeding.
  Never start the old stopped database containers while the managed databases run.

The owner-local bundle is `.local_artifacts/platform-local/`. Credentials are
split by service under its mode-0700 `private/` directory (mode-0600 files); do
not print or commit them. API receives only its own control identity. Workers
receive no control keys or Docker socket. The controller image is pinned by
image ID; workload image IDs are checked against `release.json` by the canonical
`DockerComposeControlBackend`. `tools/installation/local_platform.py` generates
Compose, its control policy, independent controller Compose, probes, ClickHouse capacity settings and hashes from
`installation.json` and the owner-local release manifest. This is a local build
trust anchor, not a signed public release.

Run normal lifecycle commands from the repository root through the owner CLI:

```sh
docker compose -p roehub-control -f .local_artifacts/platform-local/control-compose.yaml exec -T control-agent \
  python -m apps.roehubctl.main.main \
  --socket /var/run/roehub/control.sock \
  --identity-file /Users/daniildegtyarev/Projects/roehub.com/.local_artifacts/platform-local/private/owner \
  stop --profile trading --service backtest-artifact-publisher
```

Replace `stop` with `start` to resume. Other worker targets are
`market-data-ws` and `market-data-scheduler`. Omitting `--service` deliberately
starts/stops the whole workload project, including API, website and databases.
The independent controller remains available. Bootstrap it after a controller
stop with:

```sh
docker compose -p roehub-control -f .local_artifacts/platform-local/control-compose.yaml up -d --no-build --pull never --wait --wait-timeout 30
```

Do not use `down -v`. Workload stop uses a 180-second grace period and retains
all volumes, request records and ownership receipts. If a CLI response is lost,
inspect/reconcile the same operation ID through the signed client; a timeout is
not proof that the command failed. Avoid replaying an unknown action with a new ID.
Stopping Docker Desktop stops both projects and every processor in this setup.
`unless-stopped` restores previously running services when Docker restarts;
explicitly stopped services require an explicit start.

Monitoring exposes all workload probes and state-change history. Its three worker
detail pages offer Start/Stop only to the installation owner with recent
re-authentication. The server rejects empty lists, other service targets and
stale observations. Interrupted commands retain their identity across navigation;
an undelivered command can be retried with the same key after a not-found lookup.
API/Web/database start/stop is intentionally in the independent owner CLI because
stopping them also stops the browser's control path. Metrics reachability is shown
as an unconfirmed readiness signal; it does not prove successful artifact builds.
Use Artifacts request history and stored coverage for domain outcomes.

Local capacity settings bound CPU/memory for every workload. ClickHouse query
threads and memory/cache budgets reserve room for workers in Docker Desktop;
these local settings do not alter production defaults. Existing source selections,
requests-only scheduling and imported legacy demand are preserved. Frozen legacy
profiles may contain a subset of indicators; their signal engine validates that
exact frozen subset against supported rules without filling it from mutable YAML.
Ordinary full-catalog startup validation remains strict.

Before migration, PostgreSQL and artifact backups were saved under
`.local_artifacts/platform-local/backups/`. The former unmanaged database
containers are retained stopped for recovery, with restart disabled after cutover.
Rollback requires stopping the managed databases before attaching their volumes
to any old container; never run two database instances against one volume.

### ClickHouse memory-pressure recovery (2026-10-09)

The local installation previously rejected catalog reads and stalled the managed
artifact cycle with ClickHouse code 241 (`MEMORY_LIMIT_EXCEEDED`) despite a green
`/ping`. The 1 GiB server budget coexisted with large default background pools.
An observed horizontal merge of `system.metric_log` (1,435 columns) used about
611 MiB; failing log merges generated more error-log work. Merely reducing query
threads or restarting did not resolve this background allocation pattern.

`tools/installation/local_platform.py` now bounds background merges to four
tasks, lowers merge block sizes, and permits vertical merging for small wide
tables. After applying configuration with a controlled stop/start, the actual
metric-log merge reported `Vertical` with 629,824 bytes at the sampled instant.
This is an incident observation, not a comparable performance benchmark or a
guarantee for every merge. The soft merge budget controls admission, not a hard
per-merge memory ceiling. Watch part accumulation as well as memory.

The ClickHouse container healthcheck now aggregates the adopted candle coverage
table instead of using `/ping`. It detects read-path failure; it does not prove
fresh exchange data or completed artifact jobs. The Monitoring reachability
probe still explicitly does not assert domain readiness.

Do not erase logs, candle tables, failed jobs or artifact history to recover.
Use the independent controller's `stop` then `start` for ClickHouse; the generic
`restart` action rejects that service. Regenerate the image-bound bundle after
renderer changes and wait for the controller socket to become healthy before
submitting another command. The original capacity XML is retained at
`.local_artifacts/platform-local/backups/capacity-before-20261009-incident.xml`.

Local evidence: `.local_artifacts/platform-local/memory-incident-samples-20261009.json`
contains bounded post-recovery reads and memory observations. Browser verification
covered the Candles catalog and a manual BMNRBUSDT artifact refresh reaching
`ready` (request `01830260-747e-49cb-9c53-ec0bf0dd0660`). Existing failed requests
remain historical records; recovery does not silently replay all failed work.
Storage/API formats are unchanged (`none`); local capacity and health changes
are `compatible-change` for this adopted installation. No production rollout
or long-duration stability claim is implied.

## Binance 451 and socket status, 2026-10-10

From the running collector, Binance Spot REST and WebSocket initially returned
HTTP 451 with Binance's restricted-location response. A process healthcheck and
the collector lease remained healthy, so the previous UI falsely reported running
collection. After the operator's network change, the same standard endpoints
returned REST 200 and accepted the WebSocket handshake with a kline message.
No endpoint, proxy, credential or network configuration was changed by this fix.
The existing reconnect repair automatically restored all 14 distinct minutes in
`[2026-10-09 21:35, 21:49) UTC` for market 1 / `0GUSDT`.

Migration 0033 and the updated collector/API/Navigator now distinguish per-symbol
socket state from process health. See the reference API contract for freshness,
lease fencing and coordinated rollout. This adopted preview cannot replay the
complete greenfield bootstrap (0011 rejects its populated strategy tables), so
only additive phase `market-data-stream-connections-0033` was applied under the
normal migration advisory lock and its actual checksum marker recorded. Existing
schema/data and historical markers were preserved.

The controller rejects generic worker `restart`; use its documented graceful
`stop` then `start`. These succeeded (journal sequences 159 and 162); API and web
updates succeeded at sequences 153 and 156. This is an in-container local preview
update, not a rebuilt installation bundle or production delivery.

Validation: 37 focused backend tests passed across focused runs (two fixtures
using `object.__new__` needed the new connection-state member initialized);
49 DataPage tests, TypeScript, Ruff and targeted Pyright passed. The disposable
PostgreSQL database test passed for initial failure, market isolation, expiry,
replacement, capacity and old-lease fencing; the database was removed. Pre-fix
regression red was not captured. The browser showed the real stopped state during
restart, then current collection, 100% history and zero missing candles. At
21:59:29 UTC the latest stored candle opened at 21:58; no browser console errors
were observed. The disconnected UI is covered by the regression test; no new
provider outage was deliberately introduced into the user's running collector.
Review: focused self-review; expanded closed enum requires coordinated client/API
update. Long-duration provider availability is not established by this check.


## Condition-cache memory incident, 2026-10-10

The active 25.8.33.6 runtime rejected coverage/latest-candle reads with code 241
against the unchanged 1 GiB server limit. Sampled failing reads used only 20–31 MiB.
Before intervention `jemalloc.allocated` was 868,058,808 bytes; the condition cache
reported about 32 MiB. The allocator had roughly one million allocations in each
of several small size classes. A read of `system.query_condition_cache` itself
failed under pressure; do not enumerate that cache during an incident.

Hypothesis: many small parts and changing dashboard predicates accumulate cache
entries whose real overhead is undercounted. The installed-version source
[`QueryConditionCache.cpp`](https://github.com/ClickHouse/ClickHouse/blob/v25.8.33.6-lts/src/Interpreters/Cache/QueryConditionCache.cpp)
weights entries using the packed marks/vector, excluding key and other entry/cache
structures. In the same running process, `SYSTEM DROP QUERY CONDITION CACHE`
reduced allocated memory to 419,049,336 bytes after the metrics refreshed, without
a restart or log truncation. This isolates a substantial cache contribution;
it does not attribute every remaining byte or prove the exact originating query.
The previously observed expensive metric-log merge was an additional issue, not
sufficient explanation of this incident.

Applied the cache disable/cap via configuration reload, then gracefully stopped
and started ClickHouse through the controller (journal sequences 165/168) to stop
system-log writers. The server limit remains 1,073,741,824 bytes and container limit
1,610,612,736 bytes. Cleared only the disabled system tables: asynchronous_metric_log,
metric_log, metric_log_0, query_metric_log, processors_profile_log, query_views_log
and trace_log. Their measured active compressed data totalled about 234 MiB before
cleanup; they no longer have active data parts. Preserved query/part/text/error
logs and applied one-day TTLs to both current and rotated retained tables.

Validation: renderer regression test and Ruff passed. Using the real API container
and its existing gateway/config, ran 24 iterations of both production history-count
and latest-candle readers, two seconds apart (22:25:36–22:26:24 UTC). All 48 reads
succeeded, every history check had zero missing candles, and latest candle advanced
from 22:24 to 22:25. Condition-cache entries stayed zero. Allocated memory ranged
343–390 MiB; RSS ranged 805–901 MiB (allocator/page overhead remains and must not
be confused with allocated memory). These post-restart values are not a controlled
before/after performance benchmark. No speedup or long-duration stability claim.
Evidence: `.local_artifacts/platform-local/clickhouse-memory-recovery-20261010.json`.
The real browser showed no stale-data alert, 100% history and zero missing candles.

Compatibility: local diagnostics/retention/cache policy is `compatible-change` for
platform consumers, which do not require historical metric/profiling tables. API,
candle persistence and collection contracts are unchanged (`none`). Focused
self-review; no release or production rollout. Memory and candle correctness were
checked in this bounded live window; quantitative performance acceptance is not
assessed because no benchmark threshold was specified.

## Source trading gaps and derived index recovery (2026-10-10)

Successful REST history reads are authoritative about exchange availability.
A valid empty response is allowed; transport errors, malformed responses and
failed canonical writes remain errors. ClickHouse migration `0003` records
source-absent minute keys separately from real candles. Apply it before starting
updated API, request scheduler and collector. On this adopted local installation
only the new additive migration is applied and checksum-recorded; do not invent
migration receipts for its pre-existing schema.

A bounded REST fill verifies every previously stored/provider-returned minute
against canonical storage, repairs missing bitmap keys, then records successful
source omissions. Metadata writes use the request's lease checks. The derived
index repair never rewrites OHLCV and contributes zero to raw row counters.
Coverage and stream admission count actual keys plus verified unavailable keys;
actual counts and exports contain real candles only. Later actual data overrides
an old absence. Received WebSocket candle receipts still require actual storage.

If a stream is already blocked by an old receipt, first inspect its exact range.
Stop the collector via the owner CLI, acquire its advisory lease and compare
canonical keys with the expected range. Repair only missing derived keys. Use
`complete(entry)` only after canonical verification and a gap-free index read;
leave unresolved receipts intact. Do not clear all failures or reset job cursors.
Restart the collector and verify both catch-up and newly closed minute inserts.
For a failed manual job use its existing Continue action, preserving job identity,
checkpoint and events.

The ACTUSDT incident had a canonical candle at `2026-10-09T13:47:00Z` missing
from the bitmap. A second failed receipt at `07:56:00Z` already had canonical and
index data. Both exact intervals were independently checked before acknowledgement.
The initial cause of the bitmap divergence remains unproven. AAVEUSDT had stopped
at `2020-11-30T06:00:00Z`, a one-hour interval for which Binance returned no candles.
Under the revised contract this is a recorded source absence and processing
continues into later history; it is not a fabricated price series.

The catalog displays actual stream state, independently of the requested switch:
collecting, catching up, retry pending, stopped, no connection, error or checking.
An inspector reports exchange-absent minutes separately. Loading/retry/error rows
keep the same single-line geometry.

Local updates copied into the managed containers survive process restarts but
not container recreation. Repository sources and migration are the durable fix;
a future installation image must include them. This local verification does not
constitute release publication or production delivery.


The same live verification exposed a separate USD-M configuration mismatch:
`wss://fstream.binance.com/stream` established a socket but delivered no klines.
Binance now routes klines through `wss://fstream.binance.com/market/stream`
([official migration notice](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/Important-WebSocket-Change-Notice)).
Production/generated configurations already used that route; dev/test and the
adopted local configuration were corrected. The kline client normalizes that exact
legacy Binance host/path as a compatible fallback, preserving explicit routes,
query parameters and other hosts. This is a `compatible-change` to the provider
connection destination; credentials, payloads and selection semantics are unchanged.
A successful handshake alone is insufficient evidence: verify canonical `source=ws`
rows for multiple newly closed minutes after restart.

While verifying AAVEUSDT, its selected inspector retained the initial history
snapshot while the table refreshed. Active manual downloads now refresh visible
coverage every five seconds, preserving the shared job timer and mounted view.

Observed local proof at `2026-10-10T11:45:33Z`:

- ACTUSDT Spot: `1,005,225 / 1,005,225`, no remaining gaps or repair receipts.
- 0GUSDT Spot: `551,625 / 551,625`; USD-M: `558,480 / 558,480`.
- All three instruments persisted the newly closed `11:43` and `11:44` UTC
  candles with `source=ws`, after the REST catch-up had finished.
- Existing AAVEUSDT job `97d34635-d067-44c2-aef1-10b9127acc6a` advanced from
  `66,420` to `139,920` verified minutes after continuation, with no new error
  events. Source-absent minutes were `372`, including the original 60-minute
  halt. Full history remained partial and the job was left running.
- Browser `1414x959`: loading and normal rows were both `38px`; the spinner
  used `data-loading-spin`; no horizontal page overflow or console warnings/errors.
  AAVEUSDT history changed from `6.32%` to `6.41%` in both table and inspector
  without navigation or refresh.
- Verification: 239 focused backend tests, 4 tests on a disposable ClickHouse
  database, and 65 Navigator tests passed. Scoped Ruff/Pyright, TypeScript and
  the production client build passed. The build retains its existing large-chunk
  warning. The isolated ClickHouse runner disables unrelated pytest plugins and
  reports the unused asyncio configuration warning; all four tests executed.
- Focused self-review covered lease fencing, returned-but-unpersisted candles,
  partial source responses, replay/idempotency, late real candles overriding source
  omissions, migration ordering and rollback. No unresolved finding in this scope.

Sanitized snapshots and the browser screenshot are under
`.cache/candle-repair-2026-10-10/`. They prove this local observation interval, not
completion of AAVEUSDT's entire multi-year history or a published release.

## Catalog read recovery and consistent stream status (2026-10-10)

The annotated AAVEUSDT view retained an old catalog with 99.96% / disabled while
its independent inspector showed 100% / enabled. At 16:40:14 UTC ClickHouse
rejected reads with code 241 under the existing 1 GiB total memory limit. The
failed latest-candle reads each allocated approximately 138.5 MB. Navigator's
catalog/inspector interval returned `false` after any read error, so one transient
failure could freeze either surface indefinitely. This was a read failure, not a
failed candle-ingestion command; manually refreshing the old client recovered it.

The client now retries only transient GET failures with bounded backoff and
Retry-After, identifies the affected list/instrument in its error, and marks
unverified stream states `Нет обновления`. The selected row shares the fresh
inspector facts only for the same identity and current catalog key. Filter/order
membership and retained-command locks remain intact. Stream intent is labelled
`Поток: включён`, healthy runtime `Работает`, inspector `Поток свечей`.

The latest actual candle reader now queries the exact actual-minute bitmap used
by history coverage. It excludes the upper bound and never counts source absences.
No candle data, schema, endpoint shape, or stream selection was changed. As with
coverage, the derived index must be maintained/repaired using the preceding
recovery procedure; this is not a new independent source of candle truth.

Bounded local comparison: three alternating old/new reads on the same live
ClickHouse 25.8 installation, market 1 / AAVEUSDT, fixed exclusive bound
2026-10-10T16:43:00Z, one query thread and unchanged cache configuration. All six
reads returned minute 29860842. Existing warm caches were retained, no cache flush
or container resource change was used. Old query read 5,816,933 rows with
138,491,172–138,491,268 bytes of query memory; new query read 105 index rows with
4,203,644 bytes. Query-log duration ranges were 293–323 ms and 25–26 ms respectively.
These measurements compare this reader only, not total process memory or end-to-end
API latency. No numerical performance acceptance threshold was specified. An
initial canonical ORDER BY/LIMIT candidate also hit its 128 MiB query cap and was
rejected; its failed measurements remain in the local evidence directory.

Validation:

- 69 tests across DataPage, download-progress and market-data-ui passed; TypeScript
  and Vite build passed (existing bundle-size warning). New cases exercise automatic
  catalog/inspector recovery, no command replay, stale status, matching selected
  facts, capped delays/Retry-After, and non-retriable access/protocol errors.
- Eight API catalog tests passed. Five disposable real-ClickHouse tests passed,
  including empty/latest/exclusive-bound/duplicate/other-instrument/source-absence
  behavior. Ruff and targeted Pyright passed. Pytest's isolated container warned
  about an unavailable asyncio plugin configuration; all five tests executed.
- In the user's in-app browser, the updated client displayed matching AAVEUSDT
  100% / Работает. Controller API stop/start (journal 210/213) exercised real read
  failure; both panels showed specific errors and locked controls. They recovered
  automatically without refresh or command replay. Web update: journal 207.
- Filter `Поток: включён` returned four enabled instruments. Desktop 1414×959 rows
  stayed 38 px; narrow 390×844 rows stayed 42 px with no page horizontal overflow.
  The temporary viewport was reset and original filter restored. Browser console
  inspection returned no warnings/errors. Expected failed API reads were induced
  only during the controlled stop, not hidden from the verification result.
- Canonical AAVEUSDT candles at 16:47 and 16:48 UTC had `source=ws`; the browser's
  last candle subsequently advanced to 16:49 with zero missing candles. Verified
  exchange absences remained 1,445 minutes and were excluded from expected history.

Evidence and comparison harnesses: `.cache/data-read-recovery-2026-10-10/`.
Focused self-review found no remaining scoped blocker. Compatibility is
`compatible-change` for visible read recovery/copy and `none` for API, auth,
commands and persistence. This is a local in-container update; source and docs
are saved, but container recreation requires an image containing the changes.
The bounded checks do not establish long-duration availability or absence of all
future server-wide memory pressure. No release, commit or remote publication.


## Indexed artifact source reconciliation — S01 (2026-10-10)

The publisher source adapter now uses the existing actual-minute bitmap in pages
of at most 50 coordinates. A sweep has one index SELECT per page and retains the
30-second managed-cycle pause and persisted cursor. Single-coordinate hints use
the same index; repeated batch coordinates coalesce and hint failure cannot retry
a committed candle write. Existing managed worker entrypoints leave hints optional;
periodic reconciliation is required even when no observer is supplied.

A source absent from the index is not evidence that canonical history is empty.
For a suspected stale/missing state, use the preceding bounded canonical verifier
and current lease/receipt recovery procedure. Inspect actual keys and source absence
separately, then verify the next sweep observes the repaired range. Missing tables,
query errors or memory-limit failures must remain failed cycles; do not add an
unbounded fallback or reset cursors/jobs. Administrative deletion still needs a
separately scoped index rebuild. Ping/process health alone proves none of this.

The finite comparison harness is `tools/qa/market_data_resource_comparison.py`.
It requires existing `CH_*` credentials supplied privately and creates/removes only
its unique `resource_s01_*` database. `--baseline` points to the preserved pre-change
source snapshot, never Git HEAD; `--output` must name a new evidence file.
Use `--coordinates 102 --page-delay 30` for a representative cadence comparison,
and `--mode index-only --coordinates 1020` only after checking headroom. The latter
removes its disposable canonical table before reading, verifies generated expected
bounds and proves the scale path cannot fall back to canonical. It warms both paths,
alternates three pairs, verifies identical observations and gathers tagged query-log
counts/rows/bytes/memory/CPU/IO without storing SQL or credentials. Merges are paused
only for the disposable fixture tables. Existing caches and server limits remain
unchanged. Finite sweep timing includes the selected inter-page delay; it does not
measure artifact publication, provider capacity or API latency. Shared server
memory samples cannot be attributed exclusively to this workload.

Upgrade the source observation service and adapter together in an image-bound
managed bundle using the current installation lifecycle; do not use `docker cp`
as delivery. Restore the previous bound image/config for rollback, preserving
volumes, source cursor, requests and selections. No schema migration is required.
Only after activation can runtime query logs prove that the installed publisher
uses the new page path. The S01 report records fixture verification and installed
runtime state separately; a source checkout or receipt is not deployment evidence.


## ClickHouse workload budgets — S02 (2026-10-10)

The managed local renderer supports `clickhouse_workload_budgets` values `legacy`
(default, existing clients), `provisioned` (server identities installed; legacy
clients retained), and `active` (composition-owned class clients). This is an
ordered installation contract, not proof that this installation was activated.
See the pack's `reports/S02.md` for measured source/runtime identities and limits.

| Class / server identity | Participating clients | Query / aggregate tracked memory | Concurrent queries | Threads |
|---|---|---|---|---|
| `stream` / `roehub_stream` | Collector main, persistent writer, killable REST/prepare/plan children; strategy live reader | 128 / 192 MiB | 2 | 2 |
| `interactive` / `roehub_interactive` | Reference/catalog/workspace API, strategy dashboard reads | 128 / 192 MiB | 4 | 2 |
| `backfill` / `roehub_backfill` | Requests and normal scheduler, CLI catchup/raw writer | 128 / 128 MiB | 1 | 1 |
| `artifacts` / `roehub_artifacts` | Publisher source discovery, canonical/funding readers, builder and CLI publisher | 128 / 128 MiB | 1 | 1 |

All classes have a 30-second query limit (`throw`, clock-based checks). Native
checks are cooperative, not a transport cancellation deadline. Lower nonzero
query limits remain legal; aggregate/concurrency limits cannot be changed by
runtime users, including by switching profile. INSERT and chained materialized
views execute in the submitting identity. Limits aggregate across application
processes using the same user on one ClickHouse server; separate servers require
separate accounting. These are tracked-memory/admission limits, not CPU/IO or OS
memory reservations. The 1 GiB server and 1.5 GiB/2 CPU container limits remain.
The sum of class ceilings is 640 MiB; this is a configuration bound, not observed
simultaneous usage. Background merges, caches, untracked memory and allocator
retention still need headroom.

Provision only with installation authority and a verified absent/unowned-name
preflight for the four `roehub_*` users. Before `provisioned`, verify every prior
remote application uses an authenticated non-`default` identity and confirm that
identity can still access its required tables. A remote-`default` installation
is not eligible for this rollout: first perform a separately authorized identity
migration; an admin account alone does not preserve its application access. Prepare the class files through:

```sh
uv run python -m tools.installation.clickhouse_workloads \
  --private-directory .local_artifacts/platform-local/private/clickhouse-workloads
```

The mode-0700 directory contains mode-0600 credentials and private `users.xml`.
Existing passwords are reused; conflicting XML/symlinks/unsafe modes fail closed.
Verify the configured application `runtime_user` can read its own mode-0600 file;
provision under that authorized host identity rather than broadening permissions.
This command prepares files only. No credentials enter public generated outputs.
The server's generated bootstrap copies password-hash XML with the image's
ClickHouse ownership and mode 0600, then invokes the official entrypoint, which
drops privileges. Each application mounts only its class password; Compose clears
legacy shared credentials in its environment. Keep admin credentials outside
application mounts and do not put them in `/etc/roehub/roehub.env` or an arbitrary
`ROEHUB_ENV_FILE`. Remote passwordless `default` is denied; local bounded storage
health checks and separately authorized migrations/diagnostics are admin exclusions.
Catalog/funding administrative CLI commands also remain explicitly admin-only;
they fail if run with workload-required mode without their intended class.

Upgrade: prepare credentials → regenerate the owned bundle in `provisioned` mode
using the existing renderer → restart/recreate ClickHouse through the managed
controller → verify effective settings, grants, raw→canonical→stats and private
file ownership → drain application workers → regenerate `active` → activate a
proper image-bound candidate through the same lifecycle. Preserve selections,
leases, receipts, source obligations and all data. Do not use one-off `docker cp`.
The generic signed offline topology is not silently activated by this local flag.

Rollback: drain the class clients, restore the prior known application image and
`provisioned` renderer mode, retaining the same server identities/profiles and
private files. Restore the prior authorized application environment; do not
remove/rotate any account or drop data. A prior default-user remote deployment
is outside this rollout precondition; do not reopen anonymous remote access. Credential cleanup requires separate authorization.

Class profiles lock `async_insert=0` and `wait_for_async_insert=1`: INSERT work
completes synchronously inside the measured admission and memory boundary.

Diagnosis: native 202 is a class admission rejection and maps to existing
`storage_unavailable`; 241 maps to `storage_memory_pressure`. Neither implies
exchange absence. Collector receipts and Market Data checkpoints retain their
bounded backoff; client metadata initialization retries at most six times.
Application INSERTs are not blindly retried inside the gateway. Artifact failures
before publication use existing `retry_wait` / `transient_database`; after entry
they retain quarantine/reconciliation. Existing attempt ceilings remain in force.
Missing credentials/auth/grant errors fail visibly without default-user fallback.
Use query-log user/code aggregates for admitted/rejected work, per-class client
wait/retry counts and durable progress/oldest eligible jobs; check starvation
separately from an idle queue. Do not raise limits to hide pressure.

The reproducible mixed-load harness is `tools/qa/market_data_mixed_load.py`, extending
`market-data-resource-comparison/v1`. It requires `ROEHUB_S02_DISPOSABLE=1` and a
separate disposable server, never the user's installation. Supply private CH
admin credentials and class password-file references without printing them, then
run `--source <preserved-S01-source-or-candidate> --label baseline|candidate
--output <new-unique-file>`. The report fixes the six-run order, 60-second warmup,
300-second window, fixture/hash, caps and real catalog API route boundary.


## Rolling REST admission and recovery — S03 (2026-10-10)

This section describes source/config candidate behavior. It is not an installed
rollout receipt. S02 ClickHouse identities, limits, daily partitions, exact minute
verification and live-writer implementation remain the baseline.

### Participation, quotas and deadlines

All configured scheduler REST sources (candles, metadata, funding and bounds),
collector REST fill children, catchup CLIs and strategy history lookup use
`RequestsHttpClient.from_config`. They must use the **same PostgreSQL database**,
provider policy version and explicit `rest.limiter.egress_scope` for one shared
network egress. The example `local-installation-egress` is an installation label,
not IP discovery. Separate installations behind the same NAT need coordinated
scope/storage; unrelated applications on that IP are not controlled. Preserve
instrument selections and organization permissions. WebSocket traffic itself
uses no REST permits. No Redis service or permit daemon is introduced.

`shared_conservative` accurately names the policy. Legacy `autodetect` remains
parseable with explicit scope; it does not discover exchange limits. Current
conservative ceilings, before `safety_factor`, are:

| Provider dimension | Ceiling / endpoint charge |
|---|---|
| Binance spot weight | 1,200 / 60s; klines 2, exchangeInfo 20 |
| Binance futures weight | 1,200 / 60s; klines LIMIT <100: 1, <500: 2, ≤1000: 5, >1000: 10; exchangeInfo 1 |
| Binance per-product raw requests | 6,000 / 300s, cost 1 |
| Binance shared funding history/info | 500 / 300s, cost 1; also conservatively charged 1 futures weight |
| Bybit public markets combined | 300 / 5s and 50 / 1s, cost 1 |

The provider/scope concurrency ceiling is the minimum configured provider cap
(dev 4, test 2, prod 3); product weight dimensions share concurrency and cooldown.
All participating processes must have the same policy/config. Mixed policy
versions are unsupported; a per-process larger cap is not a safe rolling upgrade.
Validated supported origins/endpoints only; redirects and implicit proxy settings
are disabled. Explicit network-routing requirements need a separately validated
transport configuration, not an assumed environment proxy.

Primary policies checked on 2026-10-10: [Binance Spot REST](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md),
[Binance USD-M market data](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data),
[Bybit rate limits](https://bybit-exchange.github.io/docs/v5/rate-limit).
These local ceilings are not exchange capacity guarantees. Fixed quota windows
use PostgreSQL time; a backwards clock fails closed until the charged bucket
passes. Atomic reservation charges every active dimension and never refunds a
sent/uncertain request. `X-MBX-USED-WEIGHT-1M` only raises the matching product's
current charged weight; it cannot lower spend. Shared cooldown preserves
`Retry-After` (including a value longer than a default) and validated Bybit reset
headers. Coordination failure has no process-local fallback.

Each client owns an exclusive reusable pool of lean HTTP processes/sessions,
bounded by `ingestion.rest_concurrency_instruments` (dev 4, test 2, prod 4).
This local pool capacity is separate from the provider admission cap (prod 3).
Caller threads share this pool; idle threads cannot accumulate extra HTTP children.
The fixed leaf-module subprocess imports only stdlib and requests, avoiding a
copy of the scheduler import graph for every child. Private framed pipes carry
commands (64 KiB maximum) and JSON-decoded responses (32 MiB maximum); provider
bytes are never deserialized as pickle. HTTP bodies remain bounded at 16 MiB.
An absolute monotonic deadline covers pool wait, connection, body and internal
retry/backoff; a child watchdog also stops slow-trickle I/O if the parent is
blocked checking PG. A dedicated command reader detects parent-pipe EOF even
during HTTP. Timeout/lease loss kills and waits for the I/O
owner **before** permit completion. A sent request keeps its quota charge. No SQL
transaction spans HTTP; one permit connection per admitted request checks
ownership at most every 100ms, plus bounded short reservation/completion sessions.
The scheduler closes shared transports after all periodic tasks and REST work
have drained; collector/CLI/strategy owners close their clients in `finally`.

A quota/concurrency rejection releases the application worker into durable retry.
Requests keep their existing bounded retry policy and longer provider cooldown;
bounds stay queued with `next_retry_at`/sanitized `error_code` (minimum 2s).
An unknown permanent source response remains unavailable/failed as before.
For Bybit access-frequency `403`, automatic shared cooldown is at least 600s and
preserves a longer Retry-After. **Operator remediation must stop every participating
REST runtime, confirm all HTTP processes/sessions drained and wait at least ten
minutes after the last activity (or longer Retry-After) before restart.** Idle
sessions in other processes are not automatically closed by the shared cooldown;
full automatic cross-process ban remediation is not claimed.

### Coordinated upgrade and rollback

1. Use the managed controller above to stop/drain `market-data-scheduler` and
   `market-data-ws`; also stop participating CLI/strategy jobs under their owner.
   Confirm old processes and child HTTP I/O have exited. Do not overlap legacy
   clients (which bypass admission) with the new policy and claim a shared cap.
2. Apply the normal PostgreSQL migration phase `market-data-provider-admission-0034`
   through the installation's migration lifecycle. It adds quota/permit/coordinator
   tables and bounds retry columns, preserving selections, jobs and candles.
3. Regenerate configuration from owned sources. Dev/test and prod market-data
   templates carry the explicit scope; signed-topology outputs are generated by
   `uv run python tools/release/generate_runtime_topology.py --write` and verified
   with `--check`. This source-tree command does not activate an installation.
   Preserve existing prod cap 3/factor 0.7 and dev cap 4/factor 0.8.
4. Activate the image-bound candidate through the existing controller lifecycle,
   then verify migration version, common scope/config, fresh durable progress and
   provider admission. A build or local test alone is not activation evidence.

Rollback is coordinated: drain all new participants, restore their previous image
**and previous configuration** (old code cannot parse `shared_conservative`), then
restart through the same lifecycle. Retain additive PG tables/columns, spent quota,
cooldowns and uncertain ownership records. Do not drop data or clear quota to make
an old/new mixed deployment appear healthy. Runtime rollback was not exercised on
the user's installation by the S03 source stage.

### Stranded owner recovery: confirmed drain, exact tokens

Connection loss is not evidence of stopped I/O. Automatic session lock release,
PID absence in another namespace and elapsed time are not reclamation authority.
On a crash, permit rows and coordinator guard may intentionally remain charged.
Use this sequence only under authorized installation recovery:

1. Stop admission in **all** participating processes for the affected egress scope
   and coordinator. Identify actual owner host/process namespaces from the permit
   rows and managed runtime identity. Confirm process and descendant termination;
   if an owner/host cannot be accounted for, retain the rows and remain stopped.
2. Wait for any already-issued ClickHouse operation to drain as well. A coordinator
   guards writes in addition to HTTP. Verify no old worker or in-flight write can
   continue. Keep the database reachable; do not substitute local bookkeeping.
3. Read bounded metadata through an authorized PG connection (never print a DSN):

   ```sql
   SELECT token, provider, scope, owner_pid, owner_host, acquired_at
   FROM market_data_provider_permits ORDER BY acquired_at LIMIT 100;
   SELECT lock_key, token, acquired_at
   FROM market_data_coordinator_owners ORDER BY lock_key;
   ```

   Record the exact confirmed owners/tokens privately. Delete only those tokens
   after the preceding proof, in a transaction while all participants remain
   stopped. Bind parameters through the approved PG client; this is the predicate,
   not a wildcard cleanup command:

   ```sql
   DELETE FROM market_data_provider_permits
   WHERE token = :confirmed_token AND provider = :provider AND scope = :scope;
   DELETE FROM market_data_coordinator_owners
   WHERE lock_key = :confirmed_lock_key AND token = :confirmed_coordinator_token;
   ```

   Check affected row counts. Never truncate these tables or delete by age alone.
   Do **not** delete `market_data_provider_windows` or reset shared cooldowns.
4. Restart one coordinator through the managed lifecycle. Its pre-dispatch orphan
   recovery reconciles abandoned running jobs before dispatch. Resume other
   participants only with matching scope/config and observed durable progress.
   Bounds interrupted as running become claimable under their existing recovery
   delay after the old coordinator is proven drained.

Unknown commit outcomes are fail-closed: inspect exact token state after PG
recovers; never replay a quota refund or assume a failed response means no commit.
Real-PG tests cover session loss, blocked checks, owner death, stranded capacity
and exact-token cleanup **after** child join. They do not authorize cleanup of an
unobserved remote owner.

### Measurement boundary

`tools/qa/market_data_ingestion_comparison.py` extends the existing comparison
schema with actual PG queue/CH writes and deterministic loopback provider HTTP.
Use only a separately provisioned disposable target with the explicit opt-in and
new output filenames. The S03 report fixes source snapshots, 102-coordinate
sparse/contiguous seed, four workers, daily windows, provider quotas, two processes,
60s warmup/300s steady window and full finite drain. CH backfill queries are
serialized equally on both sides to isolate this experiment from S02 query-cap
rejection; separate real-CH 202/241 tests prove pressure recovery.

Observe completed windows, eligible-slot idle time, provider attempts/connections,
rejections, oldest work/backlog, durable retries, PG transactions/connections,
CH query rows/bytes/peak tracked memory, allocator/RSS, cgroup usage and post-drain
plateau. One-second samples can miss short peaks; cgroup cumulative peak is not a
per-run peak when the server is reused. Emulator collector latency is HTTP-only,
not receive-to-canonical or API latency. S02 mixed-load tail evidence remains its
own proof boundary; S03 does not re-establish it or claim a real-provider capacity.

S03's final six-run M4 comparison observed median completion 733.31→399.60s,
steady throughput +83.3% and eligible-slot idle time −97.5%. This is a measured
throughput/latency tradeoff: collector HTTP p99 rose 3.04→6.65s; all 120 collector
requests still finished before backfill in every candidate run, with 19–21
completions per steady minute and sampled backlog at most three. No starvation
was observed in this fixture; neither FIFO nor a general cross-runtime wait bound
is guaranteed. PostgreSQL transactions rose about sixfold (10,885→65,380), or
about elevenfold per elapsed second (14.8→163.6). Size PG capacity accordingly;
per-run PG CPU was not measured, so this is not evidence of lower total system CPU.
Host process-family peak RSS rose about 35%. A separate Linux test under the
actual scheduler 384MiB/1CPU cap bounded the repaired HTTP pool below 230MiB
for 26 callers, but did not include the full ClickHouse write working set.
See the S03 report for ranges, failures, proof boundaries and upgrade checks.
Numerical performance acceptance: `not assessed — numerical threshold not specified`.

## Asynchronous batched stream recovery — S04 (2026-10-11)

This is a source behavior contract; installed activation is separate. The
collector uses one lifecycle-owned async PostgreSQL session for both its advisory
lease and all recovery/health transactions. A 2 ms admission coalescer commits at
most 256 receipts together. Existing keys, tokens, failed/rearm states and schema
remain unchanged. The warmed, killable CH writer and S03 provider permit lifecycle
are retained. This does not optimize S03 quota-admission transactions.

Reserve capacity before persistence; acceptance requires PG commit. The buffer
accounts for waiting producers, in-flight admission, buffered rows and retries:
`2 * max_buffer_rows` total retained rows, at most `max_buffer_rows` admitted to
persistence/write, and 64 MiB accounted bytes. Byte accounting is a conservative
object/metadata reservation, not an OS RSS limit. The metadata coalescer separately
caps 20 000 tasks, 256 per transaction, symbol/reason at 128 characters. The REST
queue caps pending+admitting keys and never holds its condition lock across PG.
On capacity exhaustion, an unaccepted arrival relies on reconnect/tail recovery;
never clear receipts to make capacity available.

A 3 s PG deadline closes the socket **before** cancelling, because rollback can
otherwise wait forever for a lost reply. Unknown commit poisons the session and
requires collector restart. Do not assume the transaction rolled back or delete
its key. After lease reacquisition, stable keys return committed receipts with
their original tokens. Per-caller cancellation never cancels a shared batch.
Before persistence, a capacity waiter is cancellable without a receipt; afterward,
the owned admission completes and either the buffer drains or metadata remains
for restart. CH acknowledgement and receipt completion are separate crash
boundaries; only verified actual candles satisfy a received `ws_recovery` range.

Upgrade and rollback: stop intake, drain owned admissions and the writer, retain
all pending/failed receipts, release the collector lease last, then switch source
through the managed installation lifecycle. No new migration/config key is needed
for S04; S02/S03 prerequisites still apply. Old source reads the same journal
shape. Do not run two collectors, rearm failures automatically, reclaim tokens
by TTL, or activate source using one-off copies into an installed container.

Diagnose growing backlog by separating capacity wait, PG deadline/session failure,
CH retry and failed receipts. Inspect canonical progress and oldest pending work;
a connected socket is not delivery. Stop/restart a poisoned collector through the
managed lifecycle without deleting journal data. Retained failed receipts require
root-cause repair before the existing explicit rearm operation.

`tools/qa/market_data_recovery_comparison.py` extends the comparison schema with
finite, accelerated synchronized bursts at 4 and 100 streams, fixed scheduling,
real PG/CH, one warmed writer, full drain and canonical content parity. It requires
`ROEHUB_S04_DISPOSABLE=1`, the named dedicated loopback targets and a new output
path for every attempt. It measures scheduled/receive/receipt-to-writer-ACK;
synchronous CH/MV settings and final canonical verification bound that ACK proof.
It is not a service steady-state, provider scale or API-latency measurement.
One-second RSS samples miss short peaks; post-drain observations are short and do
not exclude leaks. The final S04 report owns measured results and limitations.

## Local activation follow-up — 2026-10-11

The local owner authorized S01–S04 activation and a local-only commit, including
necessary pre-existing implementation dependencies. See
[the activation report](market-data-efficiency-local-activation-20261011.md)
for the actual image, preserved settings/data, observed failures and proof limits.
The earlier stage reports remain source-stage evidence, not rollout receipts.

For the official ClickHouse image configured with a non-default `CLICKHOUSE_USER`,
the entrypoint removes `default`. Verify that contract and the actual absent
`default` user before provisioning with `--default-user-disabled`. In that mode,
workload XML omits `default`; a networks-only fragment would otherwise recreate
an incomplete user and prevent startup (code 347). This flag is not safe for an
installation that retains a remotely accessible `default` identity. Without the
flag, the existing localhost restriction is unchanged. Existing XML is never
silently rewritten: an authorized correction must verify its old generated bytes,
preserve the same four passwords, back up the private XML and retain mode 0600.
After restart, verify `proof` plus the four workload users, absence of `default`,
remote-default denial and successful class reads/writes.

The managed artifact publisher's reference/catalog reader explicitly selects
`artifacts`; API callers retain the `interactive` default. Do not mount the API
password into the publisher to work around a composition error.
