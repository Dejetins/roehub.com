# PostgreSQL-only storage — complete migration plan v1

Date: 2026-10-13 (Europe/Moscow). Engineering artifacts: English; owner reports: Russian.
Status: complete planning package; PostgreSQL-only direction accepted by the owner;
implementation design proposed, no migration stage executed by authoring this pack.

## 1. Authority, deliverables and limits

Owner decision: remove ClickHouse entirely and use PostgreSQL for its capabilities.
Existing development market data may be discarded and downloaded again. Prepare a
complete plan, executor prompts and operation journal, covering documentation,
installation, query implementation, verification and tests. This request authorizes
planning, not starting migration, deleting data, publishing Git or deploying now.
Submitting a stage prompt later selects its bounded implementation; selecting the
whole pack later can authorize sequential continuation. No Goal or new chats are
implicitly created. Binance and Bybit remain exchange providers; no provider removal
is inferred from the transcribed phrase “от Isbylance”.

- `plan_doc`: [implementation-plan.md](implementation-plan.md).
- `prompt_pack_dir`: `.codex/agents/generated/postgresql-only-storage-v1/`.
- `stage_ledger`: [iteration-ledger.md](iteration-ledger.md), sole mutable execution record.
- Execution mode: `manual_sequential`, S01 → S02 → S03 → S04 → S05 → S06 → S07 → S08 → S09 → S10.
- Initial state is draft, all stages pending; only S01 is technically entry-eligible.
  Eligibility is not authorization to run during this planning turn.
- Reuse `ledger_update.py` and its disposable tests byte-for-byte from the existing
  resource-efficiency pack. No new execution engine, daemon, role system or lock design.
  The copied updater binds this directory, stable directory flock, executor identity,
  receipt hashes and fsync/atomic replace. Never call another pack's updater to change this ledger.
- No Git publication, branch/worktree/stash, live trade submission, production/retired
  host access, new database extension/service or Redis removal. Preserve unrelated dirty work.
- Development-data loss permission removes the requirement for a ClickHouse→PostgreSQL
  production importer or dual-write period. Preserve accounts, secrets, settings,
  instrument selections and strategy definitions. Reset only derived market history,
  coverage/recovery and artifact state whose consistency depends on that history.
  No blanket `docker compose down -v`, whole-PostgreSQL reset or deletion of unrelated volumes.
- Do not change applied PostgreSQL migration bytes/checksums. Append versioned migrations;
  choose the next free identifier at execution time rather than hardcoding today's number.

## 2. Evidence and present architecture

The starting point is the live dirty working tree, not HEAD alone. Refresh touched files
before each stage and distinguish owner changes from foreign work. Inventory was collected
using tracked and nonignored untracked file paths, content and SQL dialect searches.
The Appendix lists the initial matches; S01 refreshes and classifies every match. Paths
without the brand in their contents (funding DDL, schemas, callers) must also be followed.

Current source-of-truth entrypoints:

- `AGENTS.md`, `docs/architecture/README.md`, `docs/architecture/project-map/AGENT_GUIDE.md`.
- `src/trading/contexts/market_data/application/ports/stores/` and the ClickHouse adapters:
  raw/ref writers, canonical readers/index, coverage, range verifier, readiness and funding.
- `migrations/clickhouse/manifest.json` has three migrations: candle/reference DDL,
  funding DDL, and source-absence DDL. Funding remains in scope even if a local data set is empty.
- `migrations/postgres/0024_market_data_full_history_v1.sql` stores exchange history
  start metadata, NOT a certificate that all stored history is consistent through time X.
- `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py`
  uses an indexed, bounded keyset source observation; do not regress it to canonical scans.
- `src/trading/contexts/market_data/application/dto/live_tail_repair.py` and
  `application/services/closed_candle_tail_provider.py` embed a `clickhouse` source enum,
  timeout/circuit names and Redis→database→REST source chain; this crosses a contract.
- `apps/migrations/storage.py`, `src/trading/platform/config/{installation,runtime_topology,operational_topology}.py`,
  `tools/installation/local_platform.py`, `tools/release/`, `src/trading/contexts/operations/backup_contracts.py`.
- `docs/runbooks/{offline-release-installation,storage-bootstrap-and-migrations,installation-backup-restore-upgrade}.md`.
- Accepted UI baseline is `apps/platform-web`; `apps/navigator-web` remains a candidate.
  Update affected supported clients/tests without promoting a candidate by accident.

Measured evidence from `.local_artifacts/pg-comparison-20261013/`: `report.html`,
`summary.json`, `measurement-contract.json`, `snapshot.json`, `measured-harness.py`,
`verification.json`. Exact copy: 15,130,115 canonical candles, 102 instruments,
89 months, all 17 fields, cutoff 2026-10-12 exclusive, 7,625 source-absence records.
Twelve retained 60-second mixed runs, 2 CPU / 4 GiB each, 4,068 operations, no errors
or scheduled-interval overruns; raw is normalized, not a complete provider payload archive.
PostgreSQL relations 6.28 GiB vs ClickHouse 1.13 GiB. PG was faster on small operations;
ClickHouse on bulk histories. Client materialization used substantial memory. These facts
support the decision but do not prove production adapters, funding, full HTTP behavior,
long-running maintenance or power-loss recovery. Do not ship the prototype as production code.
If local evidence is unavailable, regenerate equivalent evidence; do not ask the owner
to reconstruct ephemeral files or manufacture historical results.

Observed drift to resolve explicitly: the installation storage runbook certifies CH 24.8
while the measured local replica used 25.8.33.6; historical/bootstrap prose predates current
migrations. Removing CH is the accepted target, not justification to claim the old docs
already describe PostgreSQL-only runtime. Recent changes to logging, admission and topology
are dirty; incorporate their contracts, not an older committed snapshot.

## 3. Target architecture and decisions

### D1 — one durable SQL engine, existing capability boundaries

PostgreSQL 16.x, already an installation dependency, becomes the durable source of truth
for market reference data, candle observations, canonical candles, funding and coverage,
in addition to existing accounts/jobs/recovery. Redis remains bounded hot cache/transport;
NPY/mmap/CAS artifacts remain the backtest compute path. Domain/application code depends
on its ports; PostgreSQL adapters and composition roots own SQL/drivers. No TimescaleDB,
new queue, broker or second database engine. Keep current approach is no longer selected:
the owner chose removal, and the measured small-operation/storage tradeoff is accepted.

Prefer tables in a dedicated `market_data` schema of the existing PostgreSQL database;
existing public control tables need not move. Separate migration and application privileges;
never require the app to be superuser. Qualify identifiers and parameterize values.
Embedded and external PostgreSQL profiles must both provision/check the same capabilities.

### D2 — physical schema and preservation of semantics

Planned names are target contracts, not files/tables claimed to exist today.

| Current capability | PostgreSQL target | Required constraints and proof |
|---|---|---|
| `ref_market`, `ref_instruments` | Same purpose under `market_data` | Stable market IDs 1–4, enabled/tradable semantics, PK market and market/symbol, version-aware upsert; search/paging |
| `raw_binance_klines_1m`, `raw_bybit_klines_1m` | Provider-specific typed observation tables, monthly range partitions | Preserve every currently persisted provider field/nullability and provenance; no unbounded JSON duplication |
| `canonical_candles_1m` | Monthly RANGE(ts_open), PK(market_id,symbol,ts_open) | Exactly one closed minute, all 17 canonical fields, UTC half-open ranges, float64/null behavior unchanged |
| aggregate bitmap stats | `candle_coverage_days` + compact `candle_coverage_summary` | PK market/symbol/day; actual `bit(1440)`, source-absent `bit(1440)`, counts and actual bounds, durable content revision; rebuildable and generation-bound |
| `source_absent_candles_1m` | Auditable accepted-absence observations + daily mask | Proven fully consumed successful provider response only, actual candle takes precedence |
| history start metadata | Existing `market_data_history_bounds` | Keep exchange availability discovery separate from observed and verified storage coverage |
| verified historical prefix | `candle_history_verification` | Origin, exclusive verified_until, data generation and verification policy version; explicit invalidation |
| funding universe/raw/canonical | PostgreSQL equivalents for ALL funding tables | Version-aware records, nullable mark price/category, interval/source semantics, exact funding timestamps; canonical PK includes funding_time |
| artifact source coordinate index | PG observation port extended with durable generation + content revision | Keyset page ≤50, exact actual bounds before exclusive cutoff; never include source absences; unchanged bounds must not hide corrected values |

Monthly raw/canonical/funding partitions are pre-created for requested import ranges and
current/next month by a bounded maintenance/migration capability, with serialized DDL.
No per-candle DDL and no silent default-partition accumulation. Rollover and old-history
imports work across month/year boundaries. Use BIGINT where unsigned CH values require
it and range checks for positive IDs/counts; preserve Float64 as double precision rather
than silently changing financial arithmetic to numeric. Avoid redundant covering indexes
until EXPLAIN proves their benefit. Partition pruning requires direct UTC timestamp bounds.

### D3 — atomic write receipt, batching and deterministic replay

One bounded adapter transaction: stage with COPY, persist typed raw observations,
deduplicate incoming canonical keys, UPSERT canonical, update affected coverage days and
summary/dirty state, commit. Success returns only after durable commit; no successful
recovery cursor or Redis publication before the durable receipt. Keep existing buffered
writer limits, flush cadence/backpressure and asynchronous recovery batching semantics.
Do not open a new DB connection or transaction for every candle.

Canonical version precedence: newer ingested_at wins; normalize to the established
millisecond precision. Equal version + equal contents is a no-op. Equal version + different
contents uses a documented stable tie-break derived from normalized persisted contents
(not process arrival order, random ingest UUID or source priority). Persist the tie-break
only if needed to make retries/order permutations stable; test it. This tightens previously
ambiguous equal-version behavior and must be explicit in the contract. Deduplicate multiple
same-key rows inside one staging batch before ON CONFLICT; PostgreSQL cannot update a target
row twice in one such statement. Raw observations can contain retries as today; do not let
those inflate canonical counts, summary counts, artifact revisions or verified coverage.
Retries of unknown commit outcomes must converge to the same canonical/coverage state.

Acquire row/day locks in stable market/symbol/day order. Compute actual count deltas under
those locks (bitset cardinality before/after), not from attempted row count. Summary updates
must not lose concurrent increments. Publish artifact invalidation/version hints for value
corrections even when minute membership/count is unchanged. Make this recoverability concrete: persist a monotonic content revision per instrument
(or per instrument/day aggregated into its summary) in the same transaction as a change
to materialization-relevant canonical values or actual-minute membership. Identical replay
and ingestion-metadata-only changes do not increment it. Bind revision to source generation.
Extend StoredSourceObservation, persisted source targets and reconciliation comparison to
carry that durable revision; current first/last + content_unverified is insufficient.
The bounded sweep must detect a corrected old OHLCV value even if first/last/count stay
unchanged and the process dies before publishing its hint. Conservative invalidation for
an unaffected cutoff is acceptable when documented; missed relevant changes are not.
Update content-derived cache/recipe invalidation without embedding the storage engine in
public artifact identity. Hints remain optional accelerators, not the durability mechanism;
no silent failure of already committed writes.

A DB failure may simultaneously affect candles and recovery/audit metadata now that they
share a server: propagate failed receipt, preserve existing durable cursors, bound buffers,
mark degraded, and reconcile on recovery. Never call an unpersisted repair 'completed'.
Classify SQLSTATE retries (serialization/deadlock/transient disconnect) with bounded backoff
and idempotent transaction replay. Permission/schema errors fail visibly; never swallow all
exceptions or replay external order submissions. Keep DSNs, credentials and provider
payloads out of logs/evidence.

### D4 — exact coverage, verified frontier and bounded backfill

Three distinct facts: (1) actual stored minutes, (2) source-confirmed absent minutes,
(3) unverified/missing minutes. A source absence is not a candle, latest candle, artifact
coordinate or actual-count increment. An arrived late candle clears/overrides its absence
in the same transaction and updates summaries. Failed HTTP, 451/429/timeouts, partial or
unconsumed pages, stale lease and storage failure never produce accepted absence.

Verified frontier means every minute in [origin, verified_until) is either durable actual
or currently valid accepted source absence for that generation/policy. It is not max(ts),
last seen WS time, provider first-open time or a claim that the exchange cannot revise data.
Advance only across contiguous proven ranges, in a transaction protected by current lease
and generation checks. Noncontiguous completed jobs must not leap over a gap.

Normal gap search consults the compact summary/frontier then range-bounded daily masks;
it never scans all canonical history or expands one SQL/Python row per historical minute.
Missing days are gaps, no mask row does not mean complete. Locate the first incomplete day,
then its first missing minute; retain the current bounded per-verification range (≤7 days).
A frontier skips verified old prefixes; corrections, deletion, restore, reset or index rebuild
invalidate affected frontier/generation before claiming completeness. An administrative
historical delete must clear/rebuild actual masks and exact bounds, not keep OR-only truth.

Negative availability has a policy version and observation provenance. Keep old confirmed
absence valid under that policy, but invalidate on provider-history refresh/policy change
or explicit repair. Recent tail retry does not permanently suppress transient provider lag.
Do not introduce periodic full-history rescans to 'keep the watermark fresh'. Bounded
repair/rebuild is explicit, resumable and visible in health/events.

### D5 — SQL query catalogue and bounded resource use

Every replacement query gets an input/cardinality/plan record in the S04 report (updated by
S09): port/caller, SQL identifier, predicate/order, expected rows, indexes/partitions touched,
statement/lock timeout, concurrency class, EXPLAIN ANALYZE BUFFERS evidence and result parity.
Do not mechanically translate FINAL, argMax, bitmap states or MergeTree syntax.

| Query family | Required access path | Forbidden regression |
|---|---|---|
| Latest N / live-tail | PK range + ordered LIMIT; bound timestamps where known, summary to locate latest occupied range | Whole-table ORDER BY/MAX scan each poll |
| Candle range / arrays | Half-open UTC range, pruning, ordered server-side cursor/COPY or bounded binary batches | fetchall of multi-year history followed by duplicate tuple/array copies |
| Distinct timestamps / daily counts / before bounds | Actual daily masks and summary; exact partial first/last day | Counting attempted writes or source absences; per-instrument full scan |
| Catalog / instrument coverage | Batched keys, compact summary, keyset/search indexes | N+1 history queries, offset over huge histories, global repeated COUNT(DISTINCT) |
| Gap / source absence counts | Range-bounded day masks + frontier, actual precedence | Minute-level generate_series over multi-year history |
| Source reconciliation | Bounded coordinate page and clipped masks, exact `before` | Full canonical scan fallback when index is unavailable |
| Funding reads | market/symbol/funding_time ranges, canonical PK; interval metadata | Assuming fixed eight-hour funding or ignoring nullable provider fields |
| Verification/rebuild | Explicit bounded range and generation/lease checks | Hidden unbounded repair on UI or periodic scheduler hot path |

Do not promise index-only scans in every case or reject a justified sequential scan of a
small summary. Prove rows/buffers/partition scaling, not just plan node names. Full-range
artifact reads inherently scale with returned data; stream directly toward artifact output
where contracts allow. Account for client RSS as well as PostgreSQL memory.

Use the installed psycopg3 integration; add a bounded supported pool only after checking
existing gateway lifecycle and dependency policy. Set total installation connection budgets
across processes/classes, leaving capacity for identity/jobs/admin/recovery and migrations.
Per-process limits are not a global budget. Admission must bound concurrent artifact builds
and history imports; queues expose wait/backlog. Session settings must reset on pool return.
Choose shared_buffers/work_mem/maintenance/autovacuum and statement/lock timeouts from a
measured envelope, not by granting every request the whole container. Start evaluation with
the measured 2 CPU / 4 GiB envelope; optional larger ceilings are owner-authorized but rerun
comparable evidence and retain failures. No unlimited max_connections or disabled durability.

### D6 — complete product removal and compatibility

No final feature flag, old driver import, CH service/image/exporter, CH DSN/secret slot,
clickhouse workload profile, executable migration, SQL adapter, fallback query or runnable
maintenance script remains in shipped/runtime/test tooling. Delete or port stale notebooks
and RL scripts too. Retire optimize-partition operations; do not create meaningless PG
OPTIMIZE analogues. Preserve historical evidence without executable imports in product paths.

| Surface / consumers | Before → target | Classification and transition |
|---|---|---|
| Candle public DTOs, identities, UTC/NaN/order contracts | Same data semantics behind PG adapter | Target `none`; must prove consumer parity in S05/S08; not yet verified |
| write receipt and equal-version precedence | MV/eventual dedup → transactional canonical/index, stable tie | `compatible-change` for successful-write consumers; explicit internal behavior strengthening, tests required |
| persisted market schemas and restart state | CH tables/index + PG metadata → PG-only market schema | `breaking-change`; development reset permitted, no old binary against new state |
| internal source observation and target revision | first/last only → generation + durable content revision | `breaking-change` for the internal port/persisted comparison shape; update all callers and invalidate stale source targets in the same rollout |
| repair source enum, circuit/timeout names, metrics/UI/errors | clickhouse → canonical_store or postgresql as appropriate | `breaking-change`; all producers/validators/UI/alerts together; migrate affected persisted source values with append-only migration |
| installation config, status and service identity | CH required in trading/ml → PostgreSQL-only | `breaking-change`; explicit schema/version transition; reject stale keys clearly, never silently ignore |
| encrypted backup/capture owners and manifests | CH owner required → no CH owner | `breaking-change`; version manifest/policy/capture schemas consistently; reject old format before writes, no silent skip of backup integrity |
| artifact cache/source state and pending work | old history bounds/generation → reset source generation | `breaking-change` for reuse of stale coverage/inputs; reconcile/invalidate, never report stale artifacts as fresh |
| unknown external scripts/config clients | No supported inventory yet | `unknown`; S01/S05 enumerate repository callers and document supported version boundary |

Single-version local maintenance cutover is selected, not rolling old/new workers or
permanent dual write. New installers never contact CH. A minimal explicit old-config
rejection/migration diagnostic may mention its former name but must not connect to it.
Old backup restore is not promised; recovery uses matching old tooling in isolation or
new-format PG backup, not a partial mixed-version restore. Do not repurpose unchanged
schema identifiers to encode incompatible owner sets.

### D7 — logging, monitoring and user-visible events

Preserve recent connection transition/event journal improvements. Structured events for
DB unavailable/recovered, pool exhaustion/admission wait, retry exhausted, backfill
started/progress-summary/completed/failed, verified-prefix invalidation/rebuild and
partition maintenance. Correlation/job/instrument IDs and sanitized SQLSTATE, no raw SQL
parameters/DSNs. Coalesce/rate-limit progress; no per-candle or per-query informational log.
Connect recovery events to the existing user event journal and bounded pagination.

Replace CH health/exporter/action/runbook identities with meaningful PG metrics: active
connections by class, pool wait, transaction/statement failures, lock wait, long transactions,
WAL/checkpoint/autovacuum state, partition readiness, ingestion lag, coverage/recovery backlog
and artifact build queue. Do not reintroduce retired observability services; use current
operational topology and optional supported integrations only. Apply language/localization
and data-loading contracts on affected UI; no redesign.

## 4. Documentation workstream (S01, then owned updates through S10)

S01 records PostgreSQL target amendments without claiming the still-running install is
already migrated. Each later stage synchronizes its changed contract. S07 performs a full
residual audit; S10 updates actual local delivery facts only after measured proof.

| Current canonical surface | Required change / owner |
|---|---|
| `docs/architecture/market_data/market-data-application-ports.md`, `market-data-real-adapters-clickhouse-parquet.md` | Transactional write/read/index contracts; replace adapter document with PG-named equivalent, repair inbound links; S01/S03/S04 |
| All reference API/sync, WS worker, REST catchup, backfill, runtime-invariant documents under `docs/architecture/market_data/` | PG schema, bounded queries, receipt/retry/lease/frontier semantics; S01–S05 |
| live-feed/live-tail docs; strategy live-runner/runtime/milestone docs | Redis→canonical PG→REST, renamed source metrics/DTOs, failure semantics; S05 |
| backtest artifact runtime and funding policy; indicators CandleFeed/ports; ML/RL source docs | Storage-independent arrays, exact coordinates, funding parity, streaming exports; S05 |
| `docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md`, market-data activation/selection contract | Stores, budgets, bootstrap, health, source state, profile matrix; S01/S06 |
| `docs/runbooks/storage-bootstrap-and-migrations.md`, `offline-release-installation.md`, `installation-backup-restore-upgrade.md` | PG-only embedded/external bootstrap, versioned backup/restore, interrupted migration, cleanup/rebuild; S06 |
| `docs/runbooks/navigator-local-market-data-recovery.md`, `market-data-live-tail-repair.md`, `market-data-redis-streams.md` | Current PG diagnosis/recovery commands, current local target discovery; S06/S10 |
| `docs/runbooks/ops/database.clickhouse-degraded.yaml`, localized YAML, `runbooks.json`, generated runbooks | Replace with PG operation/runbook and links; regenerate via owner tool; S06/S07 |
| `docs/architecture/apps/cli/cli-backfill-1m.md`, API/UI error docs, identity/operations pages mentioning store ownership | Correct source/destination, storage errors, backup ownership; S05/S06 |
| README, project map source, docs index, schemas, examples, release notices/SBOM | Generate from source; remove active CH dependencies; S06/S07 |
| Historical stage reports/accepted ledgers, old benchmark records, `docs/runbooks/legacy/`, `.codex/delivery/evidence/` | Preserve factual history/bytes and mark/link supersession at existing navigation boundary; never rewrite “then CH” as “then PG” |

Appendix entries are candidates, not a blanket permission to rewrite historical records.
Every residual match must have a classified disposition: port/update, delete executable,
regenerate, explicit immutable history, or narrow compatibility diagnostic. No blanket
allowlist for whole active directories. The new migration plan itself naturally mentions
its source engine; this is explanatory history, not a runtime dependency.

## 5. Installation transition, reset and recovery

1. S02–S09 implement and prove on disposable labeled PostgreSQL instances/volumes.
   Existing development install remains unchanged until separately selected S10.
2. S10 resolves the then-current local target and current runbook, exact volume/service
   identities and dirty revision; no retired host. Preserve config/secrets/accounts via
   existing backup mechanisms and private locations, redact evidence. A targeted market
   reset is permitted by the owner; retaining CH data or importing it is optional, not a gate.
3. Quiesce market writers, schedulers and artifact publishers; drain or persist buffers
   according to existing semantics. Prevent old workers from restarting against new state.
4. Apply additive PG migrations and new config/bootstrap. Start a new market-data generation:
   invalidate copied catalog snapshots, verified prefixes, source/coverage caches and stale
   history jobs/cursors. Preserve selections/settings. Ensure orphaned work is terminal or
   restartable; rebuild artifact source obligations/recipes or mark old inputs unavailable.
   Never keep old 'complete history' flags for an empty PG candle table.
5. Start the PG-only stack, seed/sync reference and funding data, re-download selected
   instruments with provider admission/rate limits. Existing files may speed up rebuild
   after validation; absence of an importer must not block clean refill.
6. Prove APIs/UI, ordinary backtests/artifacts, real provider history loading and arrival
   of consecutive closed candles. Provider region restrictions are provider failures, not
   successful migration proof: preserve evidence, continue local/replay tests, leave live
   readiness unaccepted until the necessary reachable provider test succeeds.
7. Verify no CH process/service/socket/image requirement in active deployment, no pending
   CH backup/migration/health action. Remove only identified target CH service/config;
   unused source volumes may remain inert and are not a runtime dependency. Deleting them
   is optional and must be exact-scoped; never delete benchmark evidence implicitly.
8. Capture final health, volume/data disposition, fresh-install matrix and independent
   review before declaring local cutover accepted. No push/merge is part of this pack.

Point of irreversibility: targeted market-data reset or deletion of old store without a
retained copy. Owner accepts re-download cost; do not promise a data-preserving rollback.
Before that point restore previous matching config/binaries if needed. Afterwards forward
repair/rebuild is the default; reverting requires restoring matching old state or rebuilding
that old installation explicitly. User/secret/config state must remain recoverable separately.

## 6. Proof matrix and performance gates

| Boundary | Required evidence |
|---|---|
| Schema lifecycle | Real PG16 clean bootstrap twice; interrupted/resumed migration; checksum drift rejection; embedded/external profiles; month rollover and historical partition creation |
| Transactional ingestion | Two exchanges/spot+futures, nulls/UUID/UTC/float64; duplicates inside/across batches; stale/equal versions; concurrent WS+REST for same minute; injected rollback/unknown commit; lost post-commit hint plus same-minute correction with unchanged bounds; exact raw/canonical/index receipts |
| Coverage/frontier | Missing day, disjoint completed ranges, true absence vs failed fetch, late actual replacing absence, generation/lease races, bounded index repair, reset/delete/restore invalidation; boundary `before` inside minute and epoch zero |
| Funding and consumers | Reference/funding import and interval semantics, strict monotonic arrays, artifact recipe/identity/cache invalidation, NPY parity, actual backtest result parity on fixed inputs, strategy tail and Redis ACK ordering |
| Single-PG mixed workload | Include identity/API/job leases/recovery writes alongside market import, catalog, gap and artifact work; prevents understating contention from collapsing two stores |
| DB resource protection | Total connection/pool limits across processes, wait/backpressure, cancellation, no lost writes, reserved control-plane headroom; server + client memory/CPU and disk/WAL separately |
| Failure and operations | Controlled PG termination/restart during writes, disconnect/timeout, saturated pool, lock contention, interrupted partition maintenance and constrained-disk failure on disposable environment; no production chaos |
| Backup/restore | New-format full backup, quiesce/digests/owners, clean restore, old-format fail-before-write, schema/index/frontier correctness after restore; no secret material in receipts |
| UI/runtime | Real browser data/history/progress/event journal/monitoring/backtest smoke on available supported client with disposable data; screens and failed-network behavior, no fake pass from unit tests |
| Removal | Static source/dependency/config/dialect audit plus boot/test with CH unreachable and package absent; all profiles, CLIs, workers, release verification and optional external PG |

S09 must use production adapters, full ≥15,130,115-row /102-instrument representative copy
(or a larger equivalent reproducible set), not an empty/tiny benchmark. Retain fixed input
hashes, exact hardware/config/revision, ordered cold-vs-warm policy, counts/errors and client
RSS. Repeat normal and stress runs at least 3 times ×5 minutes, with at least 30 large-read
observations per regime in total; then ≥60-minute simultaneous ingestion/query/recovery/
artifact soak crossing multiple checkpoints and observing autovacuum. Add explicit workload
on shared preexisting PG application metadata. Stream reads in production adapters; do not
carry the prototype's fetchall client memory behavior into acceptance.

Proposed engineering guardrails (not a preexisting product SLA), to be confirmed or refined
with documented evidence in S01 before implementation: on the same 2 CPU/4 GiB baseline,
p95 latest/catalog/gap ≤100 ms, p95 stream write ≤100 ms, p95 1,000-row backfill ≤250 ms;
year-history artifact input ≤2 s median and full-instrument large read ≤10 s p95 under the
defined mix. No storage errors, OOM, dropped successful receipts, divergent results, unbounded
queue/pool growth or unexplained monotonic memory/disk growth. Completed throughput must meet
the fixed offered cadence (normal 5 stream rows/s, stress 100, plus 1,000 history rows/2 s).
Application HTTP latency is measured separately; it is not interchangeable with adapter time.

Do not weaken criteria silently to pass. If a guardrail fails, profile plans/I/O/locks/client
materialization, tune within bounded resources and rerun comparable evidence; record legitimate
scope/hardware changes explicitly. Disk raw/index/WAL/retention costs must be measured, not
promised to match CH compression. Raw retention remains explicit and non-destructive by
default; a future retention policy is not silently introduced by this migration.

Repository commands are selected from current `pyproject.toml`, `.github/workflows/ci.yml`
and `tools/ci/route_changes.py`. Known entrypoints to recheck at execution:
`uv run ruff check .`, `uv run pyright`, routed `uv run --no-sync pytest -q -ra ...`,
`python -m tools.docs.generate_docs_index --check`,
`python -m tools.docs.generate_project_map --check`,
`python -m tools.docs.generate_runbooks --check`,
`python -m tools.artifacts.generate_schemas --check`,
`python -m tools.backup.generate_schemas --check`,
`python tools/release/generate_runtime_topology.py --check`.
Runtime verifiers include `apps.migrations.verify_storage_runtime` and relevant existing
release/backup verifier modules. Inspect their current flags/target guards before execution;
only disposable environments for destructive drills. Future new tests use the existing
`tests/real_infra/market_data/`, `tests/real_infra/backtest/`, unit/CLI/config suites and
current client package scripts. Do not skip tests merely because they import removed CH code:
port semantic tests, retire only engine-specific assertions with a written coverage mapping.

## 7. Stage order, outputs and ownership

All stage details and acceptance criteria are also encoded in each prompt and the journal.
S01 is the documentation/contract workstream; S02–S10 implement, test and reconcile docs.
Each report: owned created/modified/deleted paths, outside-scope paths with rationale,
foreign changes excluded, actual commands/results, contract classifications, evidence,
remaining risk and next-stage entry readiness. Claims/receipts only via the bound updater.
A finished stage does not itself authorize running the next stage without the selected
continuation scope. Technical helper transition may mark the next row eligible only.

### S01 — Documentation, complete inventory and target contracts

- Prompt: [01-documentation-and-contracts.md](01-documentation-and-contracts.md). Dependencies: none.
- Required report: `reports/S01.md`.
- Outcome: Every inventory hit has a disposition and a migration owner; callers without literal CH strings are followed.
- Acceptance: Normative documentation distinguishes PG target from current CH runtime and covers D1–D7; historical evidence is preserved. Query catalogue, resource guardrails, reset manifest shape, versioned contract changes and full stage proof map are recorded.
- Proof: Read-only source audit plus normative documentation amendments; no runtime/schema activation.
- Recovery/compatibility: D6 and section5; no independent broader authority.

### S02 — PostgreSQL market schema and lifecycle

- Prompt: [02-schema-and-migrations.md](02-schema-and-migrations.md). Dependencies: S01.
- Required report: `reports/S02.md`.
- Outcome: Additive migrations create every D2 table/constraint/index and typed raw/funding fields; existing migration checksums remain intact.
- Acceptance: Clean/resumed/repeated PG bootstrap and historical/month-rollover partition provisioning pass on a disposable instance. The app uses bounded non-superuser capabilities; partitions/coverage/frontier generations and invalidation APIs are explicit.
- Proof: Real PG schema/migration/partition tests; this is not production cutover or query-performance proof.
- Recovery/compatibility: D6 and section5; no independent broader authority.

### S03 — Atomic batch ingestion and bounded PostgreSQL access

- Prompt: [03-transactional-ingestion.md](03-transactional-ingestion.md). Dependencies: S02.
- Required report: `reports/S03.md`.
- Outcome: COPY staging, raw persistence, deterministic canonical/funding upsert and coverage deltas commit atomically with truthful receipt semantics.
- Acceptance: Duplicate-in-batch, retry, equal/stale version and concurrent WS/REST tests preserve exact canonical/index truth. Connection/transaction lifetime, finite pools/admission and SQLSTATE retry behavior are bounded; existing recovery batching and provider limits survive.
- Proof: Transactional real-DB correctness and concurrency/failure receipts, not provider uptime or full-app migration.
- Recovery/compatibility: D6 and section5; no independent broader authority.

### S04 — Optimized readers, coverage index and verified history frontier

- Prompt: [04-queries-coverage-frontier.md](04-queries-coverage-frontier.md). Dependencies: S03.
- Required report: `reports/S04.md`.
- Outcome: Every CH reader/verifier/coordinate capability has an equivalent PG implementation and contract tests, including funding/readiness.
- Acceptance: All D5 query families have query-plan/row-buffer evidence and result parity; bulk readers bound client memory. Actual/source-absent/unknown states and exclusive verified frontier remain correct through failed provider reads, leases, late candles, deletion and rebuild.
- Proof: Real PG query semantics and plans on representative ranges; final full mixed performance is S09.
- Recovery/compatibility: D6 and section5; no independent broader authority.

### S05 — Switch all consumers, contracts and user-visible behavior

- Prompt: [05-all-consumers-and-wiring.md](05-all-consumers-and-wiring.md). Dependencies: S04.
- Required report: `reports/S05.md`.
- Outcome: Every supported application/CLI/worker/script is wired to PG, with no reachable CH query or fallback.
- Acceptance: Repair source enums/config/errors/events/metrics are migrated consistently; persisted old values are handled by additive migration or scoped reset. Artifacts, fixed-input backtests, strategy tails, metadata jobs and supported clients preserve behavior and current security/loading boundaries.
- Proof: Composition-root/port/API/UI integration with real PG; no local installation cutover yet.
- Recovery/compatibility: D6 and section5; no independent broader authority.

### S06 — PostgreSQL-only installation, backup and operations

- Prompt: [06-installation-backup-observability.md](06-installation-backup-observability.md). Dependencies: S05.
- Required report: `reports/S06.md`.
- Outcome: base/trading/ml and embedded/external PG profiles generate/bootstrap/run without CH service, image, secret, migration or health requirement.
- Acceptance: Config/storage-status/backup owner schemas have explicit compatible version handling; stale/old incompatible inputs fail before side effects. New-format backup/restore and runtime operations preserve digest/quiesce/security invariants; runbooks and monitoring are PG-only.
- Proof: Generated config/schema and disposable installation/backup runtime proof; no release publication or current local reset.
- Recovery/compatibility: D6 and section5; no independent broader authority.

### S07 — Remove legacy implementation and reconcile every reference

- Prompt: [07-remove-clickhouse-and-docs-audit.md](07-remove-clickhouse-and-docs-audit.md). Dependencies: S06.
- Required report: `reports/S07.md`.
- Outcome: CH adapters/DDL/client dependency and executable maintenance/benchmark paths are removed or ported; no product/test-tool runtime import remains.
- Acceptance: Every residual inventory match has a narrow documented historical or rejection-test justification; all active normative docs describe PG. Dependency lock, generated docs/map/schemas/topology and CI semantic coverage mapping are consistent.
- Proof: Repository-wide dependency/reference closure and static/build/import gates, not complete functional performance.
- Recovery/compatibility: D6 and section5; no independent broader authority.

### S08 — Full correctness, recovery and browser regression suite

- Prompt: [08-functional-recovery-verification.md](08-functional-recovery-verification.md). Dependencies: S07.
- Required report: `reports/S08.md`.
- Outcome: Real PG semantics, all consumers and supported UI/API journeys pass with CH absent; expected results are not replaced by mocks.
- Acceptance: Failure drills prove truthful receipts/cursors/events and bounded recovery after DB restart/disconnect/contention. Backup/restore, bootstrap, reset invalidation and artifact/backtest parity pass; independent reviewer finds no unresolved material migration issue.
- Proof: Functional/failure integration on isolated targets; no long-soak or current-installation delivery claim.
- Recovery/compatibility: D6 and section5; no independent broader authority.

### S09 — Production-query performance, memory and long mixed-load proof

- Prompt: [09-performance-and-soak.md](09-performance-and-soak.md). Dependencies: S08.
- Required report: `reports/S09.md`.
- Outcome: Full-size production-adapter mixed runs meet documented S01 guardrails or have explicit unresolved blockers; no tiny-data substitute.
- Acceptance: Three repeated ≥5-minute normal/stress runs and ≥60-minute soak include existing PG application control traffic and ≥30 large-read samples per regime. Plans, all result hashes/receipts, row counts, CPU, DB+client memory, pool waits, WAL/checkpoints/autovacuum, disk and failure counts are retained with reproducible inputs.
- Proof: Measured capacity for the stated hardware/dataset/mix only; not universal capacity or provider uptime.
- Recovery/compatibility: D6 and section5; no independent broader authority.

### S10 — Local development cutover, real streams and final acceptance

- Prompt: [10-local-cutover-and-final-audit.md](10-local-cutover-and-final-audit.md). Dependencies: S09.
- Required report: `reports/S10.md`.
- Outcome: Current local installation uses PG-only market storage, preserved accounts/settings/secrets/selections, and correct fresh history/artifact generation.
- Acceptance: Real selected history advances and consecutive closed candles arrive after bounded disconnect/reconnect; independent SQL/API observations and event logs agree. No reachable CH dependency/process requirement remains; final docs, setup/backup guidance, gates and known data disposition are complete.
- Proof: Actual identified local installation delivery and real-provider evidence; no remote deployment or Git publication.
- Recovery/compatibility: D6 and section5; no independent broader authority.

## 8. Review, unresolved implementation details and handoff

No owner decision is required to prepare this package. Direction, clean-refill allowance,
full consumer scope and preservation boundaries are explicit. Engineering guardrails and
physical query details remain target proposals subject to S01/source-backed refinement,
not claims of completed tuning. Current local runtime identity is deliberately re-resolved
at S10, not guessed now. Future reports/test artifacts are produced by predecessor stages.

Independent architecture/contract review is required by repository policy for irreversible
migration and backup/security boundary changes. Authoring review evaluates this plan, not
implementation. S08/S10 acceptance must have current evidence for the implemented result;
no recursive reviewer chain. Authoring validation uses the installed
`prompt-manager/scripts/validate_pack.py` (Python standard library), profile
`prompt-pack-ledger/v1` / `stage-prompt/v1`, `--root` repository, `--ledger` this journal,
`--check draft` and `--check entry --stage S01`. Pass means structural/file binding only;
status is not changed. Updater tests operate exclusively on temporary two-stage fixtures.

Authoring review completed on 2026-10-13: independent architecture/contract review,
followed by a narrow fix verification, found no unresolved P1/P2 issues in the reviewed
planning scope. One P2 about missed value corrections after a lost artifact hint was
resolved through D2/D3 durable content revision and S02–S05/S08 acceptance tests.
This verdict is planning readiness only, not implementation or deployment acceptance.

## 9. PostgreSQL references checked for this design

- [PostgreSQL16 partitioning](https://www.postgresql.org/docs/16/ddl-partitioning.html):
  partitioned PK includes the partition key; this plan uses direct timestamp range partitioning.
- [INSERT / ON CONFLICT](https://www.postgresql.org/docs/16/sql-insert.html):
  deduplicate staging keys before UPSERT; conditional version-aware updates and concurrency semantics.
- [Resource consumption](https://www.postgresql.org/docs/16/runtime-config-resource.html):
  work_mem multiplies across operations/sessions; it is not a server-wide reserved pool.
- [EXPLAIN](https://www.postgresql.org/docs/16/using-explain.html):
  validate actual rows/buffers and planner behavior on representative data.

## Appendix — observed dependency/reference inventory

This is a discovery baseline, not proof that every matching file is active and not an
allowlist. Refresh before execution. Each path must be classified in S01 and reconciled
in S07; immutable historical evidence is preserved. Generated artifacts are regenerated
from their owner, and executable legacy tools are ported or removed.

Observed HEAD: `1c29f095175d17a5a9e567cae252e3d4e15f0a40`; dirty paths at inventory: 337. Matched paths: 318.

### apps

- `apps/api/dto/market_data_reference.py`
- `apps/api/routes/market_data_reference.py`
- `apps/api/routes/monitoring.py`
- `apps/api/wiring/modules/market_data_reference.py`
- `apps/api/wiring/modules/ui_strategies_dashboard.py`
- `apps/cli/commands/backfill_1m.py`
- `apps/cli/commands/backtest_artifact_publish.py`
- `apps/cli/commands/funding_rate_catchup.py`
- `apps/cli/commands/rest_catchup_1m.py`
- `apps/cli/commands/sync_instruments.py`
- `apps/cli/wiring/db/clickhouse.py`
- `apps/cli/wiring/modules/market_data.py`
- `apps/migrations/research_runtime_probe.py`
- `apps/migrations/storage.py`
- `apps/migrations/storage_main.py`
- `apps/migrations/verify_storage_runtime.py`
- `apps/monitoring/__init__.py`
- `apps/monitoring/clickhouse_exporter.py`
- `apps/navigator-web/design-qa.md`
- `apps/navigator-web/src/data-downloads.tsx`
- `apps/navigator-web/src/monitoring-page.tsx`
- `apps/platform-web/e2e/foundation.spec.ts`
- `apps/platform-web/e2e/journey.spec.ts`
- `apps/platform-web/e2e/library.spec.ts`
- `apps/scheduler/backtest_artifact_publisher/main/main.py`
- `apps/scheduler/backtest_artifact_publisher/wiring/managed_worker.py`
- `apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py`
- `apps/scheduler/market_data_scheduler/main/main.py`
- `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`
- `apps/worker/market_data_ws/main/main.py`
- `apps/worker/market_data_ws/wiring/io_process.py`
- `apps/worker/market_data_ws/wiring/modules/market_data_ws.py`
- `apps/worker/market_data_ws/wiring/persistent_writer.py`
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`

### configs

- `configs/dev/clickhouse-local-observability.xml`
- `configs/dev/clickhouse-local-profiles.xml`
- `configs/installation/generated/base/compose.yaml`
- `configs/installation/generated/base/openbao.json`
- `configs/installation/generated/ml/compose.yaml`
- `configs/installation/generated/ml/control-policy.json`
- `configs/installation/generated/ml/effective-config.redacted.json`
- `configs/installation/generated/ml/observability/operational-manifest.json`
- `configs/installation/generated/ml/openbao.json`
- `configs/installation/generated/ml/prometheus.yml`
- `configs/installation/generated/ml/runtime-topology.json`
- `configs/installation/generated/ml/service-config.json`
- `configs/installation/generated/trading/compose.yaml`
- `configs/installation/generated/trading/control-policy.json`
- `configs/installation/generated/trading/effective-config.redacted.json`
- `configs/installation/generated/trading/observability/operational-manifest.json`
- `configs/installation/generated/trading/openbao.json`
- `configs/installation/generated/trading/prometheus.yml`
- `configs/installation/generated/trading/runtime-topology.json`
- `configs/installation/generated/trading/service-config.json`
- `configs/installation/roehub.yaml`
- `configs/installation/runtime-input-inventory.json`
- `configs/installation/runtime-service-manifest.json`

### docs

- `docs/architecture/README.md`
- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md`
- `docs/architecture/apps/cli/cli-backfill-1m.md`
- `docs/architecture/apps/web/web-ui-tests-docs-index-v1.md`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/00-baseline-and-contract-freeze.md`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/01-funding-storage-and-catchup.md`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/02-funding-artifact-family-and-coverage.md`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/03-preflight-runtime-defaults-funding-readiness.md`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
- `docs/architecture/identity/organizations-rbac-audit-v1.md`
- `docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/02-backtest-variant-strategy-creation.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/03-live-strategy-profile.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/04-run-stop-restart-lifecycle.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md`
- `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/01-baseline-handoff-freeze.md`
- `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/04-btcusdt-market-readiness.md`
- `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-2-functional-canary.md`
- `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md`
- `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-5-closure.md`
- `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-supervised-6h-soak.md`
- `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/14-final-readiness-docs-closure.md`
- `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md`
- `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md`
- `docs/architecture/market_data/market-data-application-ports.md`
- `docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md`
- `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/01-contract-foundation-audit-schema.md`
- `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/02-redis-hot-cache.md`
- `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/03-tail-provider-source-chain.md`
- `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/04-strategy-runner-integration-ack-policy.md`
- `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/05-metrics-alerts-runbook.md`
- `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/06-macstudio-repair-proof.md`
- `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/07-stage-12-4-rerun-handoff.md`
- `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md`
- `docs/architecture/market_data/market-data-live-tail-repair-v1.md`
- `docs/architecture/market_data/market-data-real-adapters-clickhouse-parquet.md`
- `docs/architecture/market_data/market-data-reference-api-v1.md`
- `docs/architecture/market_data/market-data-reference-data-sync-v2.md`
- `docs/architecture/market_data/market-data-rest-historical-catchup-1m-v2.md`
- `docs/architecture/market_data/market-data-runtime-config-invariants-v2.md`
- `docs/architecture/market_data/market-data-use-case-backfill-1m.md`
- `docs/architecture/market_data/market-data-ws-live-ingestion-worker-v1.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02a-data-source-inventory.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02b-feature-live-feed-contract.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02c-action-state-reward-contract.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04a-binance-futures-universe-whitelist.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04c-dataset-refresh-manifest.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/05-roehub-dataset-builder-v1.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07a-training-runner-smoke.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08l-reward-warm-start-research.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08m-supervised-warm-start-candidate-scorecard.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/09-model-registry-activation.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/10-per-ticker-calibration.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/10a-retraining-promotion-lifecycle.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/11-rl-tab-ui-skeleton.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/12-backend-entitlements.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `docs/architecture/operations/native-service-control-monitoring-admin-target-v1.md`
- `docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/00-contract-baseline-and-repair-route.md`
- `docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/01-controlled-egress-and-market-readiness.md`
- `docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/05-release-rebuild-local-lifecycle-and-handoff.md`
- `docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/evidence/01-market-readiness-proof.json`
- `docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/roehub-market-data-activation-and-instrument-selection-v1-stage-ledger.md`
- `docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/00-freeze-current-state-and-contracts.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/02-machine-readable-runbooks.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/03-installation-config-and-release-manifest.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/04-storage-bootstrap-and-migrations.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/05-organizations-rbac-and-audit.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/06-local-auth-sessions-and-recovery.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/07-oidc-provider-integration.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/09-research-context-organization-isolation.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/10-trading-context-organization-isolation.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/11-notification-and-telegram-provider-plugins.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/12-plugin-platform-api-and-sdk.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/14-artifact-store-and-default-bundles.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/17-containerize-all-services-and-profiles.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/20-observability-and-operational-actions.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/21-backup-restore-upgrade-and-rollback.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/23-greenfield-installation-lifecycle-rehearsal.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/24-platform-matrix-release-candidate.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/17-runtime-topology-proof.json`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/20-observability-runtime-proof.json`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/21-backup-restore-upgrade-runtime-proof.json`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/22-signed-offline-release-bundle-runtime-proof.json`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/23-greenfield-installation-lifecycle-proof.json`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/24-macos-docker-desktop-greenfield-lifecycle-proof.json`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md`
- `docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md`
- `docs/architecture/project-map/PROJECT_MAP.md`
- `docs/architecture/project-map/project-map.json`
- `docs/architecture/project-map/project-map.mmd`
- `docs/architecture/project-map/project-map.toml`
- `docs/architecture/shared-kernel-primitives.md`
- `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`
- `docs/architecture/strategy/strategy-milestone-3-epics-v1.md`
- `docs/architecture/strategy/strategy-runtime-config-v1.md`
- `docs/runbooks/clickhouse-memory-profiles.md`
- `docs/runbooks/clickhouse-partition-dedup.md`
- `docs/runbooks/generated/ru/database.clickhouse-degraded.md`
- `docs/runbooks/generated/ru/worker.market-data-live-tail-gap.md`
- `docs/runbooks/installation-backup-restore-upgrade.md`
- `docs/runbooks/legacy/backtest-artifacts-rebuild.md`
- `docs/runbooks/legacy/clickhouse-memory-profiles.md`
- `docs/runbooks/legacy/clickhouse-partition-dedup.md`
- `docs/runbooks/legacy/market-data-metrics-reference-ru.md`
- `docs/runbooks/legacy/market-data-metrics.md`
- `docs/runbooks/legacy/prod-dashboard-metrics-reference-ru.md`
- `docs/runbooks/legacy/strategy-live-worker.md`
- `docs/runbooks/locales/ru/database.clickhouse-degraded.yaml`
- `docs/runbooks/locales/ru/worker.market-data-live-tail-gap.yaml`
- `docs/runbooks/market-data-efficiency-local-activation-20261011.md`
- `docs/runbooks/market-data-live-tail-repair.md`
- `docs/runbooks/market-data-redis-streams.md`
- `docs/runbooks/navigator-local-market-data-recovery.md`
- `docs/runbooks/offline-release-installation.md`
- `docs/runbooks/ops/database.clickhouse-degraded.yaml`
- `docs/runbooks/ops/worker.market-data-live-tail-gap.yaml`
- `docs/runbooks/runbooks.json`
- `docs/runbooks/storage-bootstrap-and-migrations.md`

### infra

- `infra/docker/.env.example`
- `infra/docker/Dockerfile.storage-migrations`
- `infra/docker/docker-compose.market_data.yml`
- `infra/docker/docker-compose.yml`
- `infra/docker/runtime-entrypoint.py`
- `infra/docker/storage-embedded.compose.yml`
- `infra/docker/storage-external.compose.yml`
- `infra/monitoring/rules/market-data-funding.rules.yml`
- `infra/monitoring/rules/market-data-live-tail-repair.rules.yml`
- `infra/monitoring/rules/platform-monitoring.rules.yml`
- `infra/monitoring/rules/strategy-producer.rules.yml`

### migrations

- `migrations/clickhouse/funding_rates_ddl.sql`
- `migrations/clickhouse/history_source_absences_ddl.sql`
- `migrations/clickhouse/manifest.json`
- `migrations/clickhouse/market_data_ddl.sql`
- `migrations/postgres/0023_market_data_work_requests_v1.sql`

### pyproject.toml

- `pyproject.toml`

### schemas

- `schemas/backup/installation-backup-manifest.schema.json`
- `schemas/backup/installation-backup-policy.schema.json`
- `schemas/backup/installation-capture-record.schema.json`
- `schemas/config/roehub.schema.json`
- `schemas/config/storage-status.schema.json`

### scripts

- `scripts/ops/optimize_canonical_partitions.sh`
- `scripts/rl_trading/stage04b_binance_futures_history_backfill.py`
- `scripts/rl_trading/stage05_roehub_dataset_builder.py`
- `scripts/rl_trading/stage06_dataset_qa_session_extractor.py`
- `scripts/rl_trading/stage08j_article_session_extractor_dataset.py`

### src

- `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py`
- `src/trading/contexts/backtest_artifacts/application/use_cases/publish_backtest_artifacts_v2.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/__init__.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/btcusdt_market_readiness_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/catalog_coverage_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/enabled_instrument_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/enabled_market_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/enabled_tradable_instrument_search_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_coverage_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_range_verifier.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/instrument_coverage_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/raw_kline_writer.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/ref_instruments_writer.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/ref_market_writer.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/work_errors.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/postgres/work_request_repository.py`
- `src/trading/contexts/market_data/application/dto/live_tail_repair.py`
- `src/trading/contexts/market_data/application/ports/stores/enabled_market_reader.py`
- `src/trading/contexts/market_data/application/ports/stores/enabled_tradable_instrument_search_reader.py`
- `src/trading/contexts/market_data/application/ports/stores/raw_kline_writer.py`
- `src/trading/contexts/market_data/application/services/closed_candle_tail_provider.py`
- `src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py`
- `src/trading/contexts/market_data/application/use_cases/rest_catchup_1m.py`
- `src/trading/contexts/market_data/application/use_cases/rest_fill_range_1m.py`
- `src/trading/contexts/market_data/application/use_cases/search_enabled_tradable_instruments.py`
- `src/trading/contexts/market_data/application/use_cases/time_slicing.py`
- `src/trading/contexts/operations/backup_contracts.py`
- `src/trading/contexts/rl_trading/domain/feature_contract.py`
- `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py`
- `src/trading/contexts/strategy/application/ports/closed_candle_tail_provider.py`
- `src/trading/platform/config/installation.py`
- `src/trading/platform/config/operational_topology.py`
- `src/trading/platform/config/runtime_topology.py`

### tests

- `tests/fixtures/installation/roehub-external.yaml`
- `tests/fixtures/market_data/stage05-compose.override.yaml`
- `tests/fixtures/storage/clickhouse-interrupted/0001_interrupted.sql`
- `tests/fixtures/storage/clickhouse-interrupted/manifest.json`
- `tests/notebook_tests/02_rest_catchup_1m.ipynb`
- `tests/real_infra/backtest/test_artifact_integrated.py`
- `tests/real_infra/backtest/test_automatic_base_lifecycle.py`
- `tests/real_infra/market_data/test_clickhouse_workloads.py`
- `tests/real_infra/market_data/test_full_history_clickhouse.py`
- `tests/real_infra/market_data/test_stream_recovery_postgres.py`
- `tests/real_infra/market_data/test_work_requests_postgres.py`
- `tests/unit/apps/api/test_market_data_reference_routes.py`
- `tests/unit/apps/api/test_market_data_reference_wiring_module.py`
- `tests/unit/apps/cli/test_backfill_1m_parsing.py`
- `tests/unit/apps/cli/test_backtest_artifact_publish_cli.py`
- `tests/unit/apps/control_agent/test_backup_backend.py`
- `tests/unit/apps/migrations/test_storage_lifecycle.py`
- `tests/unit/apps/monitoring/test_clickhouse_exporter.py`
- `tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py`
- `tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py`
- `tests/unit/apps/worker/test_persistent_writer.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_btcusdt_market_readiness_reader.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_canonical_candle_index_reader.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_canonical_candle_reader.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_enabled_instrument_reader.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_enabled_market_reader.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_enabled_tradable_instrument_search_reader.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_funding_rate_store.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_instrument_coverage_reader.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_raw_kline_writer.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_ref_instruments_writer.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_thread_local_gateway.py`
- `tests/unit/contexts/market_data/adapters/test_postgres_candle_repair_audit_repository.py`
- `tests/unit/contexts/market_data/application/services/test_closed_candle_tail_provider.py`
- `tests/unit/contexts/market_data/application/services/test_work_requests.py`
- `tests/unit/contexts/rl_trading/domain/test_feature_contract.py`
- `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py`
- `tests/unit/contexts/strategy/application/test_closed_candle_tail_provider_contract.py`
- `tests/unit/contexts/strategy/application/test_strategy_live_runner.py`
- `tests/unit/infra/test_monitoring_assets.py`
- `tests/unit/platform/config/test_operational_topology.py`
- `tests/unit/tools/test_clickhouse_workload_installation.py`
- `tests/unit/tools/test_generate_runtime_topology.py`
- `tests/unit/tools/test_installation_backup_bundle.py`
- `tests/unit/tools/test_local_platform.py`
- `tests/unit/tools/test_offline_release_bundle.py`

### tools

- `tools/backup/verify_runtime.py`
- `tools/docs/generate_project_map.py`
- `tools/installation/clickhouse_workloads.py`
- `tools/installation/local_platform.py`
- `tools/qa/backtests_client_fixture.py`
- `tools/qa/market_data_ingestion_comparison.py`
- `tools/qa/market_data_mixed_load.py`
- `tools/qa/market_data_recovery_comparison.py`
- `tools/qa/market_data_resource_comparison.py`
- `tools/qa/postgres_storage_prototype.py`
- `tools/qa/workpages_monitoring.py`
- `tools/release/THIRD_PARTY_NOTICES.md`
- `tools/release/oss_policy.json`
- `tools/release/preliminary-sbom.spdx.json`
- `tools/release/release-metadata.json`
- `tools/release/verify_greenfield_lifecycle.py`
- `tools/release/verify_market_data_readiness.py`
- `tools/release/verify_observability_runtime.py`
- `tools/release/verify_runtime_topology.py`

### uv.lock

- `uv.lock`

