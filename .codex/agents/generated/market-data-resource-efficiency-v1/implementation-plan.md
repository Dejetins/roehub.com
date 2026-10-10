# Market Data resource efficiency — implementation plan v1

Date: 2026-10-10. Engineering language: English; executor reports to the owner in Russian.
Status: proposed implementation design within the owner's accepted four-part scope;
authoring complete is not implementation acceptance. Execution mode: manual_sequential.

## Intent, authority and execution artifacts

The owner requested four improvements, a measurement for each, four executor prompts,
and an iteration journal, preserving the current architecture and updating its documents.
This delivery authors that package. Submitting a stage prompt selects that implementation;
execute one stage, update its evidence/journal, and stop. No repeated product approval is
required for routine implementation within the selected stage. No stage has run yet.

- `plan_doc`: [implementation-plan.md](implementation-plan.md).
- `prompt_pack_dir`: this directory, `.codex/agents/generated/market-data-resource-efficiency-v1/`.
- `stage_ledger`: [iteration-ledger.md](iteration-ledger.md), the only execution-state record.
- Order: S01 → S02 → S03 → S04. Each consumes its predecessor's report and verified runtime state.
- Reuse the existing POSIX atomic updater unchanged; do not introduce a new execution engine.
- No production target, Git publication, branch/worktree/Goal, exchange trading, partition
  migration, data deletion, new broker/service, UI redesign, or resource-limit increase.
  Preserve all foreign dirty changes. Do not treat the currently dirty checkout as HEAD.

## Observed starting point, not a benchmark baseline

Read-only investigation on 2026-10-10; HEAD was
`718a1d8c2dcc28ccb2c0704de3b663785494e0c3` with extensive uncommitted changes.
Optional raw aggregate evidence lives in `.cache/market-data-review-2026-10-10/`;
its absence later requires fresh measurement, not reconstruction or invented results.

| Observation | Value and boundary |
|---|---|
| ClickHouse | 25.8.33.6; Docker 1.21–1.31 GiB / 1.5 GiB; two instantaneous samples |
| Internal memory | allocated about 500 MiB; RSS about 885–888 MiB; not additive |
| Limits | server tracked memory 1 GiB; default query 256 MiB; max_threads=2 |
| Data | about 6.28 million unique minutes, 102 stored coordinates, four live streams |
| Parts | canonical about 3,564 active parts / 2,187 daily partitions; no partition change here |
| Publisher, five minutes ending around 16:58 UTC | 270 bounds reads / 796,509,441 read rows; 9 coordinate reads / 56,490,258 rows; 270 first-minute checks |
| Insert workload in that window | 20 new candles; 10 inserts; average two rows per new part per table |
| Scheduler / collector / publisher memory | about 144 / 317 / 307 MiB; no historical job active |
| Live correctness | each of four coordinates gained four distinct minutes from 16:53 to 16:57 UTC; recovery queue empty |

These samples identify S01's measured cost. They do not prove a leak, saturated-load
capacity, causal improvement, or baseline performance for S02–S04. Historical job elapsed
times include downtime/retries and must not be used as clean throughput measurements.

## Architecture and sources of truth

Keep application-owned ports, outbound adapters and explicit composition roots. Domain
code must not import ClickHouse, psycopg, requests, asyncio infrastructure or app wiring.
Market Data owns canonical candle persistence, minute availability and recovery; Backtest
Artifacts owns source obligations and materialization; installation wiring owns service
identities and server settings. PostgreSQL owns durable jobs, leases and recovery metadata.
Canonical candles own prices; bitmap states are a rebuildable read index, not a second
candle store. Exchange-confirmed absent minutes never become synthetic candles, last
candle timestamps, artifact bounds or exported observations. Keep the append-only/index
repair contract and exact exclusive `before` semantics.

Verified entrypoints (repository relative):

- `src/trading/contexts/backtest_artifacts/application/services/base_source_reconciliation.py`
  and `adapters/outbound/persistence/clickhouse/stored_coordinates.py` in that context.
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/`:
  `canonical_candle_index_reader.py`, `history_coverage_reader.py`, `gateway.py`.
- `src/trading/contexts/market_data/application/services/`:
  `stored_candle_signal.py`, `work_requests.py`, `insert_buffer.py`, `rest_fill_queue.py`.
- `src/trading/contexts/market_data/application/ports/stores/stream_recovery_store.py`
  and the corresponding PostgreSQL adapter.
- `apps/scheduler/backtest_artifact_publisher/wiring/managed_app.py`;
  `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`;
  `apps/worker/market_data_ws/wiring/{modules/market_data_ws.py,io_process.py,persistent_writer.py}`.
- `apps/cli/wiring/db/clickhouse.py`, `apps/api/wiring/modules/market_data_reference.py`,
  `tools/installation/local_platform.py`, `configs/dev/clickhouse-local-profiles.xml`.

Normative documents and exact update ownership are in the documentation matrix below.
Historical whitelist-era documents must not override current instrument selections.

## Decisions and stage outcomes

### D1 / S01 — indexed, batched source reconciliation

Replace publisher discovery and per-coordinate bounds scans with a bounded page read
from the existing exact minute index. The application contract returns ordered coordinates
and actual first/last closed minute before one fixed cutoff; adapter SQL merges duplicate
aggregate states. Preserve cursor ordering and empty/epoch-zero semantics. Do not use
`sumMerge(rows_state)` as unique coverage or use a max timestamp beyond the cutoff.

Use the existing post-write hint seam for prompt re-observation, coalescing duplicates;
hints remain optional and never make a committed candle write fail. Preserve a bounded
periodic sweep of the index to recover missed hints and newly stored coordinates. Inspect
the existing hint→obligation flow before changing it; avoid introducing another queue or
source-of-truth table just to carry hints. Missing/stale index state must remain recoverable:
keep explicit bounded repair/rebuild routes and health evidence, not a hidden full-scan
fallback on every page. Append-only/delete assumptions remain explicit.

Alternative of merely slowing the old scan retains history-size cost and misses freshness.
The selected approach reuses existing data structures and the publisher's ports; it does
not change artifact identities or publication rules. Compatibility: internal port change
must update every consumer/test; public API and candle schema `none`; internal read
contract `compatible-change` when existing callers remain supported.

Proof: exact parity against canonical fixtures, no canonical table reads in the normal
discovery/bounds path, query count bounded by pages rather than 2×coordinates, replay of
lost hints, empty/duplicate/non-aligned-cutoff/source-absence/index-repair cases. A periodic
index sweep still scales with index size; do not advertise constant total work.

### D2 / S02 — server-enforced workload budgets

Introduce four installation-owned workload classes: `stream`, `interactive`, `backfill`,
`artifacts`. Assign them in trusted composition roots, never from public request input.
Use ClickHouse-native settings profiles and distinct service identities for class-wide
limits shared across processes: per-query memory/threads/timeouts plus per-user aggregate
memory and concurrent-query caps. An in-process semaphore is not global enforcement.
Account for INSERT materialized-view work in the submitting class. Audit direct clients,
child processes, artifact builders, normal scheduler mode and API paths; do not leave an
unlimited default-user bypass. Administrative/migration traffic is explicitly separate.

Read installed 25.8 capabilities and test effective settings/constraints before selecting
values. Derive caps from the S01 candidate's measured baseline plus measured query peaks,
background merges/caches and safety headroom. Bound total background admission so live
and interactive requests retain measured capacity. Do not sum unrelated maxima as actual
usage or claim these limits hard-partition OS memory/CPU/IO. Native workload CPU/IO
scheduling is optional only if supported and needed by the mixed-load test; it is not a
prerequisite or permission to upgrade ClickHouse. Record why the selected mechanism
satisfies the observed contention scenario.

Provision identities/profiles through existing installation configuration, private secret
files and migration lifecycle. Scope grants to actual reads/writes and MV execution;
no empty passwords, broad new admin grants or secrets in generated tracked files. Profile
constraints must prevent ordinary runtime clients overriding caps. Keep current identities
until new clients are verified; no credential rotation/deletion without covering authority.
Test on disposable identities first. Use one independent security-boundary review.

Budget rejection becomes a bounded, classified resource wait/retry for workers, without
discarding checkpoints/receipts or reporting false source gaps. API retains its established
error/read-recovery contract; artifact scheduling retains its existing durable states.
Bound retry queues and avoid retry storms; low-priority work must eventually progress.
Profile provisioning precedes client switch. Rollback drains new clients and restores the
previous known config; leave additive identities inert until separately authorized cleanup.

Compatibility: additive deployment/config contract `compatible-change` with explicit
upgrade order; public API `none` unless a documented existing error mapping needs repair;
security boundary changed but no end-user permission expansion. Missing supported server
settings blocks this mechanism, not permission to simulate global budgets locally.

### D3 / S03 — rolling scheduling and shared exchange admission

Retain the single PostgreSQL coordinator lease and bounded reusable thread pool. Refill a
free worker as soon as one window finishes; checkpoint/yield remains fair across jobs.
Wait only for no eligible work, cooldown deadlines or bounded idle polling. Perform orphan
recovery before dispatch or under proven quiescence; never reclaim slow live siblings.
Lease release, shutdown and cancellation must drain or definitively stop every owned write.
Isolate bounds discovery with bounded concurrency/timeouts so it cannot block all ready work.
Do not hold database transactions over network calls.
An HTTP read-inactivity timeout is not an absolute request deadline: a slow-trickle body
can keep it alive. Enforce a total lifetime including connection, body consumption and
internal retries, using owned cancellable I/O or the existing managed-process boundary.
Prove that transport/process termination finishes before releasing its concurrency permit,
including loss of the PG permit session; timing out an awaiting future while its thread
keeps running is not cancellation.

Keep existing REST range planning, exact deduplication, source-absence verification,
per-instrument exclusion, pause/cancel checkpoints and durable errors. Initially retain
the current window sizes and concurrency while measuring the scheduling change; changing
them requires a separately identified experiment within this stage. Session reuse belongs
in the HTTP adapter: one lifecycle-owned session per thread/process, closed on shutdown,
never shared unsafely or inherited across process spawn. The existing persistent live
writer is already present and must not be reimplemented.

Add a Market Data application port for shared exchange request admission and implement it
with existing PostgreSQL infrastructure. Use atomic reservations of endpoint-weighted
quota windows and shared cooldowns keyed by provider + explicit egress quota scope, not by
symbol or organization. Use database time for cross-process quota accounting; monotonic
local clocks for waits. Multiple active provider limits must be reserved atomically.
Bound request concurrency across scheduler and collector REST workers using durable owned
permit metadata. Session advisory locks may serialize reservation, but their automatic
release on connection loss is not proof that an HTTP call stopped. Keep uncertain in-flight
permits charged until confirmed transport termination or a proven end-to-end deadline and
safe reclamation condition; no permit may be reclaimed solely by TTL while its call can run.
Specify recovery for owner death, lost PG connection and unknown request outcome, failing
closed on uncertain ownership; no open SQL transaction is held during HTTP. Reuse current
managed-I/O lifecycle primitives rather than introducing a separate permit daemon.
All retries, metadata probes and recovery requests in the supported installation enter
the same admission path. Preserve Retry-After and provider weight headers; a sent request
keeps its spent quota even on error. Fail closed into a durable resource wait when quota
coordination is unavailable; WebSocket traffic itself does not consume REST permits.

Do not hardcode universal exchange limits or trust a config `autodetect` label as behavior.
Verify current primary provider documentation at implementation, support validated
conservative defaults until headers arrive, and document NAT/external-client limits.
Scope guarantees to participating installation clients; unrelated applications sharing an
IP are not controlled. No new Redis dependency, daemon or generic workflow engine.

Alternatives: larger pools/smaller sleeps alone amplify provider and storage pressure;
process-local limiters do not coordinate the two existing ingestion runtimes. Reuse PG's
current coordination boundary instead. Additive quota metadata uses the normal migration
manifest; drain participating clients before rollback, preserve job progress and cooldowns.
Compatibility: application ports/config/persistence `compatible-change`; existing job/API
semantics `none`. Adapters keep provider-specific policy out of domain entities.

### D4 / S04 — asynchronous, batched durable recovery admission

Keep the existing recovery journal, keys, tokens and restart semantics. Add an application
boundary for asynchronous recovery operations with a bounded batching coordinator and
a psycopg async adapter or equivalently bounded lifecycle-owned off-loop adapter justified
by measurement. Prefer native async PostgreSQL using the already installed driver. Batch
admission in one transaction, return per-key existing/new receipts, and batch completion
where safe. No connection-per-candle, unbounded `to_thread` fan-out or fire-and-forget tasks.

Critical order: reserve bounded admission capacity → persist metadata batch → confirm commit
→ admit to the candle write buffer → confirm canonical write → token-checked completion.
Before durable commit a caller is waiting, not accepted. Capacity bounds include pending
admissions, in-flight DB batches, the write buffer and retries, in rows and retained bytes.
Per-caller cancellation cannot cancel another caller's batch or delete a committed receipt.
Unknown commit outcome is reconciled by idempotent recovery key before retry/completion.
Shutdown either drains accepted work or leaves durable repair metadata; reconnect/tail
repair covers arrivals not yet acknowledged. Receipts for actual WS candles always require
actual canonical candles, never source-absence markers. No exactly-once claim across PG/CH.

Audit all synchronous recovery-store calls reachable from collector async tasks, including
enqueue/start/retry/completion and connection health publication; migrate or isolate those
that otherwise retain the same event-loop stall. Retain the synchronous contract for
synchronous consumers if necessary rather than hiding blocking calls behind async names.
Do not share one connection concurrently or across a fork. Keep the collector lease and
conditional token updates authoritative; receipt conflicts/failed entries retain existing
semantics. Do not weaken durability to win latency.

Compatibility: additive async/batch port `compatible-change`; preserve persisted journal
shape if possible and dual-read any necessary additive schema during rollout; public API
and source-absence semantics `none`. Drain admission on upgrade/rollback; retained pending
receipts must remain readable by the supported rollback version. Independent review must
cover any material unresolved crash/cancellation risk before acceptance.

## Comparable measurement protocol (M)

S01 establishes a small reusable harness under existing `tools/qa` / `tests/perf_smoke`
conventions; a focused new module is allowed because no matching ingestion comparison
harness was found. Do not make a separate benchmark service. Each stage extends the same
workload/result schema. Live aggregate probes are low-impact; saturated and faulted runs
use disposable databases/containers through the current runbook, never the user's live data.

1. Before mutation capture relevant source/diff hashes, image IDs versus mounted source,
   Python/DB versions, hardware, cgroup limits, pool/thread settings, active workers/tabs,
   fixed timestamps, fixture hashes, row/coordinate/part counts and supported clock mode.
   Baseline S01 is the actual pre-S01 checkout; baseline S02/S03/S04 is the verified accepted
   preceding candidate. Do not use HEAD as a substitute for dirty source. Preserve a
   scoped read-only source snapshot outside Git if needed; never reset the workspace.
2. Freeze workload and measurement commands before comparing. Separate warm/cold results;
   warm caches consistently, then run three paired baseline/candidate repetitions in
   alternating order on equivalent disposable states. Use 5-minute steady-state windows
   after 60-second warmup for service workloads; finite replays report full completion
   time as well. Retain all repetitions, failures and spread; do not rerun until green.
   Increase duration only for a recorded uncertainty, applied to both sides.
3. Required representative fixture: four streams / 102 stored coordinates; sparse and
   contiguous histories with real missing-minute semantics. S01 requires that source
   sweep; S02 requires the mixed-load experiment; S03 requires the queued-job/slow-provider
   experiment. S04 requires both four-stream and synthetic 100-stream synchronized minute
   bursts, on disposable targets with capacity preflight. A 1,000-stream burst or larger
   index sweep is optional capacity exploration: report not run/unavailable if unsuitable,
   without blocking the bounded required result or claiming support for that scale.
   Missing a required experiment keeps that stage unaccepted; no silent downgrade.
4. Measure query count/read_rows/read_bytes, per-query memory, CH tracked/allocated/RSS and
   container peaks separately; CPU time, IO/parts/merges; worker+child peak memory; backlog,
   error/retry/admission counts; rows/s; receipt-to-canonical and receive-to-canonical p50/p95/p99;
   API p95/p99; event-loop delay; PG transactions/connections per accepted candle. Attribute
   workloads by safe class/query tags, not stored raw SQL, symbol labels of unbounded
   cardinality, credentials or raw provider payloads. Sampling at 1s must disclose that
   it can miss shorter memory peaks. Include baseline steady state and post-drain plateau.
5. Report each stage's primary metric, all correctness and contention guardrails, absolute
   delta and percentage using the same denominator. Record baseline vs previous candidate
   and, at S04, an optional equivalent initial-vs-final comparison separately. Changes in
   data/cache/config/hardware require explanation; non-comparable runs are directional only.
6. Primary metrics: S01 read_rows/query fan-out per complete source sweep; S02 live/API tail
   latency plus low-priority progress under identical offered mixed load; S03 completed
   windows/s and free-worker idle time with fixed quotas/concurrency; S04 event-loop p99,
   PG transactions per accepted candle and canonical latency under identical bursts.
7. No arbitrary percentage/SLO was specified by the owner. Report performance acceptance
   `not assessed — numerical threshold not specified`; still provide measured deltas and
   a direction/noise verdict. Stage acceptance requires valid comparable before/after
   evidence, causal/structural checks and passed correctness gates, not an invented speedup.
   A regression in a primary metric or safety guardrail requires investigation and a
   reasoned resolution before advancement; noisy improvement is labelled inconclusive,
   never promoted to a numerical win. Missing required comparisons blocks acceptance;
   no automatic mock-only substitution or waiver.

## Stage-specific proof matrix

| Stage | Comparative experiment | Required correctness / failure evidence |
|---|---|---|
| S01 | Same 102-coordinate fixture + scale sweep, fixed cutoff, same publisher cadence; full canonical scans vs indexed page path | Bounds/cursor parity; zero normal canonical discovery reads; coalesced, delayed/lost hints; actual candles vs absences; repair route |
| S02 | Same replay of live writes + API reads + publisher + saturated backfill, same server/container limits; >1 process per low-priority class | Effective caps across processes, MV accounting, runtime cap override denied, least privilege, class retry/recovery, no live candle loss, finite background progress |
| S03 | Same queued jobs and deterministic provider emulator, one slow instrument, fixed 4-worker/1440-minute baseline; two REST runtimes share quota | Overlap/fairness/pause/cancel, lease-loss and slow-sibling recovery, pool drain, slow-trickle body beyond total deadline, 429/Retry-After, multi-quota/clock/restart/PG outage, no request bypass |
| S04 | Required same synchronized minute bursts at 4 and 100 streams; optional 1000-stream capacity run; PG latency/faults, identical buffer and flush bounds | Crash/cancel before and after PG commit, before/after CH acknowledgement and before receipt completion; duplicates, capacity/backpressure, PG/CH outages, shutdown/restart, actual-candle repair |

Real provider smoke after each stage uses only currently selected instruments and bounded
requests within existing authority. Observe at least two newly closed minutes in canonical
storage and corresponding recovery/stream state; a connected socket alone is insufficient.
For S03 prove a real bounded missing interval reaches verified completion if an authorized
target exists; otherwise report that provider proof unavailable separately from emulator
and real-database proof. Do not delete real candles to manufacture a gap. Synthetic replay
can establish scheduling/scale behavior, not real exchange capacity.

## Documentation matrix — update with implementation, never ahead of facts

| Existing canonical document | Stages and required amendment |
|---|---|
| `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md` | S01: batched observation, hints/reconciliation, index consistency; S02: artifact workload budget/waits |
| `docs/architecture/market_data/market-data-reference-api-v1.md` | S01: index/actual-candle semantics; S02: interactive resource error semantics; S03: scheduler/shared quota/window behavior; S04: async receipt acceptance/recovery |
| `docs/runbooks/navigator-local-market-data-recovery.md` | Each stage: current commands from managed installation, effective limits, upgrade/rollback, measurements and limits; replace stale batch-barrier description in S03 |
| `docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md` | S02: installation class/identity/config lifecycle; S03: participating collector/scheduler and egress quota scope, retaining current selections |
| `docs/runbooks/offline-release-installation.md` | S02 and S03 only where generated profiles/secrets/config/migration order change; no production delivery claim |
| `docs/architecture/market_data/market-data-application-ports.md` | S01/S03/S04 only for durable new or extended application port contracts, retaining historical context labels |
| `docs/architecture/apps/web/roehub-data-loading-contract-v1.md` | S02 only if resource-error/read retry behavior actually changes; no UI rewrite |

Update config examples and migration manifests alongside their owner code. Regenerate
project-map outputs only if component/entrypoint structure changes, then use its existing
generator check. Reports link updated sections and record explicit no-change reasons.
Do not rewrite historical claims as newly verified or create a competing architecture doc.

## Verification, review, rollout and stop rules

Extend existing metrics and runbook diagnostics, without a new monitoring service:
per-class admitted/rejected/wait time, oldest eligible job age and completion progress,
provider cooldown/deadline/permit loss, recovery backlog age/capacity, event-loop delay,
and latest actual canonical minute. Use bounded class/reason labels. Distinguish an idle
queue from starvation and a requested stream from observed delivery. Operational thresholds
come from configured deadlines/capacity and the measured workload; do not invent an owner
SLO. The runbook must map each symptom to the owning component and safe recovery action.

Select focused pytest suites from each prompt; use `pyproject.toml`,
`.github/workflows/ci.yml` and `tools/ci/route_changes.py` for required lint/type/CI routing.
Use existing real-infra opt-ins with disposable targets; default skipped tests are not
runtime proof. Browser QA is required when API resource behavior affects displayed state;
follow the current target/browser selection and loading contract.

One independent review is required for S02's service-identity/security boundary, and for
material unresolved recovery/concurrency risk in S03/S04. Reviewers do not recurse. Other
changes receive focused self-review. Review availability does not justify silently accepting
unresolved findings. Do not create release/credential operations outside covering authority.

Local runtime refresh uses only the current managed installation/runbook, preserving data,
jobs and selections. Installation files must reproduce changes after container recreation;
one-off `docker cp` is not a delivery mechanism. Build/activate only owned components with
current authority; no release publication or retired process launchers. Prefer additive
migrations and ordered drain/restart over dual writers. Report what was installed locally
separately from what was tested in fixtures and what remains only source code.

Agent-resolvable unknowns: exact workload caps (measure in S02), provider quota dimensions
(verify in S03), burst coalescing deadline (measure in S04). These are delegated technical
choices, not pre-existing owner blockers. If safe resource budgets cannot fit the unchanged
installation limit, report the measured constraint; do not silently raise it. An unrelated
foreign edit, unavailable disposable target or missing provider authority blocks only the
dependent operation/proof. Do not invent unavailable results or change accepted contracts
to make a test pass.
