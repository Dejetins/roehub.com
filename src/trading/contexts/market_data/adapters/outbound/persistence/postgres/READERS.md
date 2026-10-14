# PostgreSQL readers and verification contract

S04 implements adapters; runtime composition and consumer activation belong to S05.
The current installation is not switched by adding these classes.

## Read semantics and resources

Canonical DTO and array reads select one instrument over a half-open UTC range,
ordered by timestamp. DTO transport uses binary COPY into an anonymous temporary
file, then yields DTOs without holding an idle database transaction. Close abandoned
iterators. Arrays use server cursor batches of 4096 and anonymous temporary files;
returned int64 timestamps and float32 OHLCV are read-only NumPy memmaps. Empty arrays
retain the existing shapes/dtypes. Temporary disk usage scales with returned data;
conversion memory is bounded. Memmap pages remain subject to OS paging, so this is
not a promise of constant RSS for arbitrary downstream access or copies.

Index/coverage/source queries use compact actual/absence day masks and generation
summaries. `before` is exclusive, including a cutoff inside a minute; integer
microseconds avoid float rounding and epoch-zero falsiness. Actual masks alone
supply artifact bounds/counts. Missing/rebuilding derived state fails closed;
readers never repair or scan canonical history as a fallback. Multi-statement reads
use REPEATABLE READ READ ONLY configured before the first SELECT. A concurrent reset
cannot combine old readiness/frontier with new masks.

History counts accept at most 80000 requested keys; source observation pages at
most 50; search and tail are explicitly limited. Gap discovery expands days, not
minutes, and clips partial-day bit masks. Funding coverage uses each event's stored
interval; nullable canonical mark price and quality mapping preserve the port.
Reference/funding enumeration uses bounded transport; the existing tuple output
contract still requires memory proportional to the returned reference set.

Compose gateways by work class: bulk 30s, interactive 1s, reconcile 10s, reference
5s in the representative S04 evidence. The gateway imposes a total transaction/
connect deadline and 250ms lock timeout. Installation-wide pool composition,
bounded DNS preparation, physical connection cap, recovery reservations and build
admission are S05/S06 obligations; these adapters do not establish those guarantees.

## Provider verification and maintenance

Acquire `VerificationFence` **before** provider I/O, carrying generation and lease
token unchanged into `confirm`. Renew before the maximum 120s lease expires and
release in the caller's finally block. An expired token cannot be renewed; reacquire
requires a new provider observation. `confirm` requires a fully consumed successful
provider result, persisted observed candles, an aligned range of at most seven days,
an explicit acceptance policy and a live caller `check_lease` callback. Failed
provider reads must never call confirmation. Both database fencing and the callback
are checked inside the transaction; callback failure rolls all changes back.

Pass the exchange/instrument history origin to confirmation. Frontier advancement
covers only a contiguous prefix from that origin and is limited to a seven-day
slice per call. A disjoint verified suffix cannot jump over a gap. Actual data
supersedes accepted absence; absence has policy version and observation provenance.
Only a truly new instrument may bootstrap empty ready metadata. Existing unknown
or rebuilding metadata must go through explicit repair.

`PostgresCoverageMaintenance.begin` requires expected generation and reason. It
invalidates the generation before bounded deletion/rebuild, removes leases and
stores the rebuild checkpoint. `step` reconstructs at most seven days. Writers fail
closed until completion. Canonical deletion is limited to seven days per requested
range; raw history is retained. Maintenance privileges are narrowly granted by the
migration lifecycle. Applied migrations 0001–0037 remain unchanged; 0038 is additive.

## Artifact source observations

Source observations include generation, content revision, funding revision, exact
clipped actual count and index state. Persisted source targets carry this tuple.
A bounded sweep detects corrected old candles even when first/last/count are
unchanged and notification hints are lost. Empty/unknown observations clear desired
and published source state. The repository checks the current generation/revisions
and readiness under a generation row lock, and rejects older observation timestamps,
including stale delayed tombstones.

S05 must wire these observations, cadence, build admission and revision-fenced
publication together. Clearing source state alone does not prove rejection of an
already-running old build. Full consumer parity/format compatibility is S05/S08;
full mixed workload acceptance is S09. No UI, provider or production delivery claim
follows from the isolated PostgreSQL tests.
