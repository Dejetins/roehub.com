# Transactional ingestion (S03)

S05 activates application composition; see [CONSUMERS.md](CONSUMERS.md) for current
wiring and remaining S06 installation gates. S10 owns the actual local installation
switch. Schema 0036 is unchanged.
Additive 0037 introduces separate enrichment clocks because status-only updates omit
those fields. Bootstrap/checksum/readiness includes the new phase. No historical
schema bytes may be rewritten.

## Transaction and replay

`PostgresRawKlineWriter.write_1m` returns None only after synchronous durable COMMIT.
It rejects oversized input before SQL: <=8192 rows and a conservative <=8 MiB wire
estimate, including payload-order/escaping overhead. Callers split larger imports
before invoking the port. A call is one atomic transaction, never partially successful.
Reference market calls accept <=4 rows; instrument/universe/funding calls <=1000.
No per-candle connection, transaction or DDL. Provision requested monthly partitions
through S02 maintenance before submitting batches; absent partitions fail visibly.

COPY goes to transaction-local staging. Canonical keys are deduplicated before
UPSERT. Raw candles retain repeat observations as before; canonical/funding keys are
unique. Lock every generation in numeric market / Python string symbol order, then
coverage days in market/symbol/day order. S04 writers/rebuild must acquire generation
before day/summary/frontier locks too. Raw, canonical, masks, summary, actual-over-
absence deletion and content revision share one transaction. Count increments are
mask cardinality differences under these locks, not attempted inserts. Initial empty
keys can become ready; invalidated/rebuilding keys fail before mutation until S04
performs a bounded rebuild. Corrections remove affected verification certificates.

Version order is `(UTC ingested_at floored to milliseconds, payload_order)`. The
versioned length-prefixed encoding uses null/type tags, UTF-8, signed int64 big-endian,
UTC epoch-ms int64 and IEEE float64 with canonical NaN and positive zero. Greater
unsigned bytes win. Ingest timestamp and random ingest UUID do not enter payload
order. Equal version/payload preserves the existing ingest_id. Candle field order is
canonical DDL order. Funding uses provider raw DDL order (including mark_price or
category); raw and canonical persist the same token so provider-only ties converge.
Fixed byte-vector and opposite arrival-order fixtures exercise this contract.

Canonical projection preserves the existing raw->MV mapping: Binance absent quote/
trade/taker values become zero; Bybit turnover becomes zero, trade/taker fields NULL,
and ts_close is ts_open+1m. Bybit category defaults to linear; Binance mark_price
remains nullable. No invented provider JSON archive.

Candle revision increments once per instrument transaction with membership or
materialized-value changes. Source/ingest metadata-only changes do not increment it.
Funding has its own revision; provider-only mark_price/category changes do not change
canonical funding arrays, and therefore do not increment it. Replays after lost commit
responses converge, although raw candle observations may duplicate. Revisions survive
lost post-commit hints; S04/S05 must join generation/revisions in the source snapshot
and migrate artifact target comparison. This stage does not claim the old reconciler
already consumes revisions. Conservative invalidation across cutoffs remains allowed.

`write_then_publish` runs durable write before Redis publication or optional hint.
A failed publication/hint cannot turn a committed write into a failed receipt. The
insert buffer completes recovery entries only after its writer returns; failures
retain pending entries. The new `postgres_writer_child` is a selectable warmed,
killable IPC target with the existing 8 MiB frame and parent timeout. S05 makes this the default application writer. Installed-runtime delivery remains gated separately.

## Reference NULL semantics

Reference status/tradability uses updated_at/payload_order; enrichment independently
uses enrichment_updated_at/enrichment_order. Both parts of an enrichment call commit
atomically. A status-only update preserves omitted enrichment. An explicit newer NULL
clears a field. Older enrichment can be newer than the last enrichment even when a
later status already exists; it updates its fields without regressing that status.
Funding universe is whole-row versioned; NULL is authoritative, including all-NULL
optional fields. This intentionally breaks reliance on old CH argMax nullable-field
carry-forward. Opposite equal-version permutations are stable; CH merge parity is not
claimed for this deliberate change.

## Resource and failure boundary

`IngestionGateway` is lifecycle-owned: finite reusable local sessions, finite waiter
admission (default eight), <=1s pool wait within a default 5s total deadline, and at
most three retries with bounded backoff. Only serialization/deadlock/lock timeout,
listed server disconnect SQLSTATEs, class 08 and transport OperationalError retry.
Schema/permission failures escape immediately; no catch-all success. Local timeout,
lock timeout 250ms, UTC and synchronous_commit=on reset at transaction end. A socket
watchdog closes an in-flight session at the remaining absolute deadline; failed
sessions are discarded. Close after draining; the WS outer process can be killed.

Connection setup uses public libpq nonblocking polling with the same absolute
stop. Explicit Unix socket, literal IP, or hostname is required; implicit
service/environment/multi-host discovery is rejected. S05 resolves
hostnames in a bounded lifecycle before each physical connection and preserves host for TLS
verification. No abandoned connector thread or unbounded DNS exists inside admission.
See [PG16 PQconnectPoll](https://www.postgresql.org/docs/16/libpq-connect.html) and
[psycopg public pq wrapper](https://www.psycopg.org/psycopg3/docs/api/pq.html).

Installation envelope remains 48 physical connections: market16/control24/admin8.
DB-wide session advisory slots cap **admitted market sessions**, including idle pools:
ingestion0–3, interactive4–9, bulk10–11, recovery12, reconcile13, reference/funding14–15.
Async recovery lease holds slot12 even idle. These are shared across processes using
this gateway. `ConnectionBudget` validates allocation arithmetic; it is NOT a global
physical connection limiter. A physical handshake precedes advisory admission.

S03 proves the stated disposable composition: 16 admitted market sessions, 24 control
sessions and a successful additional admin connection; excess class admissions fail
and their physical connections close. It does not prove every installed process is
bounded. **S05/S06 must enumerate all processes/replicas, idle pools, sync/async lease
sessions and simultaneous handshakes, reserve control/admin headroom and enforce the
physical installation cap before activation.** Old control adapters outside this
seam are not silently counted as constrained by it. S09 owns representative workload
capacity, memory/CPU/WAL/soak measurements; no throughput claim arises from these tests.

### Hostname compatibility after S05

Host-only DSNs resolve under the same transaction deadline on every new physical
connection. A bounded resolver subprocess retains original host/TLS while preparing
hostaddr; its parent watchdog exits on outer writer death. See CONSUMERS.md for
measured localhost verify-full/wrong-name, timeout, termination and reconnect proof
and the remaining S06 embedded/external installation DNS gate. Original S03 reports
remain immutable historical evidence of the prepared-conninfo-only boundary.
