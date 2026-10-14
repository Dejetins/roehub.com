# PostgreSQL consumer composition (S05)

Application reads/writes now use `apps/cli/wiring/db/postgres.py`. This is a source
composition change, not an installed-environment cutover. `STRATEGY_PG_DSN` selects
the same database as control-plane publication fences; there is no separate canonical
target or ClickHouse fallback. Identity, organization access, provider HTTP, Redis and
artifact file ownership retain their existing boundaries.

## Activated seams

| Consumer | Market connection class | Work boundary |
|---|---|---|
| API reference/catalog/coverage, strategy tail | interactive | 1 second total |
| WS warmed writer, REST scheduler/CLI, Parquet import, funding writes | ingestion | 5 seconds total |
| Artifact canonical/funding export, RL dataset slabs | bulk | 30 seconds total |
| Stored-source reconciliation | reconcile | 10 seconds total |
| Reference bootstrap/metadata | reference | 5 seconds total |

One lifecycle-owned pool per DSN/class/process is cached, size one and at most eight
local waiters. WS writer owns its separate warmed child and drains/closes it. The
advisory class allocations total 16: ingestion4, interactive6, bulk2, recovery1,
reconcile1, reference2. Idle pooled sessions retain their class slot. Reconnects close
failed sessions and repeat bounded DNS preparation. Resolution runs without DSN or
credentials in child arguments, is killed/reaped on timeout, and has a parent-death
watchdog. Original hostname and TLS settings survive hostaddr preparation. Explicit
hostaddr remains supported; absent hostaddr is refreshed per physical connection.

**Installation handoff to S06:** these advisory slots bound admitted market sessions,
not simultaneous authentication handshakes or other PostgreSQL gateways. The declared
installation allocation remains market16 + control24 + administrative8 = 48. The
S05 disposable PG has `max_connections=48`; saturation tests hold market16/control24
and check remaining admin access. This does not establish a complete production
physical-connection cap. S06 must enforce and verify a topology covering API thread
concurrency, every worker/replica, idle pools, recovery/lease sessions, migrations,
monitoring/backup and concurrent handshakes. In particular the existing control
`Psycopg*Gateway` classes still open per-call connections and are not constrained by
market advisory slots. Do not count them as pooled or claim production activation
before that installation gate. Embedded/external deployment DNS and role configuration
also remain S06 proof; localhost TLS and resolver lifecycle are S05 proof.

## Import and verification

Application roles receive no partition or invalidation DDL. Maintenance pre-creates
current/next months and every requested historical import month through the bounded
schema capability. Missing partitions fail rather than accumulating in a default
partition. REST/Parquet use-case write chunks are capped at8192; reference/funding
writes and reference lookups use1000. Research export transport converts bounded
chunks into anonymous read-only memmaps; it does not materialize a Python history
list. The caller owns final finite compute arrays. Scanner/file/memmap lifetimes
remain explicit in adapters and native freeze/export paths.

REST fill acquires a verification fence before contacting the provider, renews while
consuming rows, and confirms/release with that fence. Only a fully consumed successful
response can certify missing candles. A known ready provider history origin may
advance the frontier; without it absence evidence is stored but no origin is invented.
A provider call exceeding its lease fails closed at confirmation.

## Source revisions and publication

`SourceTarget` requires generation UUID, content revision and funding revision.
Observation/queued/frozen/receipt comparisons retain all three. Equal bounds are not
content identity. Migration0039 renames old repair labels/errors in place, errors
active legacy targets, quarantines legacy builds and clears old source projections.
It preserves audit IDs/timestamps and applied migration bytes. Legacy targets are
also rejected by validation and dispatch after migration.

Before switching the physical pointer, publication holds a shared generation lock,
compares current revisions, and commits its ready projection through the same control
gateway transaction. It does not hold the lock through native compute. A correction,
funding change or generation reset during compute prevents publication. Unpublished
candidates can be cleaned under exact dead-owner ownership even after source changes;
active interrupted publication requires its recorded source identity. Recovery cannot
restore ready from an obsolete receipt. PG-backed publication forces a full rebuild
because older file manifests do not certify a PG revision tuple for prefix reuse.

Repair domain labels are `canonical_store`; operational storage errors use stable
`storage_*` codes derived from PostgreSQL SQLSTATEs. Consumers must roll out together:
legacy target acceptance, Python repair-policy keyword names and research CLI names
are deliberately incompatible. Public market/reference response shapes and the
accepted platform client's loading and authorization contracts are unchanged.
