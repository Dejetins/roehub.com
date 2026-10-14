# Local market-data recovery — PostgreSQL-only source

This runbook describes the current repository implementation. It does not certify
that an existing local installation has been switched. S10 discovers the exact
installation and executes its authorized cutover; do not reset or adopt an unknown target.
The earlier accumulated operational journal is preserved byte-for-byte in S07
`reports/S07/doc-baseline/docs/runbooks/navigator-local-market-data-recovery.md.evidence`.

## Diagnose before changing state

Use the generated operational manifest and read-only health/events interfaces.
Inspect ingestion lag, recovery backlog, artifact queue, PostgreSQL connections,
reservation/quarantine counts, lock waits, long transactions, WAL/checkpoints,
dead tuples and autovacuum. Keep DSNs, credentials and raw provider payloads private.
A reachable port alone is not proof of canonical coverage or a successful commit.

Canonical data and recovery metadata share PostgreSQL. A failed commit must not
advance the recovery cursor or trigger successful Redis publication. SQLSTATE
resource/deadlock/serialization/disconnect failures use bounded retry; permission,
authentication and schema failures fail visibly. Unknown outcomes remain quarantined.

Actual minutes, accepted provider absence and unknown coverage are distinct.
Missing coverage must fail closed. Normal reads use compact day masks and summaries;
explicit generation-fenced maintenance rebuilds a bounded range. A new generation
invalidates verified frontiers and materialized source targets. Preserve accounts,
settings, instrument selections, credentials, strategies and user results.

## Connection and operator recovery

Follow [storage bootstrap and migrations](storage-bootstrap-and-migrations.md)
and [backup/restore](installation-backup-restore-upgrade.md). The protected shared
reservation registry lives with Linux clients, not on external PostgreSQL.
External host, port, database, credentials_ref and TLS remain explicit.
Admission precedes the socket/handshake; installation budgets are 16 market,
24 control and 8 admin physical attempts. Do not reclaim a crashed owner merely
because its client process disappeared or a query no longer appears in a snapshot.

The verified recovery adapter targets one enrolled remote Docker PostgreSQL server:
all-client barrier, identity verification, server stop, reservation reset and counted
resume. It needs no shared server volume. Other admin control planes require an
equivalent verified adapter. The proof assumes a stable direct PG16 endpoint, Linux,
exclusive trusted operator control and a protected registry. Transparent proxies,
failover and same-address reassignment are not proven. Native utilities use verified
remote TLS with trusted local loopback; do not generalize that proof to arbitrary
hosted PostgreSQL recovery.

The async collector's PostgreSQL session sets server-side TCP keepalive idle/interval/
count to 10/5/3 and TCP user timeout to 15000 ms on the supported Linux PG16 target.
This bounds dead-peer detection after a lost network drops the client's final close;
the server still exclusively owns and releases the advisory lease. A healthy idle
connection remains eligible. These settings do not authorize clearing quarantined
reservations or taking over a still-live lease. Old orphan sessions created before
the change require the verified operator recovery above; keepalive timing is not a
universal exact recovery deadline.

## Writers and artifacts

Drain/quiesce through the current control policy before a disruptive operation.
Preserve pending obligations and request snapshots. Restore or maintenance must
invalidate generation/frontier state before claiming completeness. Source revision
includes content and funding corrections even when bounds/counts are unchanged.
The pre-provider lease/generation fence and final artifact publication fence remain
mandatory. Resource refusal before publication is retryable; an unknown publication
outcome is quarantined for receipt/pointer reconciliation.

Use the accepted `apps/platform-web` client for product proof; Navigator remains a
candidate. Source checks and disposable tests do not certify installed delivery.

## Current local cutover evidence

The S10 record in
`.codex/agents/generated/postgresql-only-storage-v1/reports/S10.md` owns the exact
local target identities, original blocked status and preservation proof. The authorized
replacement `.codex/agents/generated/postgresql-only-storage-v1/reports/S10-r1.md`
records subsequent provider repair, selected live history, reconnect, frontier and
artifact/backtest evidence. Keep the original S10 report and receipts immutable.
Runtime migration and live-provider acceptance are separate facts.
Binance HTTP 451 and Bybit HTTP 403 in the original attempt were provider access
failures; the current
Bybit adapter conservatively labels 403 as rate limited, which is not evidence of
an HTTP 429 quota response. Do not bypass those restrictions or silently change
selected instruments to obtain a green live result.

A bounded import from an immutable, previously validated history file can rebuild
canonical rows and fresh generation coverage. Record its original range and exact
value hash; retain original metadata timestamps and do not publish a current catalog
from a partial retained snapshot. Such import proves historical recovery only.
Live acceptance still needs actual selected-source closed minutes and controlled
reconnect with agreeing SQL, API, event and browser observations.

Before a market-only reset, preserve account/settings/selections, create a scoped
recoverable encrypted backup and verify it in a separate target. Keep the decryption
key separately, directory mode 0700 and key/private files mode 0600. Archive current
artifact pointers before invalidating source generations; retain prior slots and
user results. An old CH volume may remain inert for rollback; it must not be attached
to an active fallback. A scoped legacy snapshot is not a seven-owner release backup.

A pointer-only reset is insufficient when retained slot manifests would fail the
normal publication precheck. Under a quiesced, verified zero-reader/zero-writer
barrier, archive the entire old coordinate directory, including both slots and
journals, and record every relative path and content hash in a reset amendment.
Clear generation/hash metadata only on available slot rows; preserve writer epochs.
Reconcile a proven pre-publication refusal with a terminal reason and events before
requesting a new build. Never clear an unknown publication outcome blindly.
Prior backtest result rows remain readable; archived input generations are explicitly
retired and must not be presented as current inputs. Keep the initial reset manifest
immutable and link the amendment and verified encrypted recovery snapshot.

Frozen-range publication and verified-prefix preservation follow the current
[append certificate addendum](../architecture/market_data/market-data-append-certificate-v1.md).
