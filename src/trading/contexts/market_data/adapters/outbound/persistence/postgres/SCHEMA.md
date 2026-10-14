# Market storage lifecycle (S02)

Migration `0036_market_data_storage_v1.sql` introduces PG16 storage alongside the
existing runtime. It does not switch composition roots or migrate/delete history.
Use `apps.migrations.storage.apply_postgres_migrations`: immutable manifest checks
precede execution; this new phase and its marker commit atomically. Retrying after
an interrupted transaction recreates the phase. Bootstrap provisions current/next
UTC months on every invocation. Readiness checks every required relation and both
months for all six partitioned parents. Existing applied SQL files remain unchanged.

`apps.migrations.market_data.provision_market_data_access` grants existing roles;
it creates no cluster roles. Run as the migration owner. Application and maintenance
roles must differ and cannot inherit ownership/elevated cluster capabilities; app
cannot inherit maintenance. App gets schema USAGE and table DML, with no ownership,
TRUNCATE, DDL or lifecycle EXECUTE. Parent-table privileges cover partition routing;
new partitions do not require direct app grants. Maintenance gets only USAGE and
EXECUTE on two fixed-schema SECURITY DEFINER functions. Function search paths exclude
writable schemas, values are quoted, table names come from a closed list, and PUBLIC
EXECUTE is revoked. Role provisioning must not subsequently add privileged memberships.

Before importing a range, call `market_data.ensure_partitions(first_month, month_count)`
within an explicit transaction with client statement timeout 60 seconds and lock
 timeout 2 seconds (as bootstrap does). One call accepts 1–12 whole months in
1970–9998 and creates at most 72 partitions. Split older ranges into bounded calls.
DDL is serialized by transaction advisory lock and rolls back with the transaction.
There is no default partition and no per-candle DDL. The function returns the number
of ensured partitions, including already existing ones. Function-level settings alone
do not replace the caller's statement deadline.

`instrument_generations` owns a non-reusable UUID plus durable candle and funding
revisions per key. Reconnection/server restart preserves them. S03 must lock the key
and atomically update revisions with canonical changes; replay/metadata-only changes
must not increment them. Coverage days, summary, absence observations and verification
have generation foreign keys. Days enforce disjoint actual/absence masks and derive
counts with bit_count. Summary stores actual bounds/count only; revision is joined
from the generation row in the same read snapshot. Missing/unknown index state is
not complete history. S03/S04 own the transaction algorithms, policy and lease checks.

Maintenance-only `invalidate_instrument(market, symbol, expected_generation, reason)`
locks/fences one key, clears its derived coverage/absence/frontier, allocates a fresh
UUID and marks the index unknown. It does not delete raw/canonical data or public
control state. Call it before explicit deletion/restore/rebuild under quiescence;
S03 writers and S04 readers must honor the generation/index state and stable lock order.
This can touch all metadata for one instrument; it is an administrative operation,
not a request hot path. Use an outer statement timeout and resumable rebuild, never
claim that an invalidated key is ready before rebuilding from retained data.

Existing `public.backtest_artifact_source_state` gets nullable source_generation,
content_revision, funding_revision and source_contract_version=2 columns. These are
the persisted desired-source observation watermark; NULL means legacy/unverified,
not zero/current. S05 must write the complete tuple with desired_target atomically,
carry version/generation/revisions in published target serialization, and compare
all fields before reusing inputs. The additive columns keep old writers bootable
until the coordinated switch; they alone do not make old serializers revision-aware.

All original typed provider/reference/funding fields are preserved, with Float64
mapped to double precision, milliseconds to timestamptz(3), UUID nullable, UInt32
counts mapped to checked BIGINT, and Bybit milliseconds checked against UTC value.
Raw Bybit category defaults to non-null `linear`; Binance mark_price is nullable.
Canonical funding intentionally has no mark_price/category. `payload_order` is the
S03 deterministic normalized byte-order slot, not a hash-based identity. S03 must
supply it for version-aware upserts; an empty default permits additive bootstrap.
