# PostgreSQL storage — current source status

Status: current source documentation, 2026-10-14. The accepted
[storage design contract](market-data-real-adapters-postgresql-parquet-v1.md) is
frozen S01 evidence bound by its immutable receipt. Its opening implementation
status describes 2026-10-13, not the current source tree. D1–D7 and the accepted
compatibility decisions remain authoritative; this page updates execution status.

S02–S06 implemented PostgreSQL schema, atomic ingestion, bounded readers,
generation/revision fencing, consumers and installation/backup/observability.
S07 removes obsolete executable adapters, DDL, driver and maintenance paths,
ports meaningful tests and reconciles active documentation. Exact acceptance
state and proof are in the selected [journal](../../../.codex/agents/generated/postgresql-only-storage-v1/iteration-ledger.md).

Current source boundaries:

- Market tables use the dedicated `market_data` schema in the same PostgreSQL 16
  database as control metadata. Applied SQL0001–0039 stays byte-identical.
- Raw/canonical/coverage/revision commit atomically; canonical PKs and stable
  version precedence replace asynchronous duplicate removal. Monthly partitions
  coexist with bounded row/byte/time windows.
- Day masks and summaries distinguish actual candles, accepted source absence,
  unknown index state and verified completeness. Actual coverage percent remains
  actual candles divided by calendar minutes; completeness may include accepted
  absences without fabricating prices, advancing actual latest or increasing counts.
- Artifact source generation/content/funding revisions are fenced through final
  publication. Redis/provider/organization/security contracts are preserved.
- `STRATEGY_PG_DSN` is explicit; trusted workload pools and the shared registry
  bound handshakes/connections. Old configuration and backup versions fail before
  writes; no fallback engine is loaded.
- Current [recovery runbook](../../runbooks/navigator-local-market-data-recovery.md)
  owns diagnostics and shared-registry recovery limits. The
  [reference API contract](market-data-reference-api-v1.md) owns public semantics.

Repository and disposable tests do not prove installed runtime delivery. S08 owns
full functional/recovery verification, S09 performance/soak, and S10 exact local
cutover/reset proof. No production target, deployment, volume deletion or provider
refill is authorized by this status page. Measured prototype bytes and prior
accepted reports/receipts remain immutable historical evidence.
