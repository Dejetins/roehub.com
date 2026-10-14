# S01 independent architecture/source-audit review

Reviewer: separate `s01_review` agent, 2026-10-13. Skill: architecture-review.
Scope: S01 documentation and source evidence only. No runtime/schema activation,
release readiness or later-stage pass is established. No recursive review.

## Verdict

**Ready** for the S01 documentation/source-audit boundary after narrow
recheck and closure of R1 and R2 below. The accepted PostgreSQL-only direction is executable;
these are narrow audit/contract gaps, not grounds for replanning the pack.

## Observed state and checks

- Read AGENTS.md, selected S01 prompt, implementation-plan.md, architecture entry
  and map guide, target storage document, query catalogue and JSON evidence.
- Inspected current ClickHouse DDL, funding adapter and reference writer, plus
  related PG migrations for reset and source-target state.
- Independently checked the initial 1,254 inventory rows: every row has category, owner and
  disposition. This validates recorded rows, not completeness by itself.
- All 11 ClickHouse tables are represented, including 17 canonical fields,
  provider raw funding fields and absence state. Column lists preserve defaults
  and nullable types.
- Recomputed all 219 recorded method body signatures: no source drift found.
  Every public concrete ClickHouse adapter method has a catalogue mapping.
- Normative document distinguishes current CH source from accepted PG target;
  it covers D1–D7, transactional receipts, contiguous verified frontier,
  durable content revision, scoped reset and immutable applied migrations.
- Resource limits are evaluation choices, not measurements. Mixed control work,
  long soak, result hashes and client memory remain mandatory future proof.
- Local `python` was unavailable; read-only audit used `python3`. One exploratory
  read used a nonexistent `canonical_candles_ddl.sql`; resolved through actual
  `market_data_ddl.sql`. Neither error is evidence of absent schema.

## Findings

### R1 — Medium / Gap: dependent PostgreSQL DDL absent from inventory

Observed: `inventory.json` records `0023_market_data_work_requests_v1.sql`, but
omits `0024_market_data_full_history_v1.sql`, `0027_market_data_stream_recovery_v1.sql`,
`0029_backtest_artifact_management_v1.sql` and `0034_market_data_provider_admission_v1.sql`.
These define/alter the history bounds, recovery and artifact source state used
by the reset/version contract. `audit_sources.py` computes literal matches and
Python import closure; SQL dependencies with no CH string are not followed.

Inference: later additive migration/reset owners can miss affected columns,
constraints or preserve-only state even though the audit claims complete schema
coverage. Contract: persistence, generation/recovery state and checksum ownership.

Minimal correction: inventory the relevant PG migration/manifest chain and
schema fields or exact DDL references reached through the affected repositories;
classify applied bytes as immutable and name additive successor/reset owners.
Verify references to history/catalog/recovery/repair/artifact source-target tables
resolve to hashed DDL entries. Do not modify applied migrations.

Sources: `migrations/postgres/0024_market_data_full_history_v1.sql:15`,
`0027_market_data_stream_recovery_v1.sql:2`,
`0029_backtest_artifact_management_v1.sql:115`,
`0034_market_data_provider_admission_v1.sql:33`; target document reset table.

### R2 — Medium / Partial: nullable funding selection parity needs a fixed case

Observed: `ClickHouseFundingRateStore.list_tradable_funding_instruments` selects
nullable metadata through separate `argMax(field, updated_at)` expressions;
the target says version-aware upsert and applies a whole normalized payload
ordering to reference updates. Q13 mentions nullable metadata but does not state
how old non-null -> newer NULL and all-NULL histories are classified. The current
ReplacingMergeTree merge state also needs consideration when selecting a stable
baseline; S01 has not executed that experiment.

Inference: a straightforward latest-row PG replacement can silently change
observable reference metadata while `none` output parity is expected. Contract:
funding universe read/DTO semantics, not merely persisted column types.

Minimal correction: record this observed fieldwise baseline explicitly and add
old non-null -> newer NULL, all-NULL and equal-version cases to S03/S04 proof.
Select a deterministic target/null policy; label any intentional tightening and
require CH fixture observations before claiming parity. No owner reapproval of
PG-only is needed. Verify fixture expectations against actual source semantics.

Source: `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py:49-65`;
query catalogue Q13; target document reversible implementation decisions.

Minor consistency correction with R2: D2 says “nullable mark price/category”,
whereas the exact field section correctly says Bybit category is NOT NULL and
adapter defaults it to `linear`. Make the D2 wording agree with that refinement.

## Assessment

| Area | Status | Boundary |
|---|---|---|
| Accepted target, D1–D7, source versus target | OK | Documentation |
| Raw/canonical/funding field preservation | OK | Existing CH schema |
| Ports/caller and query mapping | OK | Recorded source signatures/import closure |
| PG dependent schemas | OK after correction | R1 closed |
| NULL/version selection semantics | OK after correction | R2 closed |
| Global resources and measured-baseline limits | OK | Future S03/S09 proof |
| Versioned backup/config/repair contracts | OK | Coordinated S05/S06 cutover |
| Scoped reset/generation/rollback | OK | S08 disposable proof then S10 exact manifest |
| Real DB/provider/browser/performance readiness | N/A | Outside S01 proof |

Private/external scripts remain unknown consumers, as explicitly recorded.
No product code, schema, ledger or publication state was changed by this review.

## Resolution recheck

Both findings are closed for S01 after author corrections:

- R1: refreshed inventory contains 1,304 classified paths and 54 schema
  dependencies. The audit now follows table references from consumer code into
  CREATE/ALTER/REFERENCES DDL and includes migration manifests. Independently
  verified presence, immutable-byte disposition, S02/S05 ownership and current
  SHA-256 for PG 0024, 0027, 0029, 0034 and manifest. Recorded DDL references
  make later additive/reset work traceable without modifying existing migrations.
- R2: target document now explicitly chooses authoritative newer NULL for
  funding-universe metadata, classifies reliance on stale non-null carry-forward
  as `breaking-change`, distinguishes before/after merge baseline uncertainty,
  and requires non-null -> NULL, all-NULL and equal-version fixtures. It also
  preserves omitted enrichment fields and corrects D2 to non-null Bybit category.
  Runtime parity remains a future obligation; no CH fixture result was fabricated.

No remaining material finding in this bounded review. Acceptance still requires
executor-owned structural gates and evidence/report/receipt checks. This review
is not a receipt, implementation test pass, or runtime readiness verdict.

## Final import-closure refinement recheck

The same reviewer performed a narrow recheck after relative `ImportFrom` and
`from package import submodule` resolution were added. No new review scope.

- Current evidence: 1,520 inventory paths, 481 closure paths, 1,371 caller edges,
  58 schema dependencies, 11 tables and 219 method signatures; no recorded errors.
- Executed the audit's read-only prefix with the previous import-resolution logic
  in memory, stopping before any evidence writes. It reconstructed exactly the
  previous 1,304 inventory paths and 552 edges; none are absent from the new set.
- Compared the new relative-module computation with Python's
  `importlib.util.resolve_name` for all 819 relative imports in the audited source
  universe: zero mismatches. Module alias handling uses the imported `alias.name`,
  not the local `asname`, and retains the parent-package edge.
- All current inventory rows still have category, owner and disposition.

Verdict remains **Ready** for S01 documentation/source audit. Static import closure
is conservative evidence; it does not prove dynamic imports, private external
scripts or runtime behavior. No ledger or implementation file was changed.
