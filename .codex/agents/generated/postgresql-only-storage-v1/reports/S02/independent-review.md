# S02 independent production-risk review

Reviewer: independent delegated reviewer `s02_review`. Scope: new 0036 DDL/manifest phase, 0036 bootstrap/status integration, market-data grant/provision helper, new real-PG schema tests, unit manifest expectation. Existing unrelated dirty changes and 0035 implementation were excluded. Product files were not edited. No recursive review.

## Findings

- **P2 — application can inherit maintenance capabilities** (high confidence): `apps/migrations/market_data.py`, `provision_market_data_access`, role validation preceding grants. Distinct unprivileged app and maintenance roles are accepted even when app is a member of maintenance. Granting EXECUTE to maintenance then gives app access to SECURITY DEFINER partition creation and destructive metadata invalidation (through inherited membership or SET ROLE), contrary to the helper's stated separation. Reject application membership in maintenance, including transitive membership, before grants; add a real PG regression case. The existing privilege test covers two unrelated roles only.

## Reviewed boundaries

- D2 table families and typed raw/funding fields are represented; source nullability has a real-PG comparison test. Canonical PK includes monthly partition key, timestamps are timestamptz(3), values preserve double precision. S03 write/version semantics are intentionally not implemented or claimed by this stage.
- Generation UUID and separate candle/funding revisions are durable. Coverage/frontier/absence rows reference generation. Invalidation locks the generation row, fences stale expected generations, clears dependent metadata and allocates a new generation. The current implementation leaves actual data available for later bounded rebuild, as documented.
- Persisted artifact state gains nullable generation/revision/version columns, preserving legacy writers; S05 must interpret NULL as legacy/unverified and roll out serializers and target comparison together. No S05 implementation is required for this review.
- 0036 DDL and checksum marker share a transaction. Separate partition transaction is retryable after committed schema migration. Migration SQL is additive; existing migration bytes were compared to the recorded S02 baseline. Fixed identifiers, schema-qualified relations, pg_catalog-first definer search_path, explicit PUBLIC revocation and bounded month count mitigate DDL privilege abuse.
- New test suite covers repeated bootstrap, checksum drift, transaction termination/resume, historical/year rollover and concurrent provisioning, distinct app/maintenance roles, non-superuser migration owner, and durable revision after server restart.

## Scope, verification and limits

No scope creep found within the supplied S02 changes. No other material P1/P2 finding from static inspection. This review does not establish production cutover, provider ingestion, query performance, S03 transactional revision correctness or S05 artifact serialization parity. Real PG16 tests are being executed by the owning executor; this reviewer inspected test intent but did not independently run infrastructure tests. Acceptance requires their actual successful output. The P2 code defect is closed by the follow-up review below. Existing dirty changes are not endorsed by this review.

## Follow-up review of correction

Re-read the changed grant helper and regression test after executor correction. The helper now rejects `pg_has_role(application_role, maintenance_role, 'MEMBER')` before issuing grants, including transitive membership, and checks membership of privileged principals/schema owner as well as migration ownership. The real-PG test explicitly grants maintenance to app, expects rejection, revokes membership and verifies normal provisioning. **P2 resolved in code; no outstanding material review finding.** The regression test run remains executor-owned evidence; this review does not fabricate its result.

## Final narrow follow-up before receipt

Reviewed `market_data_schema_ready`, its `_postgres_schema_status` integration, and the added typed-constraint/readiness tests. Readiness now checks all fourteen required market relations and attached current/next UTC monthly partitions for all six partitioned tables through `pg_inherits`; it is read-only. The test drops a disposable current partition, observes readiness rejection, and bootstraps to restore readiness. Additional tests exercise UTC timestamp equivalence, canonical primary-key uniqueness/minute alignment, unsigned count bounds, stable market IDs, funding interval/default/null semantics, and persisted artifact source-generation/revision fields. No new material finding.

Read executor-produced `real-pg-verified.log`: **10 passed in 5.32s**. This is inspected executor evidence, not a second independent infrastructure run. The previously reported membership P2 is resolved; there are no outstanding material findings in the reviewed S02 scope. Production cutover, ingestion atomicity and query-performance remain outside this stage proof boundary.
