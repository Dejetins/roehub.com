# Independent S03 review

Reviewer: delegated read-only agent `/root/review_s03`, using production-risk-review.
Scope: new ingestion/common/gateway/candle/funding/reference code, additive 0037 and
bootstrap/readiness, committed publication, recovery slot, new real-PG fixtures.
Reviewer inspected current code and installed psycopg initialization source. It did
not independently execute PG tests; execution logs are the implementing agent's proof.
No recursive reviewer was requested and reviewer edited no files.

Findings and resolution:

1. P1: staging LIKE copied NOT NULL without defaults, breaking partial reference
   COPY after 0037. Added INCLUDING DEFAULTS; status/enrichment real tests pass.
2. P2: slot acquisition was outside watchdog. Covered it by remaining deadline.
   Follow-up found blocking connection setup still exceeded short/late deadlines.
   Replaced it by public PGconn.connect_start/connect_poll with select and absolute
   stop; cleanup finishes failed connections. Prepared hostaddr prevents blocking
   DNS in this seam. Real stalled-handshake fixture proves the timeout path.
3. P2: shared 8192 limit exceeded Q10/Q11/Q13/Q14. Added market4 and other reference/
   funding1000 limits; raw uses8192/8MiB, whichever first.
4. Proof gap: all-NULL/equal-order reference/universe and provider-only funding ties.
   Added opposite arrival permutations, independent stale enrichment/status clocks,
   authoritative all-NULL and equality of raw/canonical funding tokens.

Final reviewer result: findings closed, no new mandatory fixes. The reviewer compared
_connect_until with Connection.connect/_connect_gen, generators._connect and
BaseConnection.__init__: nonblocking, default tuple factory, adapters/cursors/locks,
prepare threshold, notice/notify handlers and autocommit initialization are preserved.
An explicit tuple_row argument was then added for static typing without changing this
reviewed default behavior. A subsequent real container restart fixture strengthens
failure proof; it changes no product behavior.

Limits retained by review: global physical connection cap and hostname DNS/TLS/
re-resolution are required S05/S06 integration obligations. Advisory admission alone
is not physical cap proof. Independent review does not imply deployment or performance
readiness. Final required lint/type/unit/real checks are recorded separately.
