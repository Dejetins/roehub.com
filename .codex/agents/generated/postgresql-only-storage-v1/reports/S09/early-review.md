# Independent early review — /root/s09_review

Read-only review of setup/load and then full harness source. No recursive reviewer.

Initial findings: full output parity absent (P1), runtime configuration claims declarative,
recovery lifecycle exception could disappear (P1), bounded readers checked cardinality only
(P1), coordinator fence not called (P2), eventual drain confused with in-window cadence,
and missing-lane offered count hardcoded (P2).

Corrections before long runs: independent normalized 17-field source/output comparisons and
byte hashes plus full independent arrays; saved Docker resource/mount snapshot; guarded
lifecycle with mandatory lanes; full expected latest/catalog/gap hashes; exact HTTP response;
fence before every mutation; end timestamps/in-window counts/queue lag; correct missing-lane
interval and explicit collector failure assertions. Active-symbol read/write lane added.

Reviewer observed Docker PG2CPU/4GiB with no registry mount, separate Linux client registry.
Reviewer found no semantic defect in writer min/max pruning: winners are a subset of the
already normalized batch, inclusive bounds cannot remove a matching canonical key. Version,
locks/upsert/revision semantics remain unchanged. Rollover/old-correction/replay test covers
changed risk. Profile magnitude was not accepted as a full workload speedup claim.

This is early source review only. Full runtime/evidence acceptance remains pending.

Hash review follow-up: reviewer independently repeated chunk/layout tests (3 passed), found no actionable CanonicalHasher defect, but demonstrated Arrow timestamp[ms] can silently truncate a PostgreSQL +1 microsecond divergence before hashing (P2). Added pre-cast exact-millisecond rejection and a regression reproducing that blind spot (4 tests passed). A direct full-database timestamp precision check accompanies parity attempt3, which started before this source fix. Runtime acceptance remains pending.

Bounded verifier follow-up: reviewer confirmed exact89parts/15130115rows/102instruments, companion precision SQL, and unchanged full elementwise/hash artifact comparison after monthly7-column disk-backed expected preparation. No new findings; failedOOM/timezone attempts retained.

Binary array-reader tuning review: reviewer compared canonical-reader-before.py with current source and found no actionable precision/order/range/retry/memory issue. int64 timestamps avoid float detours, conversion remains4096 rows, same SQL/transaction/truncate-on-retry/empty/memmap lifecycle. Read23passed reader regression log and matching before/after full array hashes. Dedicated interrupted-mid-batch transport fault proof not yet observed; source retry cleanup is preserved. API wiring now uses actual persisted PostgresResearchOrganizationScopeResolver. No runtime performance acceptance claimed.

Final pre-long-run review: no open actionable findings in current harness corrections. Deferred input hashes retain same cadence and per-batch17field proof; pilot5 all12lanes offered/completed/in-window equal, stream6000canonical/6001raw and history30000/30000. Exact UTC workload and verification markers separate resource intervals; analysis records5-second sampling bounds. CLI local-variable finding was caught and fixed before long runs, focusedruff/pyright pass. Actual backend termination after4096rows regression restoredall5000 without partialprefix;24passed. Reconstructed initial contract labelled honestly; active contract equals frozenv2. Review is harness readiness, not S09 acceptance.

Stage reuse tuning review: no actionable semantic/atomicity findings. Existing candle_stage has identical normalized columns/payload_order and is not mutated before upsert. Unchanged DISTINCT/version-order/strict tuple comparison, quoted identifiers, generation locks/raw/coverage/revision/rollback/retry boundaries. Other upsert consumers preserve prior behavior.25passed transactional log includes funding/reference clocks/nulls/ties, concurrency and unknowncommit. Reviewer requires full new matrix, failedmatrix1 kept separate.

Completed six-run independent evidence review: reviewer independently checked13320raw samples, unique/complete lane indices, due/start/end validity, all successful completionswithin300s, recomputedmedian/p95 equal summaries. All6source sets36paths identical and current.90year+90full perregime match frozenexpected hashes. Every write receipt offset/count is contiguous and bound to fixtureversion, hashes and canonical/raw counts matchoffered plus1streamseed.60PGresource samples/run, maxgap<=5.060s, zero whole-driver server/clientOOMdelta, observed registry peaks9market/4control/2admin. Eachrun>=1timedcheckpoint and9–14autovacuums, WALdirectory~400–416MiB while retainedrelationsgrow~116–138MiB/run. Allper-run guardrails pass; no newactionablefinding. Warmfilesystem/syntheticcontrolfixture/5-second sampledpeak limits remain; finalsoak and finalreceipt verification still pending. Runtime was not touched byreviewer.
