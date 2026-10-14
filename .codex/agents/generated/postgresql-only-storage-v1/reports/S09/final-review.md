# Independent S09 review — /root/s09_review

## Historical revision-3 review — superseded by failed memory verdict

Read-only independent review; no recursive review. Production tuning, exact hashes, failed-attempt handling and all six 300-second runs were reviewed in earlier rounds (see early-review.md).

Final raw soak audit confirms 28,440 unique and complete samples, every offered operation completed in-window without errors; recomputed per-lane/pooled quantiles match. All 360 year and 360 full hashes match the frozen independent target. Stream: 3600 batches, 360,000 canonical / 360,001 raw; history: 1800 batches, 1,800,000 canonical/raw. Offsets and per-batch/aggregate hashes reconcile. Same 36 measured source hashes across seven runs and current files.

Resource audit confirms zero whole-driver OOM/kill deltas, 716 in-window PG samples, maximum sample gaps 5.131 s (PG) / 5.200 s (server), 12 timed checkpoints and 256 autovacuums. WAL directory is 419,430,400 bytes throughout. Relation growth 1,598,922,752 bytes is dominated by retained canonical/raw data. No accumulating queue. Report table/counts/handoff match evidence. Independently checked 325 protected migration/predecessor report/receipt files unchanged; final ruff, pyright and 29+24 test logs pass.

Memory finding: server anonymous quarter medians increase 246,161,408 to 299,749,376 bytes despite stable aggregate cgroup memory. Client RSS rises consistently with retained sample records. Closing sessions releases anonymous memory, but does not establish a steady-state plateau or exact cause. Flat-memory claim removed. Reviewer considered the measured-hour result acceptable with explicit limits; executor retained acceptance pending a focused memory-context replay to meet the stricter no-unexplained-growth criterion. No production/config change for this diagnostic.

Historical pending item resolved by the revision-4 causal fix and reviews below.

## Revision 4 causal fix and pre-matrix review

The 8000-write control on one partition and a fixed 42-partition set confirmed operation-count-dependent CacheMemoryContext growth (about 8 KiB/write), with fixed prepared-plan/index counts and plateaued local buffers. Replacing temp relation recreation with stable session-local relations in an isolated ablation eliminated that slope. Direct production probes then confirmed CacheMemoryContext used bytes 7,513,608 unchanged from operation 101 through 8000 on one partition, and 7,917,464 → 7,917,472 on the fixed 42-partition set.

Independent reviewer checked the final production source: generic stage reuse remains false, reused raw names bind source tables, all reused access is pg_temp-qualified, entry TRUNCATE and ON COMMIT DELETE ROWS prevent carryover, transactional CREATE/rollback and unchanged gateway close/retry behavior preserve atomicity. First CREATE abort/recreate, later rollback/empty/OID identity, cache regression and existing winner/unknown-commit tests pass (55 passed). No actionable findings before the new r4 matrix. Old final-* attempts remain FAILED for memory and are never pooled with r4. Funding stage behavior is unchanged and outside this candle load claim.

This pre-matrix pending item is resolved by the final revision-4 runtime and report reviews below.

## Final revision-4 runtime review

Independent audit confirms 13,320 short-run samples and 28,440 soak samples, unique/complete indices, all operations completed within windows without errors, and recomputed quantiles matching summaries. Every year/full hash matches the frozen target; all stream/history offsets, 17-field batch receipts, aggregate hashes and raw/canonical counts reconcile. The aggregate contains only r4 runs and all seven source manifests match current source.

Soak resources: zero server/client OOM and kill deltas, 716 PG / 705 server in-window samples, maximum gaps 5.064 / 5.302 seconds, 12 timed checkpoints and 198 autovacuums. WAL generated 2,153,423,657 bytes; directory 771,751,936→503,316,480. Retained relation growth 1,726,095,360 bytes is primarily canonical/raw data. No accumulating queue. Every latency guardrail passes.

Memory verdict: the same 12 backends (PID/backend_start/advisory slots) were idle at comparable 30/55-minute snapshots while operations continued between them. Writer and bulk Grand-used contexts decrease; monitor total remains stable. Largest bulk RssAnon remains 106,924 KiB; the second takes one approximately 1.4 MiB step with decreasing live contexts. Five-minute server anonymous medians are 217.98, 229.27, 240.82, 240.36, 240.18, 240.20, 240.21, 241.20, 241.20, 241.08, 241.55, 241.57 MiB. Combined with direct 8000-write plateau probes, this closes the known operation-count growth for the stated observed-hour workload. It supports bounded warmup/high-water, not an indefinite leak-free guarantee. Diagnostic UTC/durations and every affected sample are retained.

No remaining actionable source/runtime/evidence findings. Final report cross-check completed: all r4 metric/count/resource values, owned-file references, failure history, causal fix, instrumentation, test results and S10 handoff match evidence. Reviewer final verdict: no remaining actionable findings within the stated S09 hardware/dataset/mix/hour boundary. Canonical status/receipt remains executor-owned.
