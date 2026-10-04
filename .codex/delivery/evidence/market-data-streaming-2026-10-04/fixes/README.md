# Streaming failure repairs — 2026-10-04

## Scope and verdict

Authorized repair of F1–F3 in [the preceding investigation](../README.md), for Binance Spot AAVEUSDT, ADAUSDT, ALGOUSDT and ANKRUSDT. Local Navigator candidate only. No publication, branch, worktree, Goal, platform-web changes or system/Docker memory changes in this repair.

**The three reproduced failure cases are repaired and the focused verification passed.** This is not a production readiness verdict, long-duration reliability claim, Redis proof or proof for all providers. The predecessor report remains an accurate record of the original failures.

## Changes

- REST repair work now survives exhausted HTTP retries: bounded durable PostgreSQL journal, classified backoff (15/30/60/120/240/300 seconds, honoring greater Retry-After), cooldown without occupying a slot, and 60-minute confirmed checkpoints. Permanent errors remain failed and need explicit rearm. A successful empty response cannot acknowledge a received WS minute.
- Raw insertion now has a hard accepted-row bound and producer backpressure. Each received minute gets a durable repair receipt before acceptance; acknowledgment follows successful INSERT. Cancelled blocked intake hands its receipt to the live repair queue. Terminal failures stop ingestion; incomplete close raises `UnflushedBufferError` instead of reporting success. First-arrival batching avoids splitting a small burst on an unrelated timer tick.
- Production WS synchronous REST/ClickHouse work runs through bounded, cancellable spawned processes around the existing adapters/use case. Cancellation terminates and joins children; parent death has a watchdog. One raw writer, configured REST concurrency, one planner, one collector lease. WS close timeout is two seconds. The legacy scheduler sync executor does not gain a false forced-thread-cancellation guarantee.

Owned implementation: WS wiring and `io_process.py`; `insert_buffer.py`, `rest_fill_queue.py`, `ingestion_retry.py`; recovery store port/PostgreSQL adapter; migration 0027 with bootstrap/storage/manifest registration; Binance/Bybit WS close timeout; regression tests; updated WS operational contract. Generated documentation indexes are refreshed from the current workspace without changing their source catalogs or manual text.

## Real exchange, PostgreSQL and ClickHouse proof

Harness: `.venv/bin/python .local_artifacts/navigator-preview-sep26/streaming_fixed_probe.py outage`, followed by `recovered`. Both use the current WS composition root and actual Binance/ClickHouse/PostgreSQL. Existing authorized local connection settings stay in memory. Redis streams/hotcache are disabled only in the owned test YAML because no local Redis is available.

- `outage.json`: received four closed candles at 11:54 UTC. A bounded INSERT into an owned disposable table, with per-query `max_memory_usage=1`, produced actual ClickHouse code 241 / `storage_memory_pressure`. Global resources were unchanged. Attempts at 11:54:00.803, 11:54:16.650 and 11:54:47.341 demonstrate increasing cooldown, not a tight retry loop. Four candle receipts survived.
- The same run started an intentionally blocked child REST operation. Stop completed in **2.640 seconds**, raised the expected `UnflushedBufferError`, retained four minute receipts plus the blocked repair receipt, and left no test child. No forced parent cleanup was used.
- `recovered.json`: startup repaired all four received minutes through actual REST and canonical storage, then cleared the retained repair work. The extra AAVE overlap wrote zero rows. It also caught up the closed minute during the stopped interval.
- A second actual code-241 fault at 11:57:00.873 recovered **without restart**: all four candles were written at 11:57:16.498. The journal and buffer returned to empty; no WS error occurred.
- IAB disabled ANKR; the next closed minute wrote only AAVE/ADA/ALGO. Re-enabling ANKR triggered REST catch-up of the missing minute, followed by a four-symbol WS batch. A subsequent gap check of that same minute wrote zero additional rows.
- Healthy stop completed in **2.028 seconds**, no exit error and no remaining receipts/children.
- `coverage.json`: canonical range `[11:49,11:59)` has exactly **10 distinct minutes and 10 canonical rows for each of the four instruments**. All four August histories still have **44,640 distinct minutes**. No owned fault tables or disposable test schemas remain.
- `migration.json`: migration 0027 was applied locally through the existing bootstrap/lifecycle marker. Hash and count of 121 existing request state/progress rows are identical before/after.

The healthy candidate's 11 measured WS rows took **0.719–0.916 seconds** receive-to-insert; four explicitly faulted rows took **16.417–16.509 seconds**. These are distinct samples, not silently filtered failures. The preceding pre-coalescing run had 14/16 healthy rows within one second and two within two seconds. The workloads share the local machine/config but differ in timing and selected-symbol count; no controlled performance improvement or sustained p95 guarantee is claimed. The local ≤1-second target was met by the short final healthy sample only.

## Automated validation

From repository root:

```text
.venv/bin/python -m pytest -q tests/unit/contexts/market_data tests/unit/apps/worker/test_market_data_io_process.py tests/unit/apps/migrations/test_storage_lifecycle.py tests/unit/apps/api/test_market_data_reference_routes.py
220 passed in 4.48s
```

Focused `ruff check` and `pyright` passed on the changed worker, buffer/queue, recovery store/port, migration wiring and new tests (0 type errors). `git diff --check` passed. Both `python -m tools.docs.generate_docs_index --check` and `python -m tools.docs.generate_project_map --check` passed after refreshing the generated indexes (the initial check detected workspace drift). These are focused gates, not a claim that all repository CI ran. Frontend sources were not changed in this repair; existing earlier frontend results remain in the predecessor report.

```text
ROEHUB_MARKET_DATA_TEST_DSN=<existing local target, supplied in memory>
.venv/bin/python -m pytest -q tests/real_infra/market_data/test_stream_recovery_postgres.py
4 passed in 0.61s
```

Real-PG tests use disposable schemas and verify capacity, terminal/rearm behavior, token fencing, persisted retry time, checkpoint identity and collector exclusivity. They require no collector in that database: an intermediate run while the owned collector was running correctly failed its lease acquisition (3 passed / 1 failed). After the collector stopped, the complete suite passed. This was an identified test-environment conflict, not a discarded unexplained failure.

New unit/process checks cover retry-slot fairness and Retry-After, permanent errors, buffer capacity, cancelled intake handoff, uncertain INSERT acknowledgment, empty REST acknowledgment, restart at a confirmed window, bounded cancellation/timeout, parent-death child cleanup and first-arrival batching.

IAB verification used the existing Codex tab only. `ankr-disabled-iab.png`, `ankr-reenabled-iab.png`, `streaming-restored-off-iab.png` retain UI evidence. Browser console: zero warning/error entries in the inspected log. Final selections are empty, matching baseline; the owned collector/metrics listener stopped. Existing request scheduler PID 22265 remained running.

## Compatibility and review

- Public API/DTO, organization/same-origin checks, browser routes: **none**.
- PostgreSQL schema: **compatible-change**, additive migration 0027, existing request records untouched. Apply migration before the new collector. Old binaries ignore pending receipts; drain with the new collector before rollback or explicitly preserve pending recovery.
- Internal buffer call/close semantics: **breaking-change** for callers relying on unbounded synchronous admission or silent close. WS wiring now awaits backpressure and surfaces incomplete close. Existing callers/tests were checked.
- Internal REST queue: **compatible-change** for successful work; classified retry/persistence are added. Legacy sync executor remains supported, with its existing thread limitation explicitly retained.
- Runtime/config: **compatible-change** to YAML shape, but operational prerequisites increase: PostgreSQL journal availability, child-process support, and single collector lease. Journal capacity is 20,000 pending/failed entries; reaching it fails visibly rather than dropping work. No automatic memory sizing or parallelism increase.

Review mode: focused self-review plus one independent review required for data-loss/concurrency risk. Reviewer found two P1 issues (whole-range timeout replay and deferred intake only recovering at restart); both were corrected and re-reviewed. Final independent review found no blockers; final narrow batching/planner review also passed. No recursive review.

Limits: live Redis delivery/strategy consumption, other providers, long outages/soak and arbitrary PostgreSQL TCP blackholes were not proven. PostgreSQL connect/statement/lock timeouts do not establish an absolute wall-clock bound under every network failure. Persistent ClickHouse capacity pressure still needs resource diagnosis; retry is not a capacity fix. A separate supervisor is still needed to run/restart the standalone collector; this task did not install one.
