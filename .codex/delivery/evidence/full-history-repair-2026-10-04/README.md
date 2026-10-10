# Navigator full-history repair — 2026-10-04

## Scope and semantics

Local candidate: `apps/navigator-web`, http://localhost:20120. The accepted
`apps/platform-web` is unchanged. No publication, migrations, candle deletion,
resource/limit changes, new branch/worktree/Goal, or historical Goal transition.

Coverage now uses the confirmed per-instrument origin in PostgreSQL through the
latest closed UTC minute, measured with the existing ClickHouse exact minute
bitmap. Unknown origins remain explicit. The one-day Navigator selector is
removed; table and inspector describe full history with absolute counts and the
missing count. Completed jobs retain their fixed requested interval.

Full-history commands resolve each instrument's own origin and capture a fresh
closed-minute end at submission. Existing runners skip confirmed ranges and REST
reads request only missing runs. Written windows are checked before completion.
A persistent provider omission cannot falsely complete history. Three retries
without confirmed progress end in `source_history_incomplete`. Pause/cancel,
lease fencing, checkpoint continuation and four request workers are retained.

The API and UI reject new streaming admission until full history is complete.
Existing desired streams repair history through the existing durable REST queue
before WS admission. Unknown bounds trigger discovery. Pending instrument repairs
prevent overlapping cooldown bypass; persistent incomplete intervals block only
that instrument and survive restart. Verified full history clears stale failed
admission receipts. Collection intent is not presented as proof of an open WS.

## Live IAB + provider + storage proof

Before the change's live repair, ACTUSDT had **996,409 unique canonical minutes**.
Its known origin was `2024-11-11T10:00:00Z`; the first missing minute was
`2026-10-04T08:49:00Z`. All existing jobs were read before consumer restart:
16 succeeded, 1 failed, 104 cancelled, no runnable requests. No existing job was
manually rewritten.

From the Navigator `Fill missing history` button, created request:
`ef4511da-a50f-42e2-868d-a9be08c10c89`.

- Fixed interval: `2024-11-11T10:00:00Z` to `2026-10-04T19:07:00Z` (exclusive end).
- Before: 996,409 / 997,027, 99.93%, 618 missing; streaming switch disabled.
- Created: `2026-10-04T19:07:35.683690Z`.
- Finished: `2026-10-04T19:07:39.942958Z`, state `succeeded`.
- Completed/total units: 997,027 / 997,027.
- **rows_read=618; rows_written=618**. Existing 996,409 candles were skipped.
- After: 997,027 / 997,027, 100%, zero missing; switch became enabled.
- Streaming was left disabled. Later closed minutes naturally need filling until
  streaming is requested. This receipt proves the fixed interval, not perpetual
  completeness or a production deployment.
- No browser console errors in the inspected final error log.

Screenshots: [before](before-tail.png), [after](after-tail.png).
Only the Codex in-app browser was used. No saved cookies, credentials, tokens,
raw provider responses or environment dumps.

## Verification

Final focused backend pytest: **81 passed** (API catalog/workspace/reference and
wiring; REST fill; full-history readiness/admission; request runner; durable stream
repair queue; WS publishing and managed I/O/writer integration units).

Command scope:
```
.venv/bin/pytest -q \
 tests/unit/apps/api/test_market_data_catalog.py \
 tests/unit/apps/api/test_market_data_workspace.py \
 tests/unit/apps/api/test_market_data_reference_routes.py \
 tests/unit/apps/api/test_market_data_reference_wiring_module.py \
 tests/unit/apps/worker/test_persistent_writer.py \
 tests/unit/apps/worker/test_market_data_io_process.py \
 tests/unit/contexts/market_data/application/use_cases/test_rest_fill_range_1m.py \
 tests/unit/contexts/market_data/application/services/test_work_requests.py \
 tests/unit/contexts/market_data/application/services/test_full_history.py \
 tests/unit/contexts/market_data/application/services/test_rest_fill_queue.py \
 tests/unit/contexts/market_data/application/services/test_stream_recovery.py \
 tests/unit/contexts/market_data/application/services/test_ws_worker_publishes_redis.py
```

- `pnpm --dir apps/navigator-web typecheck`: pass.
- `pnpm --dir apps/navigator-web test`: **28 files, 289 tests passed**.
- `pnpm --dir apps/navigator-web build`: pass; existing large-chunk warning remains.
- Ruff on every changed/new Python file: pass.
- Pyright on 17 changed/new production Python files: zero errors/warnings.
- `git diff --check`: pass.

Real-infrastructure opt-in tests used existing authorized local credentials only
in process memory and isolated disposable schemas/tables:

- `tests/real_infra/market_data/test_full_history_clickhouse.py`: **1 passed**.
  Empty history, duplicate keys, internal holes, day boundaries, missing tail and
  complete coverage against real bitmap SQL.
- `tests/real_infra/market_data/test_stream_recovery_postgres.py` plus
  `tests/real_infra/market_data/test_work_requests_postgres.py`: **26 passed**.
  Includes a real ClickHouse code 241 fault on a disposable table with a per-query
  test memory limit; user data and system limits unchanged. Tests verify durable
  progression/backoff, queue/control fencing, previous failure reason across
  claim, bounded incomplete-history retries, failed stream admission receipt
  loading and clearing after verified recovery. All disposable data cleaned by
  fixture finalizers.

Failures found during development and fixed before final verification: missing
import, outdated UI expectations, SQL UInt32 casts, retry-count reset caused by
claim clearing error_code, and incompatible new error_phase labels. The final
implementation preserves the schema's existing `fill_window` phase and returns
previous_error_code only internally to the claiming worker.

## Resource evidence and limits

Read-only ACT snapshot measurements (not a throughput benchmark): count + first
missing-interval scan took 661.61 / 645.80 / 654.79 ms in three sequential runs.
A separate sample measured count 12.9 ms and first-gap scan 655.2 ms end to end.
Corresponding query-log samples showed approximately 4.0 MiB query memory,
100–101 aggregate rows read; server durations approximately 7 ms for count and
643 ms for the initial full-history gap scan. This is not WS ingestion latency
and makes no speedup claim against a matched baseline. The scanner emits at most
1,440 minute keys; subsequent repair windows are bounded. No million-candle
payload is materialized in Python.

Historical system memory pressure is not claimed resolved by this change.
Snapshot during diagnosis: container about 1.787 / 3.822 GiB. No resource tuning.
Aggregate correctness relies on the existing append-only candle lifecycle;
administrative candle deletion needs aggregate rebuild. Provider-imposed missing
minutes remain explicit failures; no synthetic candles are fabricated.

## Compatibility and review

- Additive API: `history_scope=full`, full history endpoint, full_history command
  flag, sanitized history_repair_error; legacy bounded coverage remains supported.
- Intentional breaking behavior: streaming selection now requires complete full
  history; disable and organization/same-origin protections remain intact.
- Worker behavior change: startup admission repairs before WS connection;
  four-way request concurrency remains unchanged. Existing persistent states and
  migrations are retained. Update API and workers together.
- Candidate UI replaces daily coverage with full-history meaning; accepted client
  files were untouched.
- Local API, candidate bundle and requests-only consumer updated. Each process
  owner/command was verified; consumer restarts were graceful with no active jobs.
  The separate WS collector was not started as part of this live repair.

Review: focused self-review plus independent read-only review by
`review_full_history`. Reported issues were addressed and the final two blocking
findings were rechecked as closed. Reviewer inspected source/tests, not independent
live execution. Real stream admission/restart edges have unit and durable-store
proof here; the live provider proof is the ACT REST repair, not a new live WS soak.
No claim of full CI, production readiness or production delivery.
