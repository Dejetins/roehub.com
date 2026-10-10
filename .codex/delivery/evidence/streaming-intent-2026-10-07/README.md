# Streaming intent and automatic history preparation — 2026-10-07

Scope: local Navigator candidate only. No publication, migration, resource changes, or changes to platform-web. Existing unrelated working-tree changes preserved.

## Behavior

PUT selection persists intent without a synchronous history gate. Unknown/empty/partial history is prepared through the existing durable REST queue. Preparation checkpoints existing 60-minute windows, then verified history admits WS. Normal reconnect/tail repair covers the moving minute boundary. The switch remains available during preparation; counts and preparation/retry status are visible. Explicit or strategy selections are respected; the collector no longer adds an implicit BTC reference subscription for an empty desired set.

Managed I/O children bound history/selection checks and are reaped on stop. Intent is re-read after preparation before admission. Preparation cooldown does not suspend desired-set polling/removal. Queued history work checks intent before each window; an in-flight window may finish. Received-candle receipts retain their independent repair behavior.

Existing selection audit persists changes. Sanitized transition logs and recovery receipts provide diagnostics. No new general browser-error telemetry service is claimed.

## Verification

- Navigator data-page suite: 48 passed, including zero-history enable/disable preparation. Typecheck and build passed (existing chunk-size warning).
- Backend focused API/history/queue suite: 30 passed before final control-race fixes.
- Wider process/writer/repository/API/history/recovery suite: 45 passed before final control-race fixes.
- Final control/process/history/WS suite: 24 passed; new intent-race regressions: 2 passed.
- Ruff and focused Pyright passed for production control changes.
- Real ClickHouse: 2 passed, isolated disposable databases; bitmap coverage/gaps and zero-history admission only after the last missing minute is present.
- Real PostgreSQL: 6 passed in a disposable database. An initial schema-only run had 1 lease collision with the live collector; rerunning in a separate database correctly isolated the database-level advisory lock. No live collector interruption for that test.
- Actual managed selected/prepare calls returned 0GUSDT and ACTUSDT, both ready, no residual child processes.
- Actual local collector restart found missing newest minutes for both instruments, queued repairs, logged history_ready and resumed collection automatically.
- IAB with built production UI/API: 0GUSDC initially had zero candles. One toggle accepted intent and showed `Preparing history · missing candles: 546,504`. Collector wrote one 60-minute window. Off restored the switch and logged `stream_history_cancelled`; newly stored candles were preserved. Test selection was removed. User selections 0GUSDT/ACTUSDT remained enabled. Screenshots: preparing-iab.png, disabled-iab.png.

## Compatibility and proof limits

API: compatible-change (relaxed admission; same response, authorization and origin checks). Worker operational behavior: compatible-change, explicit desired-set semantics and bounded control reads. No schema change. UI: compatible-change. Local API/web/collector reloaded together.

Provider completion of a full empty instrument history is not claimed. Disposable ClickHouse tests prove storage/gate behavior, not an entire exchange history. Local Redis publication is disabled by the existing preview configuration and is outside this proof. No streaming latency claim is made by this change.

Review: independent material-risk review identified stale admission and cooldown blocking Off; both fixed and regression-tested. Final independent follow-up: both P2 findings closed; 6 reviewer-run tests passed; no new material findings. Final API/repository/control suite: 18 passed.
