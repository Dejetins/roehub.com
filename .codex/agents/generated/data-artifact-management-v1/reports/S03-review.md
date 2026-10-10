# S03 independent production-risk review

Date: 2026-10-08. Reviewer: `/root/s03_production_review`, independent subagent using `production-risk-review`. Read-only product review; no recursive review. Only this reviewer-owned report was written.

## Final findings

No remaining material product-code findings in the reviewed S03 snapshot. This is a focused production-risk review, not publication approval or a substitute for final executor gates and ledger acceptance.

Initial findings and disposition:

1. **P1 — unavailable futures funding could replace last-good and appear ready. Resolved.** The managed publisher now obtains one immutable funding snapshot, rejects unavailable or zero-record funding before publication, supplies the frozen snapshot to the native builder, and includes its content/semantics in the resulting source digest. Regression proves a single provider read and byte-identical last-good pointer on unavailable/empty funding.
2. **P1 acceptance-proof gap — subset compatibility was tested only through metadata resolution. Resolved.** Actual filesystem array loader reads available prices/mappings from a managed base-only publication. Actual Backtests preflight produces a recipe for available 15m and rejects absent 1d with `backtest.artifacts_unavailable`. No zero-length timeframe payload is fabricated.
3. **P2 — caller-provided queue envelope controlled sizing and kind. Resolved.** `base_request` reads authoritative immutable base work by installation/request identity before budget estimation; claim still returns the immutable database snapshot for execution.
4. **P2 — routine pin/reservation rejection quarantined the global lane. Resolved.** Precheck failures use a typed prepublication rejection; 0028 reservation rejection becomes `ArtifactPublicationNotStarted`, before publication filesystem mutation. The worker terminates known rejection as error while preserving quarantine for unknown cross-store results. The real-infrastructure test checks a pinned old slot survives and no global quarantined build remains.
5. **P2 integration-proof gap — post-commit wrapper had no production composition seam. Resolved.** Optional REST/WS observers are wired after successful writes; hook errors cannot retry a committed write. Defaults remain disabled. The disposable actual ClickHouse raw writer/materialized-view test connects the callback to bounded reconciliation and PostgreSQL initial work. S05 still owns activation/dispatch.

## Scope and ownership assessment

Reviewed the 14 paths in `S03-owned-paths.json`, current stage acceptance, accepted S01/S02 contracts, native publisher/validator/rollup/funding behavior and the actual reader/preflight boundary. Compared all three mixed modified code files against their `*-S03-before.txt` snapshots. No unrelated scope expansion found. The runtime contract update is justified by the changed producer/consumer behavior. Accepted plan/prompts/updater and earlier reports are outside this review's edit scope.

Physical writes continue through existing 0028 ownership/pins, not the S02 queue claim. Recovery uses declared inventory only under the managed publisher's opt-in flag; legacy default behavior remains unchanged. Declared arrays are still validated, and existing pointer/owner reconciliation remains responsible for physical recovery. Source scans are finite and bounded, snapshots detach candle payload from subsequent changes, first requests remain immutable while desired source state coalesces, and available native timeframes retain independent numerical fixture expectations. Storage admission counts retained/staging files, rejects symlinks/unknown inventory and enforces reservation plus safety headroom.

## Evidence and limits

- **Independently executed:** `.venv/bin/python -m pytest -q tests/unit/contexts/backtest_artifacts/test_automatic_base.py` — **11 passed in 2.94s**. This includes all-timeframe fixtures, short/empty/gapped source, extension/correction, actual reader/preflight, frozen/unavailable/empty futures funding and storage admission.
- **Inspected implementation; executor-reported execution:** the seven tests in `tests/real_infra/backtest/test_automatic_base_lifecycle.py` passed against disposable PostgreSQL/ClickHouse/filesystem. They exercise actual raw commit/materialized view/post-commit work, missed hints/queue capacity recovery, source-growth coalescing, publication, real subprocess death before/after pointer, injected ENOSPC and physical reader pins. The reviewer did not independently repeat those infrastructure runs.
- **Read executor evidence:** `S03-static-checks.json` records owned-file Ruff and changed-product Pyright success and docs-index success. Its project-map check is failed/out-of-date; this review does not relabel it pass or independently attribute the baseline. The executor must retain accurate attribution and disposition in the final report.
- Final broader results belong to `S03-routed-checks.json`, `S03-real-infra-checks.json` and the executor report. At review finalization the executor stated these runs were in progress; this review makes no pass claim for those pending runs.

Compatibility: **compatible-change** for additive managed source/work composition and schema-2 subset production against tested existing readers; **none** for public API/schema migration and existing legacy activation defaults. Absent timeframe remains unavailable, not silently fabricated. No full live-catalog, provider funding integration, throughput, production deployment or activated schedule claim. The tested two-pass source freeze is a finite immutable payload, not a transactional multi-table ClickHouse snapshot. Logical queue reconciliation/activation remains the explicitly assigned S05 responsibility.

## Reviewed file bindings

SHA-256 bindings identify this review snapshot. Any later material product change requires focused re-review; final gate/report changes alone do not.

- `src/trading/contexts/backtest_artifacts/application/services/automatic_base.py`: `aa18931f1f6b3ef146037325b0b02929e0b7ed91bfe0cf87bb5afcca693c928e`
- `src/trading/contexts/backtest_artifacts/application/services/base_source_reconciliation.py`: `e6085d70849d2379199f7e3db297d98f6fbefa5bf55bcd9c43a6ffd98552b730`
- `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/postgres/automatic_base_repository.py`: `d00a5dce646040c90718e3940edd45f729a20d606f579170b0b4959e55a36e12`
- `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py`: `c867cc9dd09ee15bd04a3e79cb7b3f7757831ae45edd73d6fac12e28b8785782`
- `src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/base_storage_admission.py`: `c7fca685c5f112d735c876f1fd67c9d4cbefb4f128a68400592307042b2379b2`
- `src/trading/contexts/market_data/application/services/stored_candle_signal.py`: `c7ea8e57bc16abd1195a197edb70456aada3ea22d0ad2d29f05b181fafa9e6e1`
- `apps/worker/backtest_job_runner/wiring/automatic_base.py`: `d69fca5238233f4f10d54a3a0883c5458f96732dd58b2a95030b993c3b0fcc58`
- `tests/unit/contexts/backtest_artifacts/test_automatic_base.py`: `f2c33227c9e2235e1ff34a3b606ecdf1b31b407247cd0640a473308518c6b91b`
- `tests/real_infra/backtest/test_automatic_base_lifecycle.py`: `c4919989e81058ac2001f0bca7775b9ca01d04dbf903b12a7e9a85f6c27f7860`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_slot_publisher.py`: `4f266b87e326d141ff875591874ce5e6a57bb3412b3d63e69ad664b5a571d936`
- `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`: `f7078d5a24cbbeea95e63ec77a8f042037c09e6ccbfda701f74fe0038a28d1b5`
- `apps/worker/market_data_ws/wiring/modules/market_data_ws.py`: `e02bf059c055d380972205b9fb8b808a4650c43b3e93b6e10ef236f638067f28`
- `tests/unit/contexts/market_data/test_stored_candle_signal.py`: `8979f38a0c66e7158491ad9e0840a10f4ee2962451662c918faee8014c7d9160`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`: `a69d2b2590883ef5b04d65f2d1e7801c7b1b66a4ac4128069704e9799295ecc1`
