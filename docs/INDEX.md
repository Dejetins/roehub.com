# Docs Index

`docs/INDEX.md` указывает кратчайший navigation path к актуальным contract documents и runbooks.

## Canonical Entry Points

- `docs/architecture/README.md`
- `docs/architecture/backtest/README.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`
- `docs/runbooks/backtest-job-runner.md`
- `docs/runbooks/backtest-rollout-rollback.md`

## Backtest R5 / R6 / R7 / R9 / R12 Path

- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-precompute-runner-v2.md`
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `docs/architecture/backtest/backtest-v2-benchmarks.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/market-data-metrics.md`
- `docs/runbooks/market-data-metrics-reference-ru.md`
- `docs/runbooks/backtest-job-runner.md`
- `docs/runbooks/backtest-rollout-rollback.md`
- `src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py`
- `src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py`
- `src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py`
- `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`
- `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`
- `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
- `docs/architecture/backtest/backtest-runs-history-v2.md`
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`
- `docs/architecture/apps/web/web-backtest-jobs-ui-async-v1.md`
- `alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`
- R6-01 boundary: shared `slot-pinned context`, explicit `np.load(..., mmap_mode='r')`,
  `allow_pickle=False`, no runtime scanning, no hot-path hash recomputation.
- R6-02 boundary: Stage A only, `artifacts-only inputs`, deterministic `final_signal`,
  compact trades without risk exits, no-risk shortlist metrics, `chunked variant processing`.
- R6-03 boundary: Stage B artifact-backed risk execution over compact trades and shipped
  `1m hit-times`, fast TP/SL search, exact replay of best TP/SL cell, deterministic metrics,
  additive sync/background scorer bridge for pinned runtime context.
- R7-01 boundary: unified persisted-run storage in existing PG table family, denormalized
  `execution_mode/market_id/symbol/timeframe/requested_top_n/ranking_*` metadata, summary-only
  top rows with `summary_metrics_json`, `best_tp_pct`, `best_sl_pct`, and transitional
  `report_table_md/trades_json = NULL`.
- R12 boundary: stable artifact output contract plus stage-oriented `timeframe-scoped execution`,
  explicit `execution_policy`, `ChunkPlanner`, one open `current_timeframe` session at a time,
  eager `np.memmap` signal writes, and operator-facing stage/timeframe/chunk observability.
