# Docs Index

`docs/INDEX.md` указывает кратчайший navigation path к актуальным contract documents и runbooks.

## Canonical Entry Points

- `docs/architecture/README.md`
- `docs/architecture/backtest/README.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`

## Backtest R5 / R6 Path

- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-precompute-runner-v2.md`
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `docs/architecture/backtest/backtest-v2-benchmarks.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`
- `src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py`
- `src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py`
- `src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py`
- `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`
- `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`
- R6-01 boundary: shared `slot-pinned context`, explicit `np.load(..., mmap_mode='r')`,
  `allow_pickle=False`, no runtime scanning, no hot-path hash recomputation.
- R6-02 boundary: Stage A only, `artifacts-only inputs`, deterministic `final_signal`,
  compact trades without risk exits, no-risk shortlist metrics, `chunked variant processing`.
- R6-03 boundary: Stage B artifact-backed risk execution over compact trades and shipped
  `1m hit-times`, fast TP/SL search, exact replay of best TP/SL cell, deterministic metrics,
  additive sync/background scorer bridge for pinned runtime context.
