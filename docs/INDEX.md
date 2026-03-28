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
- R6-01 boundary: shared `slot-pinned context`, explicit `np.load(..., mmap_mode='r')`,
  `allow_pickle=False`, no runtime scanning, no hot-path hash recomputation.
