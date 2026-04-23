"""Archived notebook mirror for the removed legacy staged backtest runtime.

This file exists so static analysis can keep tracking the notebook entrypoint
after the v1 staged runner/scorer stack was removed from the repository.

Docs:
  - notebooks/README.md
Related:
  - notebooks/03_sync_backtest_pickle_rollup_1h_ma_grid.ipynb
  - src/trading/contexts/backtest_artifacts/application/use_cases/run_backtest_job_runner_v1.py
"""

from __future__ import annotations

from pathlib import Path

PICKLE_PATH = Path("/ABS/PATH/TO/candles.pkl")


def main() -> None:
    """Explain why the legacy notebook mirror is no longer executable."""
    raise RuntimeError(
        "The legacy staged backtest notebook mirror was archived after "
        "BacktestStagedRunnerV1/CloseFillBacktestStagedScorerV1 were removed. "
        "Port this notebook to the artifact-backed runtime before re-enabling it."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
