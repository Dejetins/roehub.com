from __future__ import annotations

import pytest

from apps.worker.backtest_job_runner.wiring.modules import build_backtest_job_runner_app


def test_build_backtest_job_runner_app_requires_strategy_pg_dsn() -> None:
    """
    Verify worker wiring fails fast when `STRATEGY_PG_DSN` is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Runtime config file exists and loads before DSN validation.
    Raises:
        AssertionError: If missing DSN does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="STRATEGY_PG_DSN"):
        build_backtest_job_runner_app(
            config_path="configs/dev/backtest.yaml",
            environ={},
            metrics_port=9204,
        )
