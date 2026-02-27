"""
Pydantic models and deterministic mapper for Backtest runtime defaults API endpoint.

Docs:
  - configs/prod/backtest.yaml
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - docs/architecture/backtest/backtest-jobs-api-v1.md
  - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from trading.contexts.backtest.adapters.outbound import BacktestRuntimeConfig


class BacktestRuntimeExecutionDefaultsResponse(BaseModel):
    """
    API response model for non-secret execution defaults used by `/backtests` web UI.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/web/dist/backtest_ui.js
    """

    model_config = ConfigDict(extra="forbid")

    init_cash_quote_default: float
    fixed_quote_default: float
    safe_profit_percent_default: float
    slippage_pct_default: float
    fee_pct_default_by_market_id: dict[str, float]


class BacktestRuntimeJobsDefaultsResponse(BaseModel):
    """
    API response model for jobs defaults required by browser-side validation hints.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/web/dist/backtest_ui.js
    """

    model_config = ConfigDict(extra="forbid")

    top_k_persisted_default: int


class BacktestRuntimeRankingDefaultsResponse(BaseModel):
    """
    API response model for ranking defaults used by browser-side ranking controls.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/web/dist/backtest_ui.js
    """

    model_config = ConfigDict(extra="forbid")

    primary_metric_default: str
    secondary_metric_default: str | None = None


class BacktestRuntimeDefaultsResponse(BaseModel):
    """
    API response model for deterministic runtime defaults contract.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/web/templates/backtests.html
    """

    model_config = ConfigDict(extra="forbid")

    warmup_bars_default: int
    top_k_default: int
    preselect_default: int
    top_trades_n_default: int
    ranking: BacktestRuntimeRankingDefaultsResponse
    execution: BacktestRuntimeExecutionDefaultsResponse
    jobs: BacktestRuntimeJobsDefaultsResponse


def build_backtest_runtime_defaults_response(
    *,
    config: BacktestRuntimeConfig,
) -> BacktestRuntimeDefaultsResponse:
    """
    Convert loaded runtime config into deterministic non-secret browser defaults payload.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/api/wiring/modules/backtest.py

    Args:
        config: Parsed startup-validated runtime config.
    Returns:
        BacktestRuntimeDefaultsResponse: Deterministic response DTO for
            `/backtests/runtime-defaults`.
    Assumptions:
        Runtime config already passed fail-fast startup validation.
    Raises:
        None.
    Side Effects:
        None.
    """
    fee_defaults = {
        str(market_id): config.execution.fee_pct_default_by_market_id[market_id]
        for market_id in sorted(config.execution.fee_pct_default_by_market_id.keys())
    }
    return BacktestRuntimeDefaultsResponse(
        warmup_bars_default=config.warmup_bars_default,
        top_k_default=config.top_k_default,
        preselect_default=config.preselect_default,
        top_trades_n_default=config.reporting.top_trades_n_default,
        ranking=BacktestRuntimeRankingDefaultsResponse(
            primary_metric_default=config.ranking.primary_metric_default,
            secondary_metric_default=config.ranking.secondary_metric_default,
        ),
        execution=BacktestRuntimeExecutionDefaultsResponse(
            init_cash_quote_default=config.execution.init_cash_quote_default,
            fixed_quote_default=config.execution.fixed_quote_default,
            safe_profit_percent_default=config.execution.safe_profit_percent_default,
            slippage_pct_default=config.execution.slippage_pct_default,
            fee_pct_default_by_market_id=fee_defaults,
        ),
        jobs=BacktestRuntimeJobsDefaultsResponse(
            top_k_persisted_default=config.jobs.top_k_persisted_default,
        ),
    )


__all__ = [
    "BacktestRuntimeDefaultsResponse",
    "BacktestRuntimeExecutionDefaultsResponse",
    "BacktestRuntimeJobsDefaultsResponse",
    "BacktestRuntimeRankingDefaultsResponse",
    "build_backtest_runtime_defaults_response",
]
