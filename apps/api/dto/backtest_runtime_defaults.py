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
from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider


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


class BacktestRuntimeRequestTimeframesContractResponse(BaseModel):
    """
    API response model for frozen R0 request-timeframe contract literals.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    allowed: list[str]
    forbidden: list[str]


class BacktestRuntimeSummaryContractResponse(BaseModel):
    """
    API response model for frozen R0 summary/ranking contract surface.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    top_n_default: int
    top_n_max: int
    ranking_metrics: list[str]
    sortable_columns: list[str]


class BacktestRuntimeSignalsContractResponse(BaseModel):
    """
    API response model for frozen R0 signal-params contract surface.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    params_path: str
    params_policy: str


class BacktestRuntimeExecutionContractResponse(BaseModel):
    """
    API response model for frozen R0 execution semantics contract.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    risk_model: str


class BacktestRuntimeLaunchContractResponse(BaseModel):
    """
    API response model for frozen R0 launch/execution-mode contract surface.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    execution_mode: str
    auto_preflight_enabled: bool
    auto_fallback_to_background_enabled: bool
    supported_indicator_ids: list[str]
    source_values_by_indicator_id: dict[str, list[str]]


class BacktestRuntimeContractsResponse(BaseModel):
    """
    API response model for additive frozen R0 contract surface.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    request_timeframes: BacktestRuntimeRequestTimeframesContractResponse
    summary: BacktestRuntimeSummaryContractResponse
    signals: BacktestRuntimeSignalsContractResponse
    execution: BacktestRuntimeExecutionContractResponse
    launch: BacktestRuntimeLaunchContractResponse


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
    contracts: BacktestRuntimeContractsResponse


def build_backtest_runtime_defaults_response(
    *,
    config: BacktestRuntimeConfig,
    defaults_provider: BacktestGridDefaultsProvider | None = None,
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
        defaults_provider: Optional startup-validated indicators defaults provider/catalog.
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
    supported_indicator_ids = (
        list(defaults_provider.supported_indicator_ids()) if defaults_provider is not None else []
    )
    source_values_by_indicator_id = (
        {
            indicator_id: list(defaults_provider.allowed_source_values(indicator_id=indicator_id))
            for indicator_id in supported_indicator_ids
        }
        if defaults_provider is not None
        else {}
    )
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
        contracts=BacktestRuntimeContractsResponse(
            request_timeframes=BacktestRuntimeRequestTimeframesContractResponse(
                allowed=list(config.contracts.allowed_request_timeframes),
                forbidden=list(config.contracts.forbidden_request_timeframes),
            ),
            summary=BacktestRuntimeSummaryContractResponse(
                top_n_default=config.contracts.top_n_default,
                top_n_max=config.contracts.top_n_max,
                ranking_metrics=list(config.contracts.ranking_metrics),
                sortable_columns=list(config.contracts.sortable_summary_columns),
            ),
            signals=BacktestRuntimeSignalsContractResponse(
                params_path=config.contracts.signals_v1_params_path,
                params_policy=config.contracts.signals_v1_params_policy,
            ),
            execution=BacktestRuntimeExecutionContractResponse(
                risk_model=config.contracts.risk_model,
            ),
            launch=BacktestRuntimeLaunchContractResponse(
                execution_mode=config.contracts.execution_mode,
                auto_preflight_enabled=config.contracts.auto_preflight_enabled,
                auto_fallback_to_background_enabled=(
                    config.contracts.auto_fallback_to_background_enabled
                ),
                supported_indicator_ids=supported_indicator_ids,
                source_values_by_indicator_id=source_values_by_indicator_id,
            ),
        ),
    )


__all__ = [
    "BacktestRuntimeContractsResponse",
    "BacktestRuntimeDefaultsResponse",
    "BacktestRuntimeExecutionDefaultsResponse",
    "BacktestRuntimeExecutionContractResponse",
    "BacktestRuntimeJobsDefaultsResponse",
    "BacktestRuntimeLaunchContractResponse",
    "BacktestRuntimeRankingDefaultsResponse",
    "BacktestRuntimeRequestTimeframesContractResponse",
    "BacktestRuntimeSignalsContractResponse",
    "BacktestRuntimeSummaryContractResponse",
    "build_backtest_runtime_defaults_response",
]
