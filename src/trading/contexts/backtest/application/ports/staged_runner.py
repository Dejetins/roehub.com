from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from trading.contexts.backtest.domain.entities import ExecutionOutcomeV1
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    ExecutionParamsV1,
    RiskParamsV1,
)
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec

BacktestSignalParamsMap = Mapping[str, Mapping[str, BacktestVariantScalar]]
RankingMetricsV1 = Mapping[str, float]
BACKTEST_RANKING_DIRECTION_BY_METRIC_LITERAL_V1 = MappingProxyType(
    {
        "total_return_pct": "DESC",
        "total_return_pct_net_of_funding": "DESC",
        "max_drawdown_pct": "ASC",
        "return_over_max_drawdown": "DESC",
        "profit_factor": "DESC",
        "sharpe_trades": "DESC",
        "win_rate_pct": "DESC",
    }
)
BACKTEST_SCORER_METRIC_KEYS_BY_RANKING_LITERAL_V1 = MappingProxyType(
    {
        "total_return_pct": ("total_return_pct", "Total Return [%]"),
        "total_return_pct_net_of_funding": ("total_return_pct_net_of_funding",),
        "max_drawdown_pct": ("max_drawdown_pct", "Max. Drawdown [%]"),
        "return_over_max_drawdown": ("return_over_max_drawdown",),
        "profit_factor": ("profit_factor",),
        "sharpe_trades": ("sharpe_trades",),
        "win_rate_pct": ("win_rate_pct",),
    }
)


@dataclass(frozen=True, slots=True)
class BacktestVariantScoreDetailsV1:
    """
    Detailed Stage-B score payload for deterministic report assembly on top-ranked variants.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
    """

    metrics: RankingMetricsV1
    target_slice: slice
    execution_params: ExecutionParamsV1
    risk_params: RiskParamsV1
    execution_outcome: ExecutionOutcomeV1

    def __post_init__(self) -> None:
        """
        Validate minimal detail payload invariants for report-building phase.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Metrics mapping includes deterministic `Total Return [%]` ranking key.
        Raises:
            ValueError: If target slice bounds are invalid.
        Side Effects:
            None.
        """
        if self.target_slice.start is None or self.target_slice.stop is None:
            raise ValueError("BacktestVariantScoreDetailsV1.target_slice must be explicit")
        if self.target_slice.start < 0:
            raise ValueError("BacktestVariantScoreDetailsV1.target_slice.start must be >= 0")
        if self.target_slice.stop < self.target_slice.start:
            raise ValueError(
                "BacktestVariantScoreDetailsV1.target_slice.stop must be >= target_slice.start"
            )


class BacktestGridDefaultsProvider(Protocol):
    """
    Port for resolving optional compute/signal grid defaults for backtest variants.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - configs/prod/indicators.yaml
    """

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Resolve compute-grid defaults for one indicator id.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            GridSpec | None: Default compute grid for the indicator when available.
        Assumptions:
            Returned grid uses deterministic axis materialization semantics.
        Raises:
            ValueError: If adapter cannot build deterministic default grid payload.
        Side Effects:
            May read in-memory/defaults configuration state.
        """
        ...

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, GridParamSpec]:
        """
        Resolve signal-parameter default axes for one indicator id.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            Mapping[str, GridParamSpec]: Optional default signal parameter axes.
        Assumptions:
            Missing defaults are represented by an empty mapping.
        Raises:
            ValueError: If adapter returns invalid signal defaults payload.
        Side Effects:
            May read in-memory/defaults configuration state.
        """
        ...

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Return deterministic ordered catalog of backtest-supported indicator ids.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
          - apps/api/dto/backtest_runtime_defaults.py
          - configs/prod/indicators.yaml

        Args:
            None.
        Returns:
            tuple[str, ...]: Stable ordered indicator ids loaded from runtime defaults config.
        Assumptions:
            Returned ids represent the authoritative R1 runtime support surface.
        Raises:
            ValueError: If adapter cannot normalize supported indicator ids deterministically.
        Side Effects:
            May read in-memory/defaults configuration state.
        """
        ...

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        """
        Return deterministic allowed `inputs.source` values for one indicator id.

        Docs:
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
          - apps/api/dto/backtest_runtime_defaults.py
          - configs/prod/indicators.yaml

        Args:
            indicator_id: Indicator identifier.
        Returns:
            tuple[str, ...]: Stable ordered allowed source literals, empty when indicator has no
                configurable source axis.
        Assumptions:
            Indicator lookup is case-insensitive after normalization.
        Raises:
            ValueError: If indicator id is blank.
        Side Effects:
            May read in-memory/defaults configuration state.
        """
        ...


__all__ = [
    "BACKTEST_RANKING_DIRECTION_BY_METRIC_LITERAL_V1",
    "BACKTEST_SCORER_METRIC_KEYS_BY_RANKING_LITERAL_V1",
    "BacktestGridDefaultsProvider",
    "BacktestSignalParamsMap",
    "BacktestVariantScoreDetailsV1",
    "RankingMetricsV1",
]
