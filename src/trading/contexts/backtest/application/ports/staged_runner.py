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
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    IndicatorVariantSelection,
)
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec

BacktestSignalParamsMap = Mapping[str, Mapping[str, BacktestVariantScalar]]
RankingMetricsV1 = Mapping[str, float]
BACKTEST_RANKING_DIRECTION_BY_METRIC_LITERAL_V1 = MappingProxyType(
    {
        "total_return_pct": "DESC",
        "max_drawdown_pct": "ASC",
        "return_over_max_drawdown": "DESC",
        "profit_factor": "DESC",
    }
)
BACKTEST_SCORER_METRIC_KEYS_BY_RANKING_LITERAL_V1 = MappingProxyType(
    {
        "total_return_pct": ("total_return_pct", "Total Return [%]"),
        "max_drawdown_pct": ("max_drawdown_pct", "Max. Drawdown [%]"),
        "return_over_max_drawdown": ("return_over_max_drawdown",),
        "profit_factor": ("profit_factor",),
    }
)


@dataclass(frozen=True, slots=True)
class BacktestVariantScoreDetailsV1:
    """
    Detailed Stage-B score payload for deterministic report assembly on top-ranked variants.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
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
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
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


class BacktestStagedVariantScorer(Protocol):
    """
    Backward-compatible Stage A / Stage B scoring port (`score_variant`).

    This protocol is kept for compatibility with existing scorer implementations.
    New ranking hot paths SHOULD prefer `BacktestStagedVariantMetricScorer`
    (`score_variant_metric`) to avoid accidental details-path coupling.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    def score_variant(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: BacktestSignalParamsMap,
        risk_params: Mapping[str, BacktestVariantScalar],
        indicator_variant_key: str,
        variant_key: str,
    ) -> RankingMetricsV1:
        """
        Legacy compatibility scorer API for one deterministic Stage A/Stage B variant.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles used by scoring backend.
            indicator_selections: Explicit compute selections for indicators key builder.
            signal_params: Signal parameter values for this variant.
            risk_params: Risk payload (`sl_enabled/sl_pct/tp_enabled/tp_pct`) for this variant.
            indicator_variant_key: Deterministic compute-only indicators key.
            variant_key: Deterministic backtest variant key.
        Returns:
            RankingMetricsV1:
                Metric mapping containing `total_return_pct` aliases and optional
                ranking literals from `BACKTEST_SCORER_METRIC_KEYS_BY_RANKING_LITERAL_V1`.
        Assumptions:
            Returned values are deterministic for identical inputs.
        Raises:
            ValueError: If scorer cannot produce required ranking metric.
        Side Effects:
            Depends on concrete adapter implementation.
        """
        ...


class BacktestStagedVariantMetricScorer(Protocol):
    """
    Metric-only Stage A / Stage B scorer port for ranking hot paths.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
    """

    def score_variant_metric(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: BacktestSignalParamsMap,
        risk_params: Mapping[str, BacktestVariantScalar],
        indicator_variant_key: str,
        variant_key: str,
    ) -> RankingMetricsV1:
        """
        Score one deterministic variant and return ranking metrics only.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles used by scoring backend.
            indicator_selections: Explicit compute selections for indicators key builder.
            signal_params: Signal parameter values for this variant.
            risk_params: Risk payload (`sl_enabled/sl_pct/tp_enabled/tp_pct`) for this variant.
            indicator_variant_key: Deterministic compute-only indicators key.
            variant_key: Deterministic backtest variant key.
        Returns:
            RankingMetricsV1:
                Metric mapping containing `total_return_pct` aliases and optional
                ranking literals from `BACKTEST_SCORER_METRIC_KEYS_BY_RANKING_LITERAL_V1`.
        Assumptions:
            Returned values are deterministic for identical inputs.
        Raises:
            ValueError: If scorer cannot produce required ranking metric payload.
        Side Effects:
            Depends on concrete adapter implementation.
        """
        ...


class BacktestStagedVariantScorerWithDetails(Protocol):
    """
    Optional scorer extension exposing deterministic Stage-B details for reporting payload.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
    """

    def score_variant_with_details(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: BacktestSignalParamsMap,
        risk_params: Mapping[str, BacktestVariantScalar],
        indicator_variant_key: str,
        variant_key: str,
    ) -> BacktestVariantScoreDetailsV1:
        """
        Score one variant and return detailed deterministic payload for report assembly.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles used by scoring backend.
            indicator_selections: Explicit compute selections for indicators key builder.
            signal_params: Signal parameter values for this variant.
            risk_params: Risk payload (`sl_enabled/sl_pct/tp_enabled/tp_pct`) for this variant.
            indicator_variant_key: Deterministic compute-only indicators key.
            variant_key: Deterministic backtest variant key.
        Returns:
            BacktestVariantScoreDetailsV1: Detailed payload used by reporting layer.
        Assumptions:
            Returned payload corresponds to the same deterministic execution as ranking score.
        Raises:
            ValueError: If scorer cannot produce deterministic details.
        Side Effects:
            Depends on concrete adapter implementation.
        """
        ...


__all__ = [
    "BACKTEST_RANKING_DIRECTION_BY_METRIC_LITERAL_V1",
    "BACKTEST_SCORER_METRIC_KEYS_BY_RANKING_LITERAL_V1",
    "BacktestGridDefaultsProvider",
    "BacktestSignalParamsMap",
    "BacktestStagedVariantMetricScorer",
    "BacktestStagedVariantScorerWithDetails",
    "BacktestStagedVariantScorer",
    "BacktestVariantScoreDetailsV1",
    "RankingMetricsV1",
]
