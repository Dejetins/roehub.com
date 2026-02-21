from __future__ import annotations

from typing import Mapping, Protocol

from trading.contexts.backtest.domain.value_objects import BacktestVariantScalar
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    IndicatorVariantSelection,
)
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec

BacktestSignalParamsMap = Mapping[str, Mapping[str, BacktestVariantScalar]]


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
    Port for Stage A / Stage B variant scoring using `Total Return [%]` ranking metric.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
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
    ) -> Mapping[str, float]:
        """
        Score one deterministic variant for Stage A or Stage B ranking.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles used by scoring backend.
            indicator_selections: Explicit compute selections for indicators key builder.
            signal_params: Signal parameter values for this variant.
            risk_params: Risk payload (`sl_enabled/sl_pct/tp_enabled/tp_pct`) for this variant.
            indicator_variant_key: Deterministic compute-only indicators key.
            variant_key: Deterministic backtest variant key.
        Returns:
            Mapping[str, float]: Metric mapping containing `Total Return [%]`.
        Assumptions:
            Returned value for `Total Return [%]` is deterministic for identical inputs.
        Raises:
            ValueError: If scorer cannot produce required ranking metric.
        Side Effects:
            Depends on concrete adapter implementation.
        """
        ...


__all__ = [
    "BacktestGridDefaultsProvider",
    "BacktestSignalParamsMap",
    "BacktestStagedVariantScorer",
]
