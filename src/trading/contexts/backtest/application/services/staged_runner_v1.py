from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from trading.contexts.backtest.application.dto import BacktestVariantPreview, RunBacktestTemplate
from trading.contexts.backtest.application.ports import (
    BacktestGridDefaultsProvider,
    BacktestStagedVariantScorer,
)
from trading.contexts.backtest.application.services.grid_builder_v1 import (
    STAGE_A_LITERAL,
    STAGE_B_LITERAL,
    BacktestGridBuildContextV1,
    BacktestGridBuilderV1,
    BacktestStageABaseVariant,
)
from trading.contexts.backtest.domain.value_objects import (
    build_backtest_variant_key_v1,
)
from trading.contexts.indicators.application.dto import CandleArrays
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.services.grid_builder import (
    MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    MAX_VARIANTS_PER_COMPUTE_DEFAULT,
)

TOTAL_RETURN_METRIC_LITERAL = "Total Return [%]"


@dataclass(frozen=True, slots=True)
class BacktestStagedRunResultV1:
    """
    Deterministic staged-run summary used by `RunBacktestUseCase`.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    variants: tuple[BacktestVariantPreview, ...]
    stage_a_variants_total: int
    stage_b_variants_total: int
    estimated_memory_bytes: int
    indicator_estimate_calls: int

    def __post_init__(self) -> None:
        """
        Validate deterministic staged-run totals and response payload shape.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variants list is already sorted by staged ranking key before construction.
        Raises:
            ValueError: If one total is invalid.
        Side Effects:
            None.
        """
        if self.stage_a_variants_total <= 0:
            raise ValueError("BacktestStagedRunResultV1.stage_a_variants_total must be > 0")
        if self.stage_b_variants_total <= 0:
            raise ValueError("BacktestStagedRunResultV1.stage_b_variants_total must be > 0")
        if self.estimated_memory_bytes <= 0:
            raise ValueError("BacktestStagedRunResultV1.estimated_memory_bytes must be > 0")
        if self.indicator_estimate_calls < 0:
            raise ValueError("BacktestStagedRunResultV1.indicator_estimate_calls must be >= 0")


@dataclass(frozen=True, slots=True)
class _StageAScoredVariant:
    """
    Internal Stage A scored row for deterministic shortlist sorting.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    base_variant: BacktestStageABaseVariant
    total_return_pct: float


@dataclass(frozen=True, slots=True)
class _StageBScoredVariant:
    """
    Internal Stage B scored row for deterministic top-K sorting.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    variant_index: int
    indicator_variant_key: str
    variant_key: str
    total_return_pct: float


class BacktestStagedRunnerV1:
    """
    Run deterministic staged pipeline (Stage A shortlist -> Stage B expand -> top-K).

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/staged_runner.py
    """

    def __init__(
        self,
        *,
        grid_builder: BacktestGridBuilderV1 | None = None,
    ) -> None:
        """
        Initialize staged runner with optional custom grid-builder implementation.

        Args:
            grid_builder: Optional custom grid builder for Stage A/Stage B context.
        Returns:
            None.
        Assumptions:
            Default grid builder follows EPIC-04 deterministic guard contracts.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._grid_builder = grid_builder or BacktestGridBuilderV1()

    def run(
        self,
        *,
        template: RunBacktestTemplate,
        candles: CandleArrays,
        preselect: int,
        top_k: int,
        indicator_compute: IndicatorCompute,
        scorer: BacktestStagedVariantScorer,
        defaults_provider: BacktestGridDefaultsProvider | None = None,
        max_variants_per_compute: int = MAX_VARIANTS_PER_COMPUTE_DEFAULT,
        max_compute_bytes_total: int = MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    ) -> BacktestStagedRunResultV1:
        """
        Execute staged ranking flow and return deterministic top-K variant previews.

        Args:
            template: Resolved backtest template payload.
            candles: Dense warmup-inclusive candle arrays.
            preselect: Stage A shortlist size.
            top_k: Stage B top-k output size.
            indicator_compute: Indicator estimate port for staged grid materialization.
            scorer: Scoring port returning `Total Return [%]` for every variant.
            defaults_provider: Optional defaults provider for compute/signal fallback.
            max_variants_per_compute: Stage variants guard limit.
            max_compute_bytes_total: Stage memory guard limit.
        Returns:
            BacktestStagedRunResultV1: Deterministic staged-run output summary.
        Assumptions:
            Scorer returns deterministic metrics for identical variant payloads.
        Raises:
            ValueError: If top-k/preselect values are invalid or scorer payload is invalid.
            RoehubError: Propagated from staged grid-builder guard checks.
        Side Effects:
            None.
        """
        if preselect <= 0:
            raise ValueError("BacktestStagedRunnerV1 preselect must be > 0")
        if top_k <= 0:
            raise ValueError("BacktestStagedRunnerV1 top_k must be > 0")

        grid_context = self._grid_builder.build(
            template=template,
            candles=candles,
            indicator_compute=indicator_compute,
            preselect=preselect,
            defaults_provider=defaults_provider,
            max_variants_per_compute=max_variants_per_compute,
            max_compute_bytes_total=max_compute_bytes_total,
        )

        stage_a_rows = self._score_stage_a(
            grid_context=grid_context,
            candles=candles,
            scorer=scorer,
        )
        shortlist_len = min(preselect, len(stage_a_rows))
        shortlist = stage_a_rows[:shortlist_len]

        stage_b_rows = self._score_stage_b(
            template=template,
            grid_context=grid_context,
            shortlist=shortlist,
            candles=candles,
            scorer=scorer,
        )
        top_rows = stage_b_rows[: min(top_k, len(stage_b_rows))]
        variants = tuple(
            BacktestVariantPreview(
                variant_index=row.variant_index,
                variant_key=row.variant_key,
                indicator_variant_key=row.indicator_variant_key,
                total_return_pct=row.total_return_pct,
            )
            for row in top_rows
        )
        return BacktestStagedRunResultV1(
            variants=variants,
            stage_a_variants_total=grid_context.stage_a_variants_total,
            stage_b_variants_total=grid_context.stage_b_variants_total,
            estimated_memory_bytes=grid_context.estimated_memory_bytes,
            indicator_estimate_calls=grid_context.indicator_estimate_calls,
        )

    def _score_stage_a(
        self,
        *,
        grid_context: BacktestGridBuildContextV1,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
    ) -> list[_StageAScoredVariant]:
        """
        Score Stage A base variants and return deterministic sorted shortlist candidates.

        Args:
            grid_context: Built staged grid context.
            candles: Dense candles forwarded to scorer.
            scorer: Stage scorer port.
        Returns:
            list[_StageAScoredVariant]: Sorted Stage A rows by ranking and tie-break key.
        Assumptions:
            Stage A uses base key tie-break with risk disabled.
        Raises:
            ValueError: If scorer payload lacks required ranking metric.
        Side Effects:
            None.
        """
        rows: list[_StageAScoredVariant] = []
        for base_variant in grid_context.iter_stage_a_variants():
            metrics = scorer.score_variant(
                stage=STAGE_A_LITERAL,
                candles=candles,
                indicator_selections=base_variant.indicator_selections,
                signal_params=base_variant.signal_params,
                risk_params={
                    "sl_enabled": False,
                    "sl_pct": None,
                    "tp_enabled": False,
                    "tp_pct": None,
                },
                indicator_variant_key=base_variant.indicator_variant_key,
                variant_key=base_variant.base_variant_key,
            )
            rows.append(
                _StageAScoredVariant(
                    base_variant=base_variant,
                    total_return_pct=_extract_total_return_pct(metrics=metrics),
                )
            )
        return sorted(
            rows,
            key=lambda row: (-row.total_return_pct, row.base_variant.base_variant_key),
        )

    def _score_stage_b(
        self,
        *,
        template: RunBacktestTemplate,
        grid_context: BacktestGridBuildContextV1,
        shortlist: list[_StageAScoredVariant],
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
    ) -> list[_StageBScoredVariant]:
        """
        Score Stage B expanded variants from Stage A shortlist and risk cartesian axes.

        Args:
            template: Resolved backtest template payload.
            grid_context: Built staged grid context.
            shortlist: Sorted Stage A shortlist rows.
            candles: Dense candles forwarded to scorer.
            scorer: Stage scorer port.
        Returns:
            list[_StageBScoredVariant]: Sorted Stage B rows by ranking and tie-break key.
        Assumptions:
            Stage B tie-break key is full `variant_key`.
        Raises:
            ValueError: If scorer payload lacks required ranking metric.
        Side Effects:
            None.
        """
        rows: list[_StageBScoredVariant] = []
        risk_variants = grid_context.risk_variants
        risk_total = len(risk_variants)
        for shortlist_index, stage_a_row in enumerate(shortlist):
            base_variant = stage_a_row.base_variant
            for risk_variant in risk_variants:
                variant_index = (shortlist_index * risk_total) + risk_variant.risk_index
                variant_key = build_backtest_variant_key_v1(
                    indicator_variant_key=base_variant.indicator_variant_key,
                    direction_mode=template.direction_mode,
                    sizing_mode=template.sizing_mode,
                    signals=base_variant.signal_params,
                    risk_params=risk_variant.risk_params,
                    execution_params=template.execution_params,
                )
                metrics = scorer.score_variant(
                    stage=STAGE_B_LITERAL,
                    candles=candles,
                    indicator_selections=base_variant.indicator_selections,
                    signal_params=base_variant.signal_params,
                    risk_params=risk_variant.risk_params,
                    indicator_variant_key=base_variant.indicator_variant_key,
                    variant_key=variant_key,
                )
                rows.append(
                    _StageBScoredVariant(
                        variant_index=variant_index,
                        indicator_variant_key=base_variant.indicator_variant_key,
                        variant_key=variant_key,
                        total_return_pct=_extract_total_return_pct(metrics=metrics),
                    )
                )
        return sorted(rows, key=lambda row: (-row.total_return_pct, row.variant_key))


def _extract_total_return_pct(*, metrics: Mapping[str, float]) -> float:
    """
    Extract `Total Return [%]` metric from scorer payload and validate numeric type.

    Args:
        metrics: Scorer metrics mapping.
    Returns:
        float: Deterministic ranking value for staged sorting.
    Assumptions:
        Scorer payload contains `Total Return [%]` key.
    Raises:
        ValueError: If metric key is absent or value is not numeric scalar.
    Side Effects:
        None.
    """
    value = metrics.get(TOTAL_RETURN_METRIC_LITERAL)
    if value is None:
        raise ValueError(f"scorer payload must contain '{TOTAL_RETURN_METRIC_LITERAL}' metric")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"metric '{TOTAL_RETURN_METRIC_LITERAL}' must be numeric")
    return float(value)


__all__ = [
    "BacktestStagedRunResultV1",
    "BacktestStagedRunnerV1",
    "TOTAL_RETURN_METRIC_LITERAL",
]

