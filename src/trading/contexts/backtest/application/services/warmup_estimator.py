from __future__ import annotations

import math

from trading.contexts.backtest.application.dto import RunBacktestTemplate
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    GridParamSpec,
    GridSpec,
    RangeValuesSpec,
)
from trading.contexts.indicators.domain.specifications.grid_param_spec import GridValue


def resolve_internal_backtest_warmup_bars(
    *,
    template: RunBacktestTemplate,
    warmup_bars: int | None = None,
) -> int:
    """
    Resolve internal warmup bars from compatibility input or derived indicator requirements.

    Args:
        template: Effective validated backtest template used for runtime execution.
        warmup_bars:
            Optional compatibility-only internal warmup value from legacy persisted reads or
            direct internal callers. Public DTOs no longer surface this field.
    Returns:
        int: Positive internal warmup bars count used for artifact timeline construction.
    Assumptions:
        New public launch and report paths should normally pass `None` and use the derived value.
    Raises:
        ValueError: If explicit compatibility warmup is non-positive.
    Side Effects:
        None.
    """
    if warmup_bars is not None:
        if warmup_bars <= 0:
            raise ValueError("internal warmup_bars must be > 0")
        return warmup_bars
    return estimate_backtest_template_warmup_bars(template=template)


def estimate_backtest_template_warmup_bars(*, template: RunBacktestTemplate) -> int:
    """
    Estimate derived warmup bars from effective indicator grid requirements.

    Args:
        template: Effective validated backtest template containing indicator grid specs.
    Returns:
        int: Deterministic derived warmup bars estimate (`>= 1`).
    Assumptions:
        The estimator follows the strategy-side `numeric_max_param_v1` spirit and treats the
        largest positive numeric indicator requirement as the safe warmup bound.
    Raises:
        None.
    Side Effects:
        None.
    """
    candidates: list[int] = []
    for grid in template.indicator_grids:
        candidates.extend(_collect_grid_warmup_candidates(grid=grid))
    if not candidates:
        return 1
    return max(candidates)


def _collect_grid_warmup_candidates(*, grid: GridSpec) -> list[int]:
    """
    Collect positive warmup candidates from one indicator grid specification.

    Args:
        grid: One effective indicator grid from the resolved template.
    Returns:
        list[int]: Positive numeric warmup candidates found in the grid.
    Assumptions:
        Grid parameter iteration is sorted to keep traversal deterministic.
    Raises:
        None.
    Side Effects:
        None.
    """
    candidates: list[int] = []
    if grid.source is not None:
        candidates.extend(_collect_axis_warmup_candidates(spec=grid.source))
    for param_name in sorted(grid.params.keys()):
        candidates.extend(_collect_axis_warmup_candidates(spec=grid.params[param_name]))
    return candidates


def _collect_axis_warmup_candidates(*, spec: GridParamSpec) -> list[int]:
    """
    Collect positive warmup candidates from one grid axis specification.

    Args:
        spec: One effective grid axis specification.
    Returns:
        list[int]: Positive integer warmup candidates for the axis.
    Assumptions:
        Range specs use `stop_inclusive` as a safe upper bound even when materialization stops
        earlier because the final step does not land exactly on the stop value.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(spec, ExplicitValuesSpec):
        candidates: list[int] = []
        for value in spec.values:
            candidates.extend(_collect_scalar_warmup_candidates(value=value))
        return candidates
    if isinstance(spec, RangeValuesSpec):
        return _collect_scalar_warmup_candidates(value=spec.stop_inclusive)

    candidates = []
    for value in spec.materialize():
        candidates.extend(_collect_scalar_warmup_candidates(value=value))
    return candidates


def _collect_scalar_warmup_candidates(*, value: GridValue) -> list[int]:
    """
    Normalize one scalar grid value into positive integer warmup candidates.

    Args:
        value: One materialized or upper-bound axis scalar value.
    Returns:
        list[int]: Zero or one positive integer warmup candidates.
    Assumptions:
        Non-numeric values such as indicator `source` literals never contribute to warmup.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value] if value > 0 else []
    if isinstance(value, float):
        if value <= 0 or math.isnan(value) or math.isinf(value):
            return []
        return [int(math.ceil(value))]
    return []
