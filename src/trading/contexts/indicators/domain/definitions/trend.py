"""
Hard indicator definitions for trend and breakout group.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
Related: trading.contexts.indicators.domain.entities.indicator_def,
  trading.contexts.indicators.domain.entities.param_def,
  trading.contexts.indicators.domain.entities.output_spec
"""

from __future__ import annotations

from trading.contexts.indicators.domain.entities import (
    IndicatorDef,
    IndicatorId,
    InputSeries,
    OutputSpec,
    ParamDef,
    ParamKind,
)

_SOURCE_INPUTS = (
    InputSeries.CLOSE,
    InputSeries.HLC3,
    InputSeries.OHLC4,
    InputSeries.LOW,
    InputSeries.HIGH,
    InputSeries.OPEN,
)


def defs() -> tuple[IndicatorDef, ...]:
    """
    Return hard trend-family definitions sorted by indicator_id.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered trend indicator definitions.
    Assumptions:
        Definitions represent compute-relevant contracts without runtime kernels.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = _window(default=20)
    smoothing = _window_named(name="smoothing", default=14)
    mult = _float_param(name="mult", minimum=0.1, maximum=10.0, step=0.01, default=2.0)

    items = (
        IndicatorDef(
            indicator_id=IndicatorId("trend.adx"),
            title="Average Directional Index",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(window, smoothing),
            axes=("window", "smoothing"),
            output=OutputSpec(names=("adx", "plus_di", "minus_di")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.aroon"),
            title="Aroon",
            inputs=(InputSeries.HIGH, InputSeries.LOW),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("aroon_up", "aroon_down", "aroon_osc")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.chandelier_exit"),
            title="Chandelier Exit",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(window, mult),
            axes=("window", "mult"),
            output=OutputSpec(names=("long_stop", "short_stop")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.donchian"),
            title="Donchian Channel",
            inputs=(InputSeries.HIGH, InputSeries.LOW),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("upper", "middle", "lower")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.ichimoku"),
            title="Ichimoku Cloud",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(
                _window_named(name="conversion_window", default=9),
                _window_named(name="base_window", default=26),
                _window_named(name="span_b_window", default=52),
                _window_named(name="displacement", default=26),
            ),
            axes=("conversion_window", "base_window", "span_b_window", "displacement"),
            output=OutputSpec(
                names=("tenkan", "kijun", "senkou_a", "senkou_b", "chikou")
            ),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.keltner"),
            title="Keltner Channel",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(window, mult),
            axes=("window", "mult"),
            output=OutputSpec(names=("basis", "upper", "lower")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.linreg_slope"),
            title="Linear Regression Slope",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("slope",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.psar"),
            title="Parabolic SAR",
            inputs=(InputSeries.HIGH, InputSeries.LOW),
            params=(
                _float_param(
                    name="accel_start",
                    minimum=0.001,
                    maximum=1.0,
                    step=0.001,
                    default=0.02,
                ),
                _float_param(
                    name="accel_step",
                    minimum=0.001,
                    maximum=1.0,
                    step=0.001,
                    default=0.02,
                ),
                _float_param(
                    name="accel_max",
                    minimum=0.01,
                    maximum=5.0,
                    step=0.01,
                    default=0.2,
                ),
            ),
            axes=("accel_max", "accel_start", "accel_step"),
            output=OutputSpec(names=("sar",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.supertrend"),
            title="SuperTrend",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(window, mult),
            axes=("window", "mult"),
            output=OutputSpec(names=("line", "direction")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.vortex"),
            title="Vortex Indicator",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("plus_vi", "minus_vi")),
        ),
    )
    return _sorted_defs(items)


def _window(*, default: int) -> ParamDef:
    """
    Build default integer window parameter for trend indicators.

    Args:
        default: Recommended default window value.
    Returns:
        ParamDef: Window parameter with shared hard bounds.
    Assumptions:
        Trend windows use hard bounds 2..2000 with step 1.
    Raises:
        ValueError: If ParamDef invariants are violated.
    Side Effects:
        None.
    """
    return _window_named(name="window", default=default)


def _window_named(*, name: str, default: int) -> ParamDef:
    """
    Build named integer parameter with common trend hard bounds.

    Args:
        name: Parameter name.
        default: Recommended default value.
    Returns:
        ParamDef: Integer parameter definition.
    Assumptions:
        Parameter domain is constrained to 2..2000 with unit step.
    Raises:
        ValueError: If ParamDef invariants are violated.
    Side Effects:
        None.
    """
    return ParamDef(
        name=name,
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=2_000,
        step=1,
        default=default,
    )


def _float_param(
    *,
    name: str,
    minimum: float,
    maximum: float,
    step: float,
    default: float,
) -> ParamDef:
    """
    Build floating parameter definition for trend indicators.

    Args:
        name: Parameter name.
        minimum: Inclusive hard lower bound.
        maximum: Inclusive hard upper bound.
        step: Hard grid step.
        default: Recommended default value.
    Returns:
        ParamDef: Float parameter definition.
    Assumptions:
        Bounds and step are chosen to avoid impossible granularities.
    Raises:
        ValueError: If ParamDef invariants are violated.
    Side Effects:
        None.
    """
    return ParamDef(
        name=name,
        kind=ParamKind.FLOAT,
        hard_min=minimum,
        hard_max=maximum,
        step=step,
        default=default,
    )


def _sorted_defs(items: tuple[IndicatorDef, ...]) -> tuple[IndicatorDef, ...]:
    """
    Sort trend indicator definitions by stable identifier.

    Args:
        items: Unsorted or partially sorted indicator definitions.
    Returns:
        tuple[IndicatorDef, ...]: Deterministic tuple sorted by `indicator_id`.
    Assumptions:
        Indicator ids are unique in `items`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(sorted(items, key=lambda item: item.indicator_id.value))
