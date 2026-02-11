"""
Hard indicator definitions for volatility group.

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
    Return hard volatility-family definitions sorted by indicator_id.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered volatility indicator definitions.
    Assumptions:
        Source-based indicators expose source via dedicated axis.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = _window(default=20)
    mult = _float_param(name="mult", minimum=0.1, maximum=10.0, step=0.01, default=2.0)

    items = (
        IndicatorDef(
            indicator_id=IndicatorId("volatility.atr"),
            title="Average True Range",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("atr",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.bbands"),
            title="Bollinger Bands",
            inputs=_SOURCE_INPUTS,
            params=(window, mult),
            axes=("mult", "source", "window"),
            output=OutputSpec(names=("basis", "upper", "lower")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.bbands_bandwidth"),
            title="Bollinger Bandwidth",
            inputs=_SOURCE_INPUTS,
            params=(window, mult),
            axes=("mult", "source", "window"),
            output=OutputSpec(names=("bandwidth",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.bbands_percent_b"),
            title="Bollinger Percent B",
            inputs=_SOURCE_INPUTS,
            params=(window, mult),
            axes=("mult", "source", "window"),
            output=OutputSpec(names=("percent_b",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.hv"),
            title="Historical Volatility (Log Returns)",
            inputs=_SOURCE_INPUTS,
            params=(window, _window_named(name="annualization", default=365)),
            axes=("annualization", "source", "window"),
            output=OutputSpec(names=("hv",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.stddev"),
            title="Rolling Standard Deviation",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("stddev",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.tr"),
            title="True Range",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(),
            axes=(),
            output=OutputSpec(names=("tr",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.variance"),
            title="Rolling Variance",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("variance",)),
        ),
    )
    return _sorted_defs(items)


def _window(*, default: int) -> ParamDef:
    """
    Build default integer window parameter for volatility indicators.

    Args:
        default: Recommended default window value.
    Returns:
        ParamDef: Window parameter with shared hard bounds.
    Assumptions:
        Volatility windows use hard bounds 2..2000 with step 1.
    Raises:
        ValueError: If ParamDef invariants are violated.
    Side Effects:
        None.
    """
    return _window_named(name="window", default=default)


def _window_named(*, name: str, default: int) -> ParamDef:
    """
    Build named integer parameter with common volatility hard bounds.

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
    Build floating parameter definition for volatility indicators.

    Args:
        name: Parameter name.
        minimum: Inclusive hard lower bound.
        maximum: Inclusive hard upper bound.
        step: Hard grid step.
        default: Recommended default value.
    Returns:
        ParamDef: Float parameter definition.
    Assumptions:
        Bounds and step are chosen for sensible volatility tuning.
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
    Sort volatility indicator definitions by stable identifier.

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
