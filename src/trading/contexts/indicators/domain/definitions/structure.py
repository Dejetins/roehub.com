"""
Hard indicator definitions for structure and normalization group.

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
    Return hard structure-family definitions sorted by indicator_id.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered structure indicator definitions.
    Assumptions:
        Candle-stat outputs are registry-level contracts for future compute epics.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = _window(default=20)

    items = (
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_stats"),
            title="Candle Body and Wicks",
            inputs=(InputSeries.OPEN, InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(),
            axes=(),
            output=OutputSpec(names=("body", "upper_wick", "lower_wick", "range")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_stats_atr_norm"),
            title="ATR-Normalized Candle Stats",
            inputs=(InputSeries.OPEN, InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(_window_named(name="atr_window", default=14),),
            axes=("atr_window",),
            output=OutputSpec(
                names=("body_norm", "upper_wick_norm", "lower_wick_norm", "range_norm")
            ),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.heikin_ashi"),
            title="Heikin-Ashi Candle Transform",
            inputs=(InputSeries.OPEN, InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(),
            axes=(),
            output=OutputSpec(names=("ha_open", "ha_high", "ha_low", "ha_close")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.percent_rank"),
            title="Rolling Percentile Rank",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("percent_rank",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.pivots"),
            title="Pivot High/Low",
            inputs=(InputSeries.HIGH, InputSeries.LOW),
            params=(
                _window_named(name="left", default=5),
                _window_named(name="right", default=5),
            ),
            axes=("left", "right"),
            output=OutputSpec(names=("pivot_high", "pivot_low")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.zscore"),
            title="Rolling Z-Score",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("zscore",)),
        ),
    )
    return _sorted_defs(items)


def _window(*, default: int) -> ParamDef:
    """
    Build default integer window parameter for structure indicators.

    Args:
        default: Recommended default window value.
    Returns:
        ParamDef: Window parameter with shared hard bounds.
    Assumptions:
        Structure windows use hard bounds 2..2000 with step 1.
    Raises:
        ValueError: If ParamDef invariants are violated.
    Side Effects:
        None.
    """
    return _window_named(name="window", default=default)


def _window_named(*, name: str, default: int) -> ParamDef:
    """
    Build named integer parameter with common structure hard bounds.

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


def _sorted_defs(items: tuple[IndicatorDef, ...]) -> tuple[IndicatorDef, ...]:
    """
    Sort structure indicator definitions by stable identifier.

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
