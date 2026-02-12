"""
Hard indicator definitions for moving-average group.

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
    InputSeries.HL2,
    InputSeries.HLC3,
    InputSeries.OHLC4,
    InputSeries.LOW,
    InputSeries.HIGH,
    InputSeries.OPEN,
)


def defs() -> tuple[IndicatorDef, ...]:
    """
    Return hard MA-family definitions sorted by indicator_id.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered MA indicator definitions.
    Assumptions:
        Indicator ids are stable and unique in the MA group.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = _window(default=20)

    items = (
        IndicatorDef(
            indicator_id=IndicatorId("ma.dema"),
            title="Double Exponential Moving Average",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.ema"),
            title="Exponential Moving Average",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.hma"),
            title="Hull Moving Average",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.lwma"),
            title="Linear Weighted Moving Average",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.rma"),
            title="RMA (SMMA)",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.sma"),
            title="Simple Moving Average",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.tema"),
            title="Triple Exponential Moving Average",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.vwma"),
            title="Volume Weighted Moving Average",
            inputs=(InputSeries.CLOSE, InputSeries.VOLUME),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.wma"),
            title="Weighted Moving Average",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.zlema"),
            title="Zero-Lag Exponential Moving Average",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
    )
    return _sorted_defs(items)


def _window(*, default: int) -> ParamDef:
    """
    Build common integer window parameter definition for MA indicators.

    Args:
        default: Recommended default window value.
    Returns:
        ParamDef: Shared integer window parameter.
    Assumptions:
        MA windows use hard bounds 2..2000 with unit grid.
    Raises:
        ValueError: If ParamDef invariants are violated.
    Side Effects:
        None.
    """
    return ParamDef(
        name="window",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=2_000,
        step=1,
        default=default,
    )


def _sorted_defs(items: tuple[IndicatorDef, ...]) -> tuple[IndicatorDef, ...]:
    """
    Sort MA indicator definitions by stable identifier.

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
