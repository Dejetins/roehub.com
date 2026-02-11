"""
Hard indicator definitions for moving-average group.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
Related: trading.contexts.indicators.domain.entities.indicator_def,
  trading.contexts.indicators.domain.entities.param_def
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

_PRICE_SOURCES = (
    InputSeries.CLOSE,
    InputSeries.OPEN,
    InputSeries.HIGH,
    InputSeries.LOW,
    InputSeries.HL2,
    InputSeries.HLC3,
    InputSeries.OHLC4,
)


def defs() -> tuple[IndicatorDef, ...]:
    """
    Return hard MA-family definitions in deterministic order.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered MA indicator definitions.
    Assumptions:
        Indicator ids are stable and unique in registry namespace.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = ParamDef(
        name="window",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=2_000,
        step=1,
        default=20,
    )

    return (
        IndicatorDef(
            indicator_id=IndicatorId("ma.sma"),
            title="Simple Moving Average",
            inputs=_PRICE_SOURCES,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.ema"),
            title="Exponential Moving Average",
            inputs=_PRICE_SOURCES,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.wma"),
            title="Weighted Moving Average",
            inputs=_PRICE_SOURCES,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.hma"),
            title="Hull Moving Average",
            inputs=_PRICE_SOURCES,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("value",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("ma.rma"),
            title="RMA (SMMA)",
            inputs=_PRICE_SOURCES,
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
    )
