"""
Hard indicator definitions for volatility group.

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
    Return hard volatility-family definitions in deterministic order.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered volatility indicator definitions.
    Assumptions:
        TR/ATR use high-low-close inputs (including previous close for TR semantics).
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = ParamDef(
        name="window",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=500,
        step=1,
        default=20,
    )
    mult = ParamDef(
        name="mult",
        kind=ParamKind.FLOAT,
        hard_min=0.1,
        hard_max=10.0,
        step=0.05,
        default=2.0,
    )

    return (
        IndicatorDef(
            indicator_id=IndicatorId("volatility.tr"),
            title="True Range",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(),
            axes=(),
            output=OutputSpec(names=("tr",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.atr"),
            title="Average True Range",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("atr",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.stddev"),
            title="Rolling Standard Deviation",
            inputs=_PRICE_SOURCES,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("stddev",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volatility.bbands"),
            title="Bollinger Bands",
            inputs=_PRICE_SOURCES,
            params=(window, mult),
            axes=("source", "window", "mult"),
            output=OutputSpec(names=("basis", "upper", "lower")),
        ),
    )
