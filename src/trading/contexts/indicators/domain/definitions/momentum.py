"""
Hard indicator definitions for momentum group.

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
    Return hard momentum-family definitions in deterministic order.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered momentum indicator definitions.
    Assumptions:
        Stochastic uses high/low/close fixed inputs with three smoothing windows.
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
        default=14,
    )
    k_window = ParamDef(
        name="k_window",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=500,
        step=1,
        default=14,
    )
    d_window = ParamDef(
        name="d_window",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=200,
        step=1,
        default=3,
    )
    smoothing = ParamDef(
        name="smoothing",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=100,
        step=1,
        default=3,
    )

    return (
        IndicatorDef(
            indicator_id=IndicatorId("momentum.rsi"),
            title="Relative Strength Index",
            inputs=_PRICE_SOURCES,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("rsi",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.roc"),
            title="Rate of Change",
            inputs=_PRICE_SOURCES,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("roc",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.stoch"),
            title="Stochastic Oscillator",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(k_window, d_window, smoothing),
            axes=("k_window", "d_window", "smoothing"),
            output=OutputSpec(names=("k", "d")),
        ),
    )
