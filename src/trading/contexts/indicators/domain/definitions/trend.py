"""
Hard indicator definitions for trend group.

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


def defs() -> tuple[IndicatorDef, ...]:
    """
    Return hard trend-family definitions in deterministic order.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered trend indicator definitions.
    Assumptions:
        ADX and Donchian contracts are registry-compatible and compute-ready later.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    adx_window = ParamDef(
        name="window",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=500,
        step=1,
        default=14,
    )
    adx_smoothing = ParamDef(
        name="smoothing",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=500,
        step=1,
        default=14,
    )
    donchian_window = ParamDef(
        name="window",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=1_000,
        step=1,
        default=20,
    )

    return (
        IndicatorDef(
            indicator_id=IndicatorId("trend.adx"),
            title="Average Directional Index",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(adx_window, adx_smoothing),
            axes=("window", "smoothing"),
            output=OutputSpec(names=("adx", "plus_di", "minus_di")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("trend.donchian"),
            title="Donchian Channel",
            inputs=(InputSeries.HIGH, InputSeries.LOW),
            params=(donchian_window,),
            axes=("window",),
            output=OutputSpec(names=("upper", "middle", "lower")),
        ),
    )
