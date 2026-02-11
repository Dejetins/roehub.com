"""
Hard indicator definitions for volume group.

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
    Return hard volume-family definitions in deterministic order.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered volume indicator definitions.
    Assumptions:
        Rolling VWAP uses typical-price inputs and volume as mandatory series.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = ParamDef(
        name="window",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=1_000,
        step=1,
        default=20,
    )

    return (
        IndicatorDef(
            indicator_id=IndicatorId("volume.vwap"),
            title="Rolling VWAP",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE, InputSeries.VOLUME),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("vwap",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volume.obv"),
            title="On-Balance Volume",
            inputs=(InputSeries.CLOSE, InputSeries.VOLUME),
            params=(),
            axes=(),
            output=OutputSpec(names=("obv",)),
        ),
    )
