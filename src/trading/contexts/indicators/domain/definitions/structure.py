"""
Hard indicator definitions for market-structure group.

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
    Return hard structure-family definitions in deterministic order.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered structure indicator definitions.
    Assumptions:
        Pivot detection uses symmetric left/right integer lookback windows.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    left = ParamDef(
        name="left",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=200,
        step=1,
        default=5,
    )
    right = ParamDef(
        name="right",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=200,
        step=1,
        default=5,
    )

    return (
        IndicatorDef(
            indicator_id=IndicatorId("structure.pivots"),
            title="Pivot High/Low",
            inputs=(InputSeries.HIGH, InputSeries.LOW),
            params=(left, right),
            axes=("left", "right"),
            output=OutputSpec(names=("pivot_high", "pivot_low")),
        ),
    )
