"""
Indicators hard-definition registry grouped by domain area.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
Related: trading.contexts.indicators.domain.entities.indicator_def
"""

from __future__ import annotations

from trading.contexts.indicators.domain.entities import IndicatorDef

from .ma import defs as ma_defs
from .momentum import defs as momentum_defs
from .structure import defs as structure_defs
from .trend import defs as trend_defs
from .volatility import defs as volatility_defs
from .volume import defs as volume_defs


def all_defs() -> tuple[IndicatorDef, ...]:
    """
    Return the full hard-definition set in stable cross-group order.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable concatenation ordered as
            ma, trend, volatility, momentum, volume, structure.
    Assumptions:
        Each group-level defs() already returns deterministic tuples.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (
        *ma_defs(),
        *trend_defs(),
        *volatility_defs(),
        *momentum_defs(),
        *volume_defs(),
        *structure_defs(),
    )


__all__ = ["all_defs"]
