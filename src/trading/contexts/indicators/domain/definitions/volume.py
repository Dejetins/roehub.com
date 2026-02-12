"""
Hard indicator definitions for volume group.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
  docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
Related: trading.contexts.indicators.domain.entities.indicator_def,
  trading.contexts.indicators.domain.entities.param_def,
  trading.contexts.indicators.domain.entities.output_spec,
  docs/architecture/indicators/indicators_formula.yaml
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
    Return hard volume-family definitions sorted by indicator_id.

    Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volume.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py,
      docs/architecture/indicators/indicators_formula.yaml

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered volume indicator definitions.
    Assumptions:
        Volume-based indicators require explicit OHLCV inputs where applicable.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = _window(default=20)

    items = (
        IndicatorDef(
            indicator_id=IndicatorId("volume.ad_line"),
            title="Accumulation/Distribution Line",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE, InputSeries.VOLUME),
            params=(),
            axes=(),
            output=OutputSpec(names=("ad_line",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volume.cmf"),
            title="Chaikin Money Flow",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE, InputSeries.VOLUME),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("cmf",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volume.mfi"),
            title="Money Flow Index",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE, InputSeries.VOLUME),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("mfi",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volume.obv"),
            title="On-Balance Volume",
            inputs=(InputSeries.CLOSE, InputSeries.VOLUME),
            params=(),
            axes=(),
            output=OutputSpec(names=("obv",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volume.volume_sma"),
            title="Volume Moving Average",
            inputs=(InputSeries.VOLUME,),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("volume_sma",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volume.vwap"),
            title="Rolling VWAP",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE, InputSeries.VOLUME),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("vwap",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("volume.vwap_deviation"),
            title="Rolling VWAP Deviation Bands",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE, InputSeries.VOLUME),
            params=(
                _float_param(
                    name="mult",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.01,
                    default=2.0,
                ),
                window,
            ),
            axes=("mult", "window"),
            output=OutputSpec(names=("vwap", "vwap_upper", "vwap_lower", "vwap_stdev")),
        ),
    )
    return _sorted_defs(items)


def _window(*, default: int) -> ParamDef:
    """
    Build default integer window parameter for volume indicators.

    Args:
        default: Recommended default window value.
    Returns:
        ParamDef: Window parameter with shared hard bounds.
    Assumptions:
        Volume windows use hard bounds 2..2000 with step 1.
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


def _float_param(
    *,
    name: str,
    minimum: float,
    maximum: float,
    step: float,
    default: float,
) -> ParamDef:
    """
    Build floating parameter definition for volume indicators.

    Args:
        name: Parameter name.
        minimum: Inclusive hard lower bound.
        maximum: Inclusive hard upper bound.
        step: Hard grid step.
        default: Recommended default value.
    Returns:
        ParamDef: Float parameter definition.
    Assumptions:
        Bounds and step are chosen for deviation-band tuning.
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
    Sort volume indicator definitions by stable identifier.

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
