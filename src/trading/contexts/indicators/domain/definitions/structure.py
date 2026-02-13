"""
Hard indicator definitions for structure/normalization group.

Docs: docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
Related: docs/architecture/indicators/indicators_formula.yaml,
  trading.contexts.indicators.domain.entities.indicator_def,
  trading.contexts.indicators.adapters.outbound.compute_numba.kernels.structure,
  configs/prod/indicators.yaml
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

_OHLC_INPUTS = (
    InputSeries.OPEN,
    InputSeries.HIGH,
    InputSeries.LOW,
    InputSeries.CLOSE,
)


def defs() -> tuple[IndicatorDef, ...]:
    """
    Return hard structure-family definitions sorted by indicator_id.

    Docs: docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
    Related:
      docs/architecture/indicators/indicators_formula.yaml,
      configs/dev/indicators.yaml,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/structure.py

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered structure indicator definitions.
    Assumptions:
        Multi-output indicators keep v1 primary-output compute semantics and wrapper ids expose
        additional outputs without changing API shapes.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = _window(default=20)

    items = (
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_body"),
            title="Candle Body",
            inputs=_OHLC_INPUTS,
            params=(),
            axes=(),
            output=OutputSpec(names=("body",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_body_atr"),
            title="Candle Body ATR-Normalized",
            inputs=_OHLC_INPUTS,
            params=(_window_named(name="atr_window", default=14),),
            axes=("atr_window",),
            output=OutputSpec(names=("body_atr",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_body_pct"),
            title="Candle Body Percent of Range",
            inputs=_OHLC_INPUTS,
            params=(),
            axes=(),
            output=OutputSpec(names=("body_pct",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_lower_wick"),
            title="Candle Lower Wick",
            inputs=_OHLC_INPUTS,
            params=(),
            axes=(),
            output=OutputSpec(names=("lower_wick",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_lower_wick_atr"),
            title="Candle Lower Wick ATR-Normalized",
            inputs=_OHLC_INPUTS,
            params=(_window_named(name="atr_window", default=14),),
            axes=("atr_window",),
            output=OutputSpec(names=("lower_wick_atr",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_lower_wick_pct"),
            title="Candle Lower Wick Percent of Range",
            inputs=_OHLC_INPUTS,
            params=(),
            axes=(),
            output=OutputSpec(names=("lower_wick_pct",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_range"),
            title="Candle Range",
            inputs=_OHLC_INPUTS,
            params=(),
            axes=(),
            output=OutputSpec(names=("range",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_range_atr"),
            title="Candle Range ATR-Normalized",
            inputs=_OHLC_INPUTS,
            params=(_window_named(name="atr_window", default=14),),
            axes=("atr_window",),
            output=OutputSpec(names=("range_atr",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_stats"),
            title="Candle Body and Wicks",
            inputs=_OHLC_INPUTS,
            params=(),
            axes=(),
            output=OutputSpec(
                names=(
                    "body",
                    "range",
                    "upper_wick",
                    "lower_wick",
                    "body_pct",
                    "upper_wick_pct",
                    "lower_wick_pct",
                )
            ),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_stats_atr_norm"),
            title="ATR-Normalized Candle Stats",
            inputs=_OHLC_INPUTS,
            params=(_window_named(name="atr_window", default=14),),
            axes=("atr_window",),
            output=OutputSpec(
                names=(
                    "body_atr",
                    "range_atr",
                    "upper_wick_atr",
                    "lower_wick_atr",
                )
            ),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_upper_wick"),
            title="Candle Upper Wick",
            inputs=_OHLC_INPUTS,
            params=(),
            axes=(),
            output=OutputSpec(names=("upper_wick",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_upper_wick_atr"),
            title="Candle Upper Wick ATR-Normalized",
            inputs=_OHLC_INPUTS,
            params=(_window_named(name="atr_window", default=14),),
            axes=("atr_window",),
            output=OutputSpec(names=("upper_wick_atr",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.candle_upper_wick_pct"),
            title="Candle Upper Wick Percent of Range",
            inputs=_OHLC_INPUTS,
            params=(),
            axes=(),
            output=OutputSpec(names=("upper_wick_pct",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.distance_to_ma_norm"),
            title="Distance to EMA in ATR Units",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("distance_to_ma_norm",)),
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
            indicator_id=IndicatorId("structure.pivot_high"),
            title="Pivot High",
            inputs=(InputSeries.HIGH, InputSeries.LOW),
            params=(
                _window_named(name="left", default=5),
                _window_named(name="right", default=5),
            ),
            axes=("left", "right"),
            output=OutputSpec(names=("pivot_high",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("structure.pivot_low"),
            title="Pivot Low",
            inputs=(InputSeries.HIGH, InputSeries.LOW),
            params=(
                _window_named(name="left", default=5),
                _window_named(name="right", default=5),
            ),
            axes=("left", "right"),
            output=OutputSpec(names=("pivot_low",)),
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
