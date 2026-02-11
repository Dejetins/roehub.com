"""
Hard indicator definitions for momentum group.

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
    InputSeries.HLC3,
    InputSeries.OHLC4,
    InputSeries.LOW,
    InputSeries.HIGH,
    InputSeries.OPEN,
)


def defs() -> tuple[IndicatorDef, ...]:
    """
    Return hard momentum-family definitions sorted by indicator_id.

    Args:
        None.
    Returns:
        tuple[IndicatorDef, ...]: Immutable ordered momentum indicator definitions.
    Assumptions:
        Source-based indicators expose configurable source axis.
    Raises:
        ValueError: If any definition violates domain invariants.
    Side Effects:
        None.
    """
    window = _window(default=14)

    items = (
        IndicatorDef(
            indicator_id=IndicatorId("momentum.cci"),
            title="Commodity Channel Index",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("cci",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.fisher"),
            title="Fisher Transform",
            inputs=(InputSeries.HIGH, InputSeries.LOW),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("fisher", "trigger")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.macd"),
            title="MACD",
            inputs=_SOURCE_INPUTS,
            params=(
                _window_named(name="fast_window", default=12),
                _window_named(name="slow_window", default=26),
                _window_named(name="signal_window", default=9),
            ),
            axes=("fast_window", "signal_window", "slow_window", "source"),
            output=OutputSpec(names=("macd", "signal", "hist")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.ppo"),
            title="Percentage Price Oscillator",
            inputs=_SOURCE_INPUTS,
            params=(
                _window_named(name="fast_window", default=12),
                _window_named(name="slow_window", default=26),
                _window_named(name="signal_window", default=9),
            ),
            axes=("fast_window", "signal_window", "slow_window", "source"),
            output=OutputSpec(names=("ppo", "signal", "hist")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.roc"),
            title="Rate of Change",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("roc",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.rsi"),
            title="Relative Strength Index",
            inputs=_SOURCE_INPUTS,
            params=(window,),
            axes=("source", "window"),
            output=OutputSpec(names=("rsi",)),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.stoch"),
            title="Stochastic Oscillator",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(
                _window_named(name="d_window", default=3),
                _window_named(name="k_window", default=14),
                _window_named(name="smoothing", default=3),
            ),
            axes=("d_window", "k_window", "smoothing"),
            output=OutputSpec(names=("k", "d")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.stoch_rsi"),
            title="Stochastic RSI",
            inputs=_SOURCE_INPUTS,
            params=(
                _window_named(name="d_window", default=3),
                _window_named(name="k_window", default=14),
                _window_named(name="rsi_window", default=14),
                _window_named(name="smoothing", default=3),
            ),
            axes=("d_window", "k_window", "rsi_window", "smoothing", "source"),
            output=OutputSpec(names=("k", "d")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.trix"),
            title="TRIX",
            inputs=_SOURCE_INPUTS,
            params=(
                _window_named(name="signal_window", default=9),
                _window_named(name="window", default=15),
            ),
            axes=("signal_window", "source", "window"),
            output=OutputSpec(names=("trix", "signal", "hist")),
        ),
        IndicatorDef(
            indicator_id=IndicatorId("momentum.williams_r"),
            title="Williams %R",
            inputs=(InputSeries.HIGH, InputSeries.LOW, InputSeries.CLOSE),
            params=(window,),
            axes=("window",),
            output=OutputSpec(names=("williams_r",)),
        ),
    )
    return _sorted_defs(items)


def _window(*, default: int) -> ParamDef:
    """
    Build default integer window parameter for momentum indicators.

    Args:
        default: Recommended default window value.
    Returns:
        ParamDef: Window parameter with shared hard bounds.
    Assumptions:
        Momentum windows use hard bounds 2..2000 with step 1.
    Raises:
        ValueError: If ParamDef invariants are violated.
    Side Effects:
        None.
    """
    return _window_named(name="window", default=default)


def _window_named(*, name: str, default: int) -> ParamDef:
    """
    Build named integer parameter with common momentum hard bounds.

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
    Sort momentum indicator definitions by stable identifier.

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
