from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from trading.contexts.backtest.application.services import (
    IndicatorSignalEvaluationInputV1,
    aggregate_indicator_signals_v1,
    build_indicator_signal_inputs_from_tensors_v1,
    evaluate_and_aggregate_signals_v1,
    evaluate_indicator_signal_v1,
    expand_indicator_grids_with_signal_dependencies_v1,
    list_signal_rule_registry_v1,
    supported_indicator_ids_for_signals_v1,
)
from trading.contexts.backtest.domain.value_objects import IndicatorSignalsV1, SignalV1
from trading.contexts.indicators.application.dto import CandleArrays, IndicatorTensor, TensorMeta
from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId, Layout
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.shared_kernel.primitives import MarketId, Symbol, Timeframe, TimeRange, UtcTimestamp


def _build_candles(
    *,
    open_values: tuple[float, ...],
    high_values: tuple[float, ...],
    low_values: tuple[float, ...],
    close_values: tuple[float, ...],
    volume_values: tuple[float, ...],
) -> CandleArrays:
    """
    Build deterministic candle arrays fixture aligned on a single synthetic timeline.

    Args:
        open_values: Open series values.
        high_values: High series values.
        low_values: Low series values.
        close_values: Close series values.
        volume_values: Volume series values.
    Returns:
        CandleArrays: Deterministic float32 test fixture.
    Assumptions:
        All tuples have equal length.
    Raises:
        ValueError: If CandleArrays primitive contracts are violated.
    Side Effects:
        None.
    """
    bars = len(close_values)
    ts_open = np.asarray([int(index * 60_000) for index in range(bars)], dtype=np.int64)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 0, bars, tzinfo=timezone.utc)),
        ),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=np.asarray(open_values, dtype=np.float32),
        high=np.asarray(high_values, dtype=np.float32),
        low=np.asarray(low_values, dtype=np.float32),
        close=np.asarray(close_values, dtype=np.float32),
        volume=np.asarray(volume_values, dtype=np.float32),
    )


def test_signal_registry_covers_all_prod_indicator_ids() -> None:
    """
    Verify signal-rule registry supports every indicator id present in prod defaults config.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Prod indicator ids are top-level keys under `defaults` in YAML.
    Raises:
        AssertionError: If at least one prod indicator id is missing in signal registry.
    Side Effects:
        Reads local YAML file from repository workspace.
    """
    config_path = Path("configs/prod/indicators.yaml")
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = parsed["defaults"]
    prod_indicator_ids = sorted(
        indicator_id
        for indicator_id in defaults.keys()
        if isinstance(indicator_id, str) and "." in indicator_id
    )
    supported_ids = set(supported_indicator_ids_for_signals_v1())
    missing = sorted(
        indicator_id
        for indicator_id in prod_indicator_ids
        if indicator_id not in supported_ids
    )
    assert missing == []


def test_compare_price_to_output_uses_selected_source_and_nan_is_neutral() -> None:
    """
    Verify compare-price rule uses selected source series and maps NaN inputs to NEUTRAL.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `inputs.source=high` should compare high series against indicator output.
    Raises:
        AssertionError: If signal labels diverge from expected deterministic mapping.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(10.0, 10.0, 10.0, 10.0),
        high_values=(20.0, 20.0, 20.0, 20.0),
        low_values=(5.0, 5.0, 5.0, 5.0),
        close_values=(8.0, 8.0, 8.0, 8.0),
        volume_values=(100.0, 100.0, 100.0, 100.0),
    )
    indicator_input = IndicatorSignalEvaluationInputV1(
        indicator_id="ma.ema",
        primary_output=np.asarray((7.0, 25.0, np.nan, 5.0), dtype=np.float32),
        indicator_inputs={"source": "high"},
    )

    result = evaluate_indicator_signal_v1(candles=candles, indicator_input=indicator_input)

    assert result.indicator_id == "ma.ema"
    assert result.signals.tolist() == [
        SignalV1.LONG.value,
        SignalV1.SHORT.value,
        SignalV1.NEUTRAL.value,
        SignalV1.LONG.value,
    ]


def test_threshold_band_supports_mean_reversion_and_trend_orientations() -> None:
    """
    Verify threshold-band orientation changes deterministically with threshold ordering.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `long<=short` means mean-reversion orientation; otherwise trend-following.
    Raises:
        AssertionError: If either orientation result mismatches expected labels.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(1.0, 1.0, 1.0, 1.0),
        high_values=(1.0, 1.0, 1.0, 1.0),
        low_values=(1.0, 1.0, 1.0, 1.0),
        close_values=(1.0, 1.0, 1.0, 1.0),
        volume_values=(1.0, 1.0, 1.0, 1.0),
    )
    values = np.asarray((20.0, 50.0, 80.0, np.nan), dtype=np.float32)

    mean_reversion = evaluate_indicator_signal_v1(
        candles=candles,
        indicator_input=IndicatorSignalEvaluationInputV1(
            indicator_id="momentum.rsi",
            primary_output=values,
            signal_params={"long_threshold": 30, "short_threshold": 70},
        ),
    )
    trend_following = evaluate_indicator_signal_v1(
        candles=candles,
        indicator_input=IndicatorSignalEvaluationInputV1(
            indicator_id="trend.aroon",
            primary_output=values,
            signal_params={"long_threshold": 70, "short_threshold": 30},
        ),
    )

    assert mean_reversion.signals.tolist() == [
        SignalV1.LONG.value,
        SignalV1.NEUTRAL.value,
        SignalV1.SHORT.value,
        SignalV1.NEUTRAL.value,
    ]
    assert trend_following.signals.tolist() == [
        SignalV1.SHORT.value,
        SignalV1.NEUTRAL.value,
        SignalV1.LONG.value,
        SignalV1.NEUTRAL.value,
    ]


def test_sign_rule_family_maps_positive_negative_and_nan_values() -> None:
    """
    Verify sign-family rule maps positive to LONG, negative to SHORT, and NaN to NEUTRAL.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Zero value is neutral in sign-family semantics.
    Raises:
        AssertionError: If emitted labels diverge from deterministic mapping.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(1.0, 1.0, 1.0, 1.0),
        high_values=(1.0, 1.0, 1.0, 1.0),
        low_values=(1.0, 1.0, 1.0, 1.0),
        close_values=(1.0, 1.0, 1.0, 1.0),
        volume_values=(1.0, 1.0, 1.0, 1.0),
    )
    result = evaluate_indicator_signal_v1(
        candles=candles,
        indicator_input=IndicatorSignalEvaluationInputV1(
            indicator_id="trend.linreg_slope",
            primary_output=np.asarray((1.0, 0.0, -1.0, np.nan), dtype=np.float32),
        ),
    )

    assert result.signals.tolist() == [
        SignalV1.LONG.value,
        SignalV1.NEUTRAL.value,
        SignalV1.SHORT.value,
        SignalV1.NEUTRAL.value,
    ]


def test_delta_sign_uses_negative_periods_as_lookback_and_neutralizes_conflicts() -> None:
    """
    Verify delta-sign uses abs(period), handles warmup, and neutralizes long/short conflicts.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `long_delta_periods=-1` and `short_delta_periods=-2` use lookbacks 1 and 2.
    Raises:
        AssertionError: If labels violate expected delta semantics.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(1.0, 1.0, 1.0, 1.0),
        high_values=(1.0, 1.0, 1.0, 1.0),
        low_values=(1.0, 1.0, 1.0, 1.0),
        close_values=(1.0, 1.0, 1.0, 1.0),
        volume_values=(1.0, 1.0, 1.0, 1.0),
    )
    result = evaluate_indicator_signal_v1(
        candles=candles,
        indicator_input=IndicatorSignalEvaluationInputV1(
            indicator_id="trend.adx",
            primary_output=np.asarray((10.0, 0.0, 5.0, 6.0), dtype=np.float32),
            signal_params={"long_delta_periods": -1, "short_delta_periods": -2},
        ),
    )

    assert result.signals.tolist() == [
        SignalV1.NEUTRAL.value,
        SignalV1.NEUTRAL.value,
        SignalV1.NEUTRAL.value,
        SignalV1.LONG.value,
    ]


def test_compare_volume_to_output_rule_family() -> None:
    """
    Verify compare-volume rule compares candle volume against indicator output.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Equal values map to neutral and NaN values map to neutral.
    Raises:
        AssertionError: If labels diverge from expected deterministic values.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(1.0, 1.0, 1.0, 1.0),
        high_values=(1.0, 1.0, 1.0, 1.0),
        low_values=(1.0, 1.0, 1.0, 1.0),
        close_values=(1.0, 1.0, 1.0, 1.0),
        volume_values=(110.0, 80.0, 80.0, 100.0),
    )
    result = evaluate_indicator_signal_v1(
        candles=candles,
        indicator_input=IndicatorSignalEvaluationInputV1(
            indicator_id="volume.volume_sma",
            primary_output=np.asarray((100.0, 90.0, 80.0, np.nan), dtype=np.float32),
        ),
    )

    assert result.signals.tolist() == [
        SignalV1.LONG.value,
        SignalV1.SHORT.value,
        SignalV1.NEUTRAL.value,
        SignalV1.NEUTRAL.value,
    ]


def test_candle_body_direction_rule_family_uses_threshold_and_candle_direction() -> None:
    """
    Verify candle-body-direction rule checks both body magnitude and open/close direction.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Bars below threshold remain neutral regardless of direction.
    Raises:
        AssertionError: If emitted labels violate candle-body-direction contract.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(1.0, 2.0, 3.0, 4.0),
        high_values=(2.0, 3.0, 4.0, 5.0),
        low_values=(0.5, 1.0, 2.0, 3.0),
        close_values=(2.0, 1.0, 3.5, 5.0),
        volume_values=(1.0, 1.0, 1.0, 1.0),
    )
    result = evaluate_indicator_signal_v1(
        candles=candles,
        indicator_input=IndicatorSignalEvaluationInputV1(
            indicator_id="structure.candle_stats_atr_norm",
            primary_output=np.asarray((0.4, 0.7, 0.8, np.nan), dtype=np.float32),
            signal_params={"min_body_atr": 0.5},
        ),
    )

    assert result.signals.tolist() == [
        SignalV1.NEUTRAL.value,
        SignalV1.SHORT.value,
        SignalV1.LONG.value,
        SignalV1.NEUTRAL.value,
    ]


def test_pivot_events_rule_family_uses_wrapper_dependency_outputs() -> None:
    """
    Verify pivot-events rule reads both wrapper outputs and neutralizes same-bar conflicts.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Finite pivot-low emits LONG and finite pivot-high emits SHORT.
    Raises:
        AssertionError: If labels diverge from expected pivot-event mapping.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(1.0, 1.0, 1.0, 1.0),
        high_values=(1.0, 1.0, 1.0, 1.0),
        low_values=(1.0, 1.0, 1.0, 1.0),
        close_values=(1.0, 1.0, 1.0, 1.0),
        volume_values=(1.0, 1.0, 1.0, 1.0),
    )
    result = evaluate_indicator_signal_v1(
        candles=candles,
        indicator_input=IndicatorSignalEvaluationInputV1(
            indicator_id="structure.pivots",
            primary_output=np.asarray((np.nan, np.nan, np.nan, np.nan), dtype=np.float32),
            dependency_outputs={
                "structure.pivot_low": np.asarray((np.nan, 1.0, np.nan, 2.0), dtype=np.float32),
                "structure.pivot_high": np.asarray((np.nan, np.nan, 3.0, 2.0), dtype=np.float32),
            },
        ),
    )

    assert result.signals.tolist() == [
        SignalV1.NEUTRAL.value,
        SignalV1.LONG.value,
        SignalV1.SHORT.value,
        SignalV1.NEUTRAL.value,
    ]


def test_threshold_centered_rule_family_for_trend_vortex() -> None:
    """
    Verify centered-threshold family emits long/short around center with absolute threshold.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        For trend.vortex center is fixed at `1.0`.
    Raises:
        AssertionError: If labels diverge from expected centered-threshold behavior.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(1.0, 1.0, 1.0, 1.0),
        high_values=(1.0, 1.0, 1.0, 1.0),
        low_values=(1.0, 1.0, 1.0, 1.0),
        close_values=(1.0, 1.0, 1.0, 1.0),
        volume_values=(1.0, 1.0, 1.0, 1.0),
    )
    result = evaluate_indicator_signal_v1(
        candles=candles,
        indicator_input=IndicatorSignalEvaluationInputV1(
            indicator_id="trend.vortex",
            primary_output=np.asarray((1.3, 1.0, 0.6, np.nan), dtype=np.float32),
            signal_params={"abs_threshold": 0.2},
        ),
    )

    assert result.signals.tolist() == [
        SignalV1.LONG.value,
        SignalV1.NEUTRAL.value,
        SignalV1.SHORT.value,
        SignalV1.NEUTRAL.value,
    ]


def test_and_aggregation_returns_expected_long_short_and_conflict_metadata() -> None:
    """
    Verify AND aggregation emits deterministic final long/short vectors and conflict counter.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Empty indicator set leads to vacuous true/true conflict resolved to neutral per bar.
    Raises:
        AssertionError: If aggregation output diverges from v1 contract.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(1.0, 1.0, 1.0, 1.0),
        high_values=(1.0, 1.0, 1.0, 1.0),
        low_values=(1.0, 1.0, 1.0, 1.0),
        close_values=(1.0, 1.0, 1.0, 1.0),
        volume_values=(1.0, 1.0, 1.0, 1.0),
    )
    non_empty = aggregate_indicator_signals_v1(
        candles=candles,
        indicator_signals=(
            IndicatorSignalsV1(
                indicator_id="b.indicator",
                signals=np.asarray(
                    (
                        SignalV1.LONG.value,
                        SignalV1.SHORT.value,
                        SignalV1.NEUTRAL.value,
                        SignalV1.LONG.value,
                    ),
                    dtype=np.str_,
                ),
            ),
            IndicatorSignalsV1(
                indicator_id="a.indicator",
                signals=np.asarray(
                    (
                        SignalV1.LONG.value,
                        SignalV1.SHORT.value,
                        SignalV1.LONG.value,
                        SignalV1.LONG.value,
                    ),
                    dtype=np.str_,
                ),
            ),
        ),
    )
    empty = aggregate_indicator_signals_v1(candles=candles, indicator_signals=())

    assert [item.indicator_id for item in non_empty.per_indicator_signals] == [
        "a.indicator",
        "b.indicator",
    ]
    assert non_empty.final_signal.tolist() == [
        SignalV1.LONG.value,
        SignalV1.SHORT.value,
        SignalV1.NEUTRAL.value,
        SignalV1.LONG.value,
    ]
    assert non_empty.final_long.tolist() == [True, False, False, True]
    assert non_empty.final_short.tolist() == [False, True, False, False]
    assert non_empty.conflicting_signals == 0

    assert empty.final_signal.tolist() == [
        SignalV1.NEUTRAL.value,
        SignalV1.NEUTRAL.value,
        SignalV1.NEUTRAL.value,
        SignalV1.NEUTRAL.value,
    ]
    assert empty.final_long.tolist() == [False, False, False, False]
    assert empty.final_short.tolist() == [False, False, False, False]
    assert empty.conflicting_signals == 4


def test_evaluate_and_aggregate_is_order_deterministic() -> None:
    """
    Verify evaluation+aggregation output is stable regardless of input ordering.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Engine sorts indicator payloads by indicator_id before aggregation.
    Raises:
        AssertionError: If order-dependent behavior appears.
    Side Effects:
        None.
    """
    candles = _build_candles(
        open_values=(1.0, 1.0, 1.0),
        high_values=(3.0, 3.0, 3.0),
        low_values=(0.5, 0.5, 0.5),
        close_values=(2.0, 2.0, 2.0),
        volume_values=(1.0, 1.0, 1.0),
    )
    input_a = IndicatorSignalEvaluationInputV1(
        indicator_id="trend.linreg_slope",
        primary_output=np.asarray((1.0, -1.0, 1.0), dtype=np.float32),
    )
    input_b = IndicatorSignalEvaluationInputV1(
        indicator_id="ma.sma",
        primary_output=np.asarray((1.0, 4.0, 1.0), dtype=np.float32),
        indicator_inputs={"source": "close"},
    )

    left = evaluate_and_aggregate_signals_v1(
        candles=candles,
        indicator_inputs=(input_b, input_a),
    )
    right = evaluate_and_aggregate_signals_v1(
        candles=candles,
        indicator_inputs=(input_a, input_b),
    )

    assert [item.indicator_id for item in left.per_indicator_signals] == [
        "ma.sma",
        "trend.linreg_slope",
    ]
    assert [item.indicator_id for item in right.per_indicator_signals] == [
        "ma.sma",
        "trend.linreg_slope",
    ]
    assert left.final_signal.tolist() == right.final_signal.tolist()
    assert left.final_long.tolist() == right.final_long.tolist()
    assert left.final_short.tolist() == right.final_short.tolist()
    assert left.conflicting_signals == right.conflicting_signals


def test_expand_indicator_grids_adds_pivot_wrappers_deterministically() -> None:
    """
    Verify compute-plan expansion appends required pivot wrapper dependencies once.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `structure.pivots` dependencies are `structure.pivot_high` and `structure.pivot_low`.
    Raises:
        AssertionError: If compute-plan ids diverge from deterministic expected sequence.
    Side Effects:
        None.
    """
    expanded = expand_indicator_grids_with_signal_dependencies_v1(
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("structure.pivots"),
                params={
                    "left": ExplicitValuesSpec(name="left", values=(3,)),
                    "right": ExplicitValuesSpec(name="right", values=(2,)),
                },
            ),
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
                source=ExplicitValuesSpec(name="source", values=("close",)),
            ),
        ),
    )

    assert tuple(grid.indicator_id.value for grid in expanded) == (
        "ma.sma",
        "structure.pivots",
        "structure.pivot_high",
        "structure.pivot_low",
    )


def test_build_inputs_from_tensors_uses_primary_output_and_registry_listing_is_sorted() -> None:
    """
    Verify tensor wrapper extracts requested variant and registry debug listing is sorted.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        TIME_MAJOR tensor extraction uses `values[:, variant_index]`.
    Raises:
        AssertionError: If extracted series or listing order are not deterministic.
    Side Effects:
        None.
    """
    tensor = IndicatorTensor(
        indicator_id=IndicatorId("ma.sma"),
        layout=Layout.TIME_MAJOR,
        axes=(AxisDef(name="window", values_int=(10, 20)),),
        values=np.asarray(
            (
                (1.0, 10.0),
                (2.0, 20.0),
                (3.0, 30.0),
            ),
            dtype=np.float32,
        ),
        meta=TensorMeta(t=3, variants=2),
    )
    inputs = build_indicator_signal_inputs_from_tensors_v1(
        tensors={"ma.sma": tensor},
        indicator_inputs={"ma.sma": {"source": "close"}},
        variant_index_by_indicator={"ma.sma": 1},
    )
    registry_pairs = list_signal_rule_registry_v1()

    assert len(inputs) == 1
    assert inputs[0].primary_output.tolist() == [10.0, 20.0, 30.0]
    assert inputs[0].indicator_inputs["source"] == "close"
    assert registry_pairs == tuple(sorted(registry_pairs, key=lambda item: item[0]))
