from __future__ import annotations

from trading.contexts.backtest.domain.value_objects import build_backtest_variant_key_v1
from trading.contexts.indicators.application.dto import (
    IndicatorVariantSelection,
    build_variant_key_v1,
)


def test_build_backtest_variant_key_v1_normalizes_signals_order_and_case() -> None:
    """
    Verify backtest variant key is stable for semantically equal `signals` mappings.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Signal indicator ids and param names are normalized to lowercase and sorted.
    Raises:
        AssertionError: If variant key depends on insertion order or key casing.
    Side Effects:
        None.
    """
    key_a = build_backtest_variant_key_v1(
        indicator_variant_key="b" * 64,
        direction_mode="long-short",
        sizing_mode="all_in",
        signals={
            "Trend.ADX": {"Short_Delta_Periods": -10, "Long_Delta_Periods": -5},
            "Momentum.RSI": {"short_threshold": 70, "long_threshold": 30},
        },
        risk_params={"sl_enabled": True, "sl_pct": 3.0, "tp_enabled": False, "tp_pct": None},
    )
    key_b = build_backtest_variant_key_v1(
        indicator_variant_key="b" * 64,
        direction_mode="LONG-SHORT",
        sizing_mode="ALL_IN",
        signals={
            "momentum.rsi": {"long_threshold": 30, "short_threshold": 70},
            "trend.adx": {"long_delta_periods": -5, "short_delta_periods": -10},
        },
        risk_params={"sl_enabled": True, "sl_pct": 3.0, "tp_enabled": False, "tp_pct": None},
    )

    assert key_a == key_b


def test_build_backtest_variant_key_v1_changes_when_signals_change() -> None:
    """
    Verify signal parameter changes produce different deterministic backtest keys.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Signals are part of canonical `build_backtest_variant_key_v1` payload.
    Raises:
        AssertionError: If key remains unchanged after signal-value modification.
    Side Effects:
        None.
    """
    base = build_backtest_variant_key_v1(
        indicator_variant_key="a" * 64,
        direction_mode="long-short",
        sizing_mode="all_in",
        signals={"momentum.rsi": {"long_threshold": 30, "short_threshold": 70}},
    )
    changed = build_backtest_variant_key_v1(
        indicator_variant_key="a" * 64,
        direction_mode="long-short",
        sizing_mode="all_in",
        signals={"momentum.rsi": {"long_threshold": 25, "short_threshold": 70}},
    )

    assert base != changed


def test_indicators_build_variant_key_v1_remains_compute_only() -> None:
    """
    Verify indicators key semantics stay compute-only and independent from backtest signals.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Indicators key is built only from instrument/timeframe/compute selections.
    Raises:
        AssertionError: If indicators key changes due to backtest signal variations.
    Side Effects:
        None.
    """
    selection = (
        IndicatorVariantSelection(
            indicator_id="momentum.rsi",
            inputs={"source": "close"},
            params={"window": 14},
        ),
    )
    indicator_key_a = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1h",
        indicators=selection,
    )
    indicator_key_b = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1h",
        indicators=selection,
    )

    backtest_key_a = build_backtest_variant_key_v1(
        indicator_variant_key=indicator_key_a,
        direction_mode="long-short",
        sizing_mode="all_in",
        signals={"momentum.rsi": {"long_threshold": 30, "short_threshold": 70}},
    )
    backtest_key_b = build_backtest_variant_key_v1(
        indicator_variant_key=indicator_key_a,
        direction_mode="long-short",
        sizing_mode="all_in",
        signals={"momentum.rsi": {"long_threshold": 25, "short_threshold": 75}},
    )

    assert indicator_key_a == indicator_key_b
    assert backtest_key_a != backtest_key_b
