from __future__ import annotations

from trading.contexts.indicators.application.dto import (
    IndicatorVariantSelection,
    build_variant_key_v1,
)


def _base_variant_indicators() -> tuple[IndicatorVariantSelection, ...]:
    """
    Build deterministic baseline explicit indicator selections for key tests.

    Args:
        None.
    Returns:
        tuple[IndicatorVariantSelection, ...]: Baseline indicator selection tuple.
    Assumptions:
        Values are valid explicit configurations.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (
        IndicatorVariantSelection(
            indicator_id="ma.sma",
            inputs={"source": "close"},
            params={"window": 36},
        ),
        IndicatorVariantSelection(
            indicator_id="momentum.rsi",
            inputs={"source": "close"},
            params={"window": 14},
        ),
    )


def test_variant_key_same_payload_produces_same_hash() -> None:
    """
    Verify deterministic key for identical payloads.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical serialization is deterministic for identical inputs.
    Raises:
        AssertionError: If hashes differ for equal payload.
    Side Effects:
        None.
    """
    indicators = _base_variant_indicators()

    first = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1m",
        indicators=indicators,
    )
    second = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1m",
        indicators=indicators,
    )

    assert first == second


def test_variant_key_changes_when_instrument_changes() -> None:
    """
    Verify key sensitivity to instrument_id.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        variant_key v1 is bound to instrument_id.
    Raises:
        AssertionError: If hash does not change across instruments.
    Side Effects:
        None.
    """
    indicators = _base_variant_indicators()

    btc_key = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1m",
        indicators=indicators,
    )
    eth_key = build_variant_key_v1(
        instrument_id="1:ETHUSDT",
        timeframe="1m",
        indicators=indicators,
    )

    assert btc_key != eth_key


def test_variant_key_changes_when_timeframe_changes() -> None:
    """
    Verify key sensitivity to timeframe.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        variant_key v1 is bound to timeframe.
    Raises:
        AssertionError: If hash does not change across timeframes.
    Side Effects:
        None.
    """
    indicators = _base_variant_indicators()

    one_minute_key = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1m",
        indicators=indicators,
    )
    five_minutes_key = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="5m",
        indicators=indicators,
    )

    assert one_minute_key != five_minutes_key


def test_variant_key_changes_when_param_value_changes() -> None:
    """
    Verify key sensitivity to explicit parameter values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Changing explicit params must produce different key.
    Raises:
        AssertionError: If hash does not change after param update.
    Side Effects:
        None.
    """
    base_indicators = _base_variant_indicators()
    changed_indicators = (
        IndicatorVariantSelection(
            indicator_id="ma.sma",
            inputs={"source": "close"},
            params={"window": 50},
        ),
        base_indicators[1],
    )

    base_key = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1m",
        indicators=base_indicators,
    )
    changed_key = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1m",
        indicators=changed_indicators,
    )

    assert base_key != changed_key


def test_variant_key_ignores_input_mapping_and_indicator_ordering() -> None:
    """
    Verify canonicalization is stable for unordered dict and tuple ordering.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical payload sorts indicators and key/value pairs.
    Raises:
        AssertionError: If ordering differences change produced hash.
    Side Effects:
        None.
    """
    ordered = (
        IndicatorVariantSelection(
            indicator_id="ma.sma",
            inputs={"source": "close", "signal": "line"},
            params={"window": 36, "offset": 0},
        ),
        IndicatorVariantSelection(
            indicator_id="momentum.rsi",
            inputs={"source": "close"},
            params={"window": 14},
        ),
    )

    reversed_and_unordered = (
        IndicatorVariantSelection(
            indicator_id="momentum.rsi",
            inputs={"source": "close"},
            params={"window": 14},
        ),
        IndicatorVariantSelection(
            indicator_id="ma.sma",
            inputs={"signal": "line", "source": "close"},
            params={"offset": 0, "window": 36},
        ),
    )

    first_key = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1m",
        indicators=ordered,
    )
    second_key = build_variant_key_v1(
        instrument_id="1:BTCUSDT",
        timeframe="1m",
        indicators=reversed_and_unordered,
    )

    assert first_key == second_key
