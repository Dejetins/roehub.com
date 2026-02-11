from datetime import datetime, timezone

import numpy as np
import pytest

from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import MarketId, Symbol, Timeframe, TimeRange, UtcTimestamp


def _time_range() -> TimeRange:
    """
    Build deterministic time range metadata for candle payload tests.

    Args:
        None.
    Returns:
        TimeRange: Fixed UTC half-open range used by fixture payloads.
    Assumptions:
        Shared-kernel TimeRange validates start < end.
    Raises:
        None.
    Side Effects:
        None.
    """
    start = UtcTimestamp(datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 11, 10, 3, 0, tzinfo=timezone.utc))
    return TimeRange(start=start, end=end)


def test_candle_arrays_accepts_valid_dense_payload() -> None:
    """
    Verify acceptance of valid dense arrays with required dtypes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        OHLCV arrays are 1D float32 and timestamps are 1D int64.
    Raises:
        AssertionError: If valid payload is rejected or length assertion fails.
    Side Effects:
        None.
    """
    candles = CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=_time_range(),
        timeframe=Timeframe("1m"),
        ts_open=np.array([1_700_000_000, 1_700_000_060, 1_700_000_120], dtype=np.int64),
        open=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        high=np.array([101.0, 102.0, 103.0], dtype=np.float32),
        low=np.array([99.0, 100.0, 101.0], dtype=np.float32),
        close=np.array([100.5, 101.5, 102.5], dtype=np.float32),
        volume=np.array([10.0, 11.0, 12.0], dtype=np.float32),
    )
    assert candles.close.shape[0] == 3


def test_candle_arrays_rejects_length_mismatch() -> None:
    """
    Verify rejection when an OHLCV array length mismatches baseline timeline.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        All series must share identical length.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        CandleArrays(
            market_id=MarketId(1),
            symbol=Symbol("BTCUSDT"),
            time_range=_time_range(),
            timeframe=Timeframe("1m"),
            ts_open=np.array([1_700_000_000, 1_700_000_060], dtype=np.int64),
            open=np.array([100.0, 101.0], dtype=np.float32),
            high=np.array([101.0, 102.0], dtype=np.float32),
            low=np.array([99.0, 100.0], dtype=np.float32),
            close=np.array([100.5], dtype=np.float32),
            volume=np.array([10.0, 11.0], dtype=np.float32),
        )


def test_candle_arrays_rejects_invalid_dtypes() -> None:
    """
    Verify rejection when timestamp or OHLCV dtypes violate contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Contract requires int64 timestamps and float32 OHLCV.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        CandleArrays(
            market_id=MarketId(1),
            symbol=Symbol("BTCUSDT"),
            time_range=_time_range(),
            timeframe=Timeframe("1m"),
            ts_open=np.array([1_700_000_000, 1_700_000_060], dtype=np.int32),
            open=np.array([100.0, 101.0], dtype=np.float32),
            high=np.array([101.0, 102.0], dtype=np.float32),
            low=np.array([99.0, 100.0], dtype=np.float32),
            close=np.array([100.5, 101.5], dtype=np.float32),
            volume=np.array([10.0, 11.0], dtype=np.float32),
        )


def test_candle_arrays_rejects_non_1d_arrays() -> None:
    """
    Verify rejection when any payload array is not one-dimensional.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Contract requires every timestamp and OHLCV array to be 1D.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        CandleArrays(
            market_id=MarketId(1),
            symbol=Symbol("BTCUSDT"),
            time_range=_time_range(),
            timeframe=Timeframe("1m"),
            ts_open=np.array([[1_700_000_000, 1_700_000_060]], dtype=np.int64),
            open=np.array([100.0, 101.0], dtype=np.float32),
            high=np.array([101.0, 102.0], dtype=np.float32),
            low=np.array([99.0, 100.0], dtype=np.float32),
            close=np.array([100.5, 101.5], dtype=np.float32),
            volume=np.array([10.0, 11.0], dtype=np.float32),
        )

    with pytest.raises(ValueError):
        CandleArrays(
            market_id=MarketId(1),
            symbol=Symbol("BTCUSDT"),
            time_range=_time_range(),
            timeframe=Timeframe("1m"),
            ts_open=np.array([1_700_000_000, 1_700_000_060], dtype=np.int64),
            open=np.array([100.0, 101.0], dtype=np.float64),
            high=np.array([101.0, 102.0], dtype=np.float32),
            low=np.array([99.0, 100.0], dtype=np.float32),
            close=np.array([100.5, 101.5], dtype=np.float32),
            volume=np.array([10.0, 11.0], dtype=np.float32),
        )
