from unittest.mock import Mock

import numpy as np
import pytest

from trading.contexts.backtest.application.services.v2.result_candles import project_result_candles
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactPriceArraysV2,
)


def test_ohlc_groups_preserve_extrema_and_exclude_end_boundary():
    times = np.arange(202) * 60_000 + 1767225600000
    rows = np.array([[10 + i, 30 + i, 1 + i, 20 + i, 1] for i in range(202)])
    prices = Mock(spec=ArtifactPriceArraysV2, timeframe="1m", open_time=times, ohlcv=rows)
    result = project_result_candles(
        prices=prices,
        start="2026-01-01T00:01:00Z",
        end="2026-01-01T03:21:00Z",
        max_bars=100,
    )
    assert result["source_bars"] == 200
    assert result["group_size"] == 2
    assert len(result["candles"]) == 100
    assert result["candles"][0] == {
        "time": "2026-01-01T00:01:00Z",
        "open": 11.0,
        "high": 32.0,
        "low": 2.0,
        "close": 22.0,
    }
    assert result["candles"][-1]["close"] == 220.0
    with pytest.raises(ValueError):
        project_result_candles(prices=prices, start="", end="", max_bars=60001)


def test_timeframe_aggregation_uses_utc_boundaries_and_real_ohlc():
    prices = Mock(
        spec=ArtifactPriceArraysV2,
        timeframe="1m",
        open_time=np.arange(30) * 60000 + 1767225600000,
        ohlcv=np.array([[i + 10, i + 30, i + 1, i + 20] for i in range(30)]),
    )
    result = project_result_candles(
        prices=prices,
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:30:00Z",
        max_bars=100,
        timeframe="15m",
    )
    assert result["timeframe"] == "15m"
    assert result["source_bars"] == 2
    assert result["candles"][0] == {
        "time": "2026-01-01T00:00:00Z",
        "open": 10.0,
        "high": 44.0,
        "low": 1.0,
        "close": 34.0,
    }
    assert result["candles"][1]["time"] == "2026-01-01T00:15:00Z"
