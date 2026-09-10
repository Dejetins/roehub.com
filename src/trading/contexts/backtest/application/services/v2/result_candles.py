"""Bounded OHLC projection of the job-pinned artifact, preserving price extrema."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactPriceArraysV2,
)


def project_result_candles(
    *, prices: ArtifactPriceArraysV2, start: str, end: str, max_bars: int,
    timeframe: str | None = None
) -> dict[str, Any]:
    if not 100 <= max_bars <= 60000:
        raise ValueError("max_bars must be between 100 and 60000")
    start_ms = int(datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp() * 1000)
    first = int(np.searchsorted(prices.open_time, start_ms, side="left"))
    stop = int(np.searchsorted(prices.open_time, end_ms, side="left"))
    timeframe = timeframe or prices.timeframe
    minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    if timeframe not in minutes:
        raise ValueError("Unsupported chart timeframe")
    times = prices.open_time[first:stop]
    values = prices.ohlcv[first:stop]
    if timeframe != prices.timeframe:
        if prices.timeframe != "1m":
            raise ValueError("Chart aggregation requires canonical one-minute candles")
        buckets = times // (minutes[timeframe] * 60000)
        starts = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1] if len(times) else []
        stops = list(starts[1:]) + [len(times)] if len(times) else []
        values = np.array([
            [values[a, 0], np.max(values[a:b, 1]), np.min(values[a:b, 2]), values[b-1, 3]]
            for a, b in zip(starts, stops)
        ])
        times = times[starts] if len(times) else times
    count = len(times)
    group_size = max(1, (count + max_bars - 1) // max_bars)
    items = []
    for index in range(0, count, group_size):
        block = values[index:min(index + group_size, count)]
        items.append({
            "time": datetime.fromtimestamp(
                int(times[index]) / 1000, UTC
            ).isoformat().replace("+00:00", "Z"),
            "open": float(block[0, 0]), "high": float(np.max(block[:, 1])),
            "low": float(np.min(block[:, 2])), "close": float(block[-1, 3]),
        })
    return {"candles": items, "source_bars": count, "group_size": group_size,
            "timeframe": timeframe}
