from __future__ import annotations

from enum import Enum


class InputSeries(str, Enum):
    """
    Logical input series available to indicator definitions.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .indicator_def, ...application.ports.feeds.candle_feed
    """

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    HL2 = "hl2"
    HLC3 = "hlc3"
    OHLC4 = "ohlc4"
