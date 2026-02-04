"""
Shared Kernel primitives.

This package re-exports the minimal set of domain primitives so that other
modules can import them from one place:

    from trading.shared_kernel.primitives import Candle, CandleMeta, InstrumentId
"""

from .candle import Candle
from .candle_meta import CandleMeta
from .instrument_id import InstrumentId
from .market_id import MarketId
from .symbol import Symbol
from .time_range import TimeRange
from .timeframe import Timeframe
from .utc_timestamp import UtcTimestamp

__all__ = [
    "Candle",
    "CandleMeta",
    "InstrumentId",
    "MarketId",
    "Symbol",
    "TimeRange",
    "Timeframe",
    "UtcTimestamp",
]
