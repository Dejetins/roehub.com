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
from .paid_level import PaidLevel
from .symbol import Symbol
from .time_range import TimeRange
from .timeframe import Timeframe
from .user_id import UserId
from .utc_timestamp import UtcTimestamp

__all__ = [
    "Candle",
    "CandleMeta",
    "InstrumentId",
    "MarketId",
    "PaidLevel",
    "Symbol",
    "TimeRange",
    "Timeframe",
    "UtcTimestamp",
    "UserId",
]
