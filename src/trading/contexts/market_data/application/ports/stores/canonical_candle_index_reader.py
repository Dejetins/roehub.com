from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence

from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class DailyTsOpenCount:
    day: date
    count: int


class CanonicalCandleIndexReader(Protocol):
    """
    Индексное/агрегатное чтение canonical_candles_1m (без потока свечей).

    Нужно для:
    - max(ts_open) перед end (для catch-up start)
    - min/max bounds (для gap scan)
    - day-level countDistinct(ts_open) (быстрое выявление дней с пропусками)
    - список ts_open в конкретном окне (точное восстановление missing ranges)
    """

    def bounds(self, instrument_id: InstrumentId) -> tuple[UtcTimestamp, UtcTimestamp] | None:
        ...

    def max_ts_open_lt(self, *, instrument_id: InstrumentId, before: UtcTimestamp) -> UtcTimestamp | None:  # noqa: E501
        ...

    def daily_counts(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> Sequence[DailyTsOpenCount]:  # noqa: E501
        ...

    def distinct_ts_opens(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> Sequence[UtcTimestamp]:  # noqa: E501
        ...
