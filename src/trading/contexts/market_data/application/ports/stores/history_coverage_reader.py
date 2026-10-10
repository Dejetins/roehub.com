"""Exact full-history coverage and bounded gap discovery, without candle payloads."""

from datetime import datetime
from typing import Mapping, Protocol

from trading.shared_kernel.primitives import InstrumentId, TimeRange


class HistoryCoverageReader(Protocol):
    def counts(
        self, *, starts: Mapping[tuple[int, str], datetime], end_at: datetime
    ) -> Mapping[tuple[int, str], int]: ...

    def source_absent_counts(
        self, *, starts: Mapping[tuple[int, str], datetime], end_at: datetime
    ) -> Mapping[tuple[int, str], int]: ...

    def first_gap(
        self, *, instrument_id: InstrumentId, time_range: TimeRange
    ) -> TimeRange | None: ...
