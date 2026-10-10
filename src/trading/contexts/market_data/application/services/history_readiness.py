"""Full-history completeness from the persisted exchange origin through closed minutes."""

from datetime import datetime
from math import floor
from typing import Any, Mapping, Protocol

from trading.contexts.market_data.application.ports.stores.history_coverage_reader import (
    HistoryCoverageReader,
)
from trading.shared_kernel.primitives import InstrumentId


class HistoryOrigins(Protocol):
    def read_or_request(
        self, *, market_id: int, symbol: str, now: datetime
    ) -> Mapping[str, Any] | None: ...
    def ready_starts(self, *, market_ids: list[int]) -> Mapping[tuple[int, str], datetime]: ...


def history_snapshot(
    *, first: datetime | None, end: datetime, actual: int, source_absent: int = 0
) -> dict[str, Any]:
    calendar = max(0, int((end - first).total_seconds() // 60)) if first else 0
    source_absent = min(max(0, source_absent), max(0, calendar - actual))
    expected = calendar - source_absent
    actual = min(actual, expected)
    return dict(
        start_at=first,
        end_at=end,
        expected_candles=expected,
        actual_candles=actual,
        source_absent_minutes=source_absent,
        missing_candles=expected - actual if first else None,
        coverage_percent=(
            100
            if actual == expected and calendar
            else min(99.99, floor(actual / expected * 10000) / 100)
            if expected
            else None
        ),
        state=(
            "unknown"
            if not calendar
            else "complete"
            if actual == expected
            else "partial"
            if actual
            else "empty"
        ),
    )


class HistoryReadiness:
    def __init__(self, *, origins: HistoryOrigins, coverage: HistoryCoverageReader) -> None:
        self.origins, self.coverage = origins, coverage

    def read(self, *, instrument_id: InstrumentId, now: datetime) -> dict[str, Any]:
        key = (instrument_id.market_id.value, str(instrument_id.symbol))
        first = self.origins.ready_starts(market_ids=[key[0]]).get(key)
        end = now.replace(second=0, microsecond=0)
        counts = self.coverage.counts(starts={key: first} if first else {}, end_at=end)
        absent = self.coverage.source_absent_counts(
            starts={key: first} if first else {}, end_at=end
        )
        return history_snapshot(first=first, end=end, actual=counts.get(key, 0),
                                source_absent=absent.get(key, 0))
