from __future__ import annotations

from datetime import datetime, timezone

from trading.contexts.market_data.application.use_cases.time_slicing import (
    slice_time_range_by_utc_days,
)
from trading.shared_kernel.primitives.time_range import TimeRange
from trading.shared_kernel.primitives.utc_timestamp import UtcTimestamp


def test_slice_time_range_10_days_into_two_chunks() -> None:
    tr = TimeRange(
        start=UtcTimestamp(datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)),
    )
    chunks = slice_time_range_by_utc_days(tr, max_days=7)
    assert len(chunks) == 2

    assert str(chunks[0].start) == "2026-02-01T12:00:00.000Z"
    assert str(chunks[0].end) == "2026-02-08T00:00:00.000Z"  # day boundary

    assert str(chunks[1].start) == "2026-02-08T00:00:00.000Z"
    assert str(chunks[1].end) == "2026-02-11T12:00:00.000Z"
