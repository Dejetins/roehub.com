from __future__ import annotations

from datetime import datetime, timezone

import pytest

from apps.cli.commands.funding_rate_catchup import _parse_optional_time_range


def test_parse_optional_time_range_requires_both_bounds() -> None:
    with pytest.raises(SystemExit):
        _parse_optional_time_range(start="2026-06-22T00:00:00Z", end=None)


def test_parse_optional_time_range_uses_half_open_utc_bounds() -> None:
    time_range = _parse_optional_time_range(
        start="2026-06-22T00:00:00Z",
        end="2026-06-22T08:00:00Z",
    )

    assert time_range is not None
    assert time_range.start.value == datetime(2026, 6, 22, 0, 0, tzinfo=timezone.utc)
    assert time_range.end.value == datetime(2026, 6, 22, 8, 0, tzinfo=timezone.utc)
