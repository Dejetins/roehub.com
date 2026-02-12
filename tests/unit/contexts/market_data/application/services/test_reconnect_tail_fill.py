from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    DailyTsOpenCount,
)
from trading.contexts.market_data.application.services.reconnect_tail_fill import (
    ReconnectTailFillPlanner,
)
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


class _FixedClock:
    """Clock fake returning one constant UTC timestamp."""

    def __init__(self, now_value: UtcTimestamp) -> None:
        """Store fixed timestamp for later `now()` calls."""
        self._now_value = now_value

    def now(self) -> UtcTimestamp:
        """Return preconfigured fixed timestamp."""
        return self._now_value


class _FakeIndex:
    """Canonical index fake exposing configurable `max_ts_open_lt` output."""

    def __init__(self, last_value: UtcTimestamp | None) -> None:
        """Store latest timestamp result used by planner test."""
        self._last_value = last_value

    def max_ts_open_lt(self, *, instrument_id, before):  # noqa: ANN001
        """Return configured latest timestamp."""
        _ = instrument_id
        _ = before
        return self._last_value

    def bounds(self, instrument_id: InstrumentId) -> tuple[UtcTimestamp, UtcTimestamp] | None:
        """
        Return canonical bounds placeholder required by protocol.

        Parameters:
        - instrument_id: requested instrument id.

        Returns:
        - None because planner tests only rely on `max_ts_open_lt`.
        """
        _ = instrument_id
        return None

    def bounds_1m(
        self,
        *,
        instrument_id: InstrumentId,
        before: UtcTimestamp,
    ) -> tuple[UtcTimestamp | None, UtcTimestamp | None]:
        """
        Return minute bounds placeholder required by protocol.

        Parameters:
        - instrument_id: requested instrument id.
        - before: exclusive upper bound.

        Returns:
        - Tuple `(None, None)` because this fake does not model bounds.
        """
        _ = instrument_id
        _ = before
        return (None, None)

    def daily_counts(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Sequence[DailyTsOpenCount]:
        """
        Return empty day counts placeholder required by protocol.

        Parameters:
        - instrument_id: requested instrument id.
        - time_range: requested range.

        Returns:
        - Empty sequence for planner-only tests.
        """
        _ = instrument_id
        _ = time_range
        return ()

    def distinct_ts_opens(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Sequence[UtcTimestamp]:
        """
        Return empty distinct timestamp list placeholder required by protocol.

        Parameters:
        - instrument_id: requested instrument id.
        - time_range: requested range.

        Returns:
        - Empty sequence for planner-only tests.
        """
        _ = instrument_id
        _ = time_range
        return ()


def test_reconnect_tail_planner_bootstrap_when_canonical_is_empty() -> None:
    """Ensure planner creates bootstrap task when canonical index has no rows."""
    planner = ReconnectTailFillPlanner(
        index_reader=_FakeIndex(last_value=None),
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 5, 12, 34, 45, tzinfo=timezone.utc))),
        bootstrap_start_by_market={
            1: UtcTimestamp(datetime(2017, 1, 1, tzinfo=timezone.utc)),
        },
    )
    task = planner.plan_for_instrument(InstrumentId(MarketId(1), Symbol("BTCUSDT")))
    assert task is not None
    assert task.reason == "bootstrap"
    assert str(task.time_range.end) == str(
        UtcTimestamp(datetime(2026, 2, 5, 12, 34, tzinfo=timezone.utc))
    )


def test_reconnect_tail_planner_enqueues_tail_fill_for_old_last_minute() -> None:
    """Ensure planner creates reconnect-tail range when canonical lag is above one minute."""
    planner = ReconnectTailFillPlanner(
        index_reader=_FakeIndex(
            last_value=UtcTimestamp(datetime(2026, 2, 5, 12, 30, tzinfo=timezone.utc))
        ),
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 5, 12, 34, 30, tzinfo=timezone.utc))),
        bootstrap_start_by_market={
            1: UtcTimestamp(datetime(2017, 1, 1, tzinfo=timezone.utc)),
        },
    )
    task = planner.plan_for_instrument(InstrumentId(MarketId(1), Symbol("BTCUSDT")))
    assert task is not None
    assert task.reason == "reconnect_tail"
    assert str(task.time_range.start) == str(
        UtcTimestamp(datetime(2026, 2, 5, 12, 31, tzinfo=timezone.utc))
    )
    assert str(task.time_range.end) == str(
        UtcTimestamp(datetime(2026, 2, 5, 12, 34, tzinfo=timezone.utc))
    )
