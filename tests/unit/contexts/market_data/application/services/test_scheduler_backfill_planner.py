from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading.contexts.market_data.application.services.scheduler_backfill_planner import (
    SchedulerBackfillPlanner,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp


def _instrument() -> InstrumentId:
    """
    Build deterministic instrument id for planner tests.

    Parameters:
    - None.

    Returns:
    - Instrument id `(1, BTCUSDT)`.
    """
    return InstrumentId(MarketId(1), Symbol("BTCUSDT"))


def test_planner_bootstrap_when_canonical_is_empty() -> None:
    """Ensure empty canonical bounds produce one bootstrap task."""
    planner = SchedulerBackfillPlanner(tail_lookback_minutes=180)
    earliest = UtcTimestamp(datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc))
    now_floor = UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))

    tasks = planner.plan_for_instrument(
        instrument_id=_instrument(),
        earliest_market_ts=earliest,
        bounds_1m=(None, None),
        now_floor=now_floor,
    )

    assert len(tasks) == 1
    assert tasks[0].reason == "scheduler_bootstrap"
    assert str(tasks[0].time_range.start) == str(earliest)
    assert str(tasks[0].time_range.end) == str(now_floor)


def test_planner_creates_historical_range_from_earliest_to_canonical_min() -> None:
    """Ensure historical gap before canonical min is planned as half-open range."""
    planner = SchedulerBackfillPlanner(tail_lookback_minutes=180)
    earliest = UtcTimestamp(datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc))
    canonical_min = UtcTimestamp(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc))
    canonical_max = UtcTimestamp(datetime(2026, 2, 9, 13, 0, tzinfo=timezone.utc))
    now_floor = UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))

    tasks = planner.plan_for_instrument(
        instrument_id=_instrument(),
        earliest_market_ts=earliest,
        bounds_1m=(canonical_min, canonical_max),
        now_floor=now_floor,
    )

    reasons = [task.reason for task in tasks]
    assert "historical_backfill" in reasons
    historical = next(task for task in tasks if task.reason == "historical_backfill")
    assert str(historical.time_range.start) == str(earliest)
    assert str(historical.time_range.end) == str(canonical_min)


def test_planner_tail_fill_starts_from_canonical_max_plus_one_minute() -> None:
    """Ensure tail task starts from `canonical_max + 1m` when it is beyond lookback start."""
    planner = SchedulerBackfillPlanner(tail_lookback_minutes=180)
    earliest = UtcTimestamp(datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc))
    canonical_min = UtcTimestamp(datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc))
    canonical_max = UtcTimestamp(datetime(2026, 2, 9, 13, 57, tzinfo=timezone.utc))
    now_floor = UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))

    tasks = planner.plan_for_instrument(
        instrument_id=_instrument(),
        earliest_market_ts=earliest,
        bounds_1m=(canonical_min, canonical_max),
        now_floor=now_floor,
    )

    assert len(tasks) == 1
    assert tasks[0].reason == "scheduler_tail"
    assert str(tasks[0].time_range.start) == str(
        UtcTimestamp(datetime(2026, 2, 9, 13, 58, tzinfo=timezone.utc))
    )


def test_planner_tail_only_canonical_still_plans_historical_backfill() -> None:
    """Ensure tail-only canonical state does not suppress historical backfill planning."""
    planner = SchedulerBackfillPlanner(tail_lookback_minutes=180)
    now_floor = UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))
    earliest = UtcTimestamp(datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc))
    canonical_min = UtcTimestamp(datetime(2026, 2, 9, 13, 50, tzinfo=timezone.utc))
    canonical_max = UtcTimestamp(datetime(2026, 2, 9, 13, 59, tzinfo=timezone.utc))

    tasks = planner.plan_for_instrument(
        instrument_id=_instrument(),
        earliest_market_ts=earliest,
        bounds_1m=(canonical_min, canonical_max),
        now_floor=now_floor,
    )

    assert any(task.reason == "historical_backfill" for task in tasks)
    historical = next(task for task in tasks if task.reason == "historical_backfill")
    assert str(historical.time_range.start) == str(earliest)
    assert str(historical.time_range.end) == str(canonical_min)
    # no tail task in fully up-to-date tail state
    assert not any(task.reason == "scheduler_tail" for task in tasks)


def test_planner_does_not_plan_historical_when_canonical_starts_at_earliest() -> None:
    """Ensure no historical task is created when canonical min is already at earliest boundary."""
    planner = SchedulerBackfillPlanner(tail_lookback_minutes=180)
    now_floor = UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))
    earliest = UtcTimestamp(datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc))
    canonical_min = UtcTimestamp(datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc))
    canonical_max = UtcTimestamp(datetime(2026, 2, 9, 13, 50, tzinfo=timezone.utc))

    tasks = planner.plan_for_instrument(
        instrument_id=_instrument(),
        earliest_market_ts=earliest,
        bounds_1m=(canonical_min, canonical_max),
        now_floor=now_floor,
    )

    assert not any(task.reason == "historical_backfill" for task in tasks)
    assert any(task.reason == "scheduler_tail" for task in tasks)


def test_planner_ranges_do_not_overlap() -> None:
    """Ensure historical and tail ranges never overlap in one plan output."""
    planner = SchedulerBackfillPlanner(tail_lookback_minutes=180)
    earliest = UtcTimestamp(datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc))
    canonical_min = UtcTimestamp(datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc))
    canonical_max = UtcTimestamp(datetime(2026, 2, 9, 13, 50, tzinfo=timezone.utc))
    now_floor = UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))

    tasks = planner.plan_for_instrument(
        instrument_id=_instrument(),
        earliest_market_ts=earliest,
        bounds_1m=(canonical_min, canonical_max),
        now_floor=now_floor,
    )

    historical = next(task for task in tasks if task.reason == "historical_backfill")
    tail = next(task for task in tasks if task.reason == "scheduler_tail")

    assert historical.time_range.end.value <= canonical_min.value
    assert tail.time_range.start.value >= canonical_max.value + timedelta(minutes=1)
    assert historical.time_range.end.value <= tail.time_range.start.value
