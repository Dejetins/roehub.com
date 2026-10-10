"""Calendar occurrences; UTC elapsed durations never define a calendar-day policy."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from trading.contexts.backtest_artifacts.application.services.management import Schedule


def occurrence(schedule: Schedule, day: date) -> datetime:
    zone = ZoneInfo(schedule.timezone)
    wall = datetime.combine(day, schedule.local_time)
    # First fold; gaps advance to the first valid civil minute (not by gap duration).
    for _ in range(1441):
        candidate = wall.replace(tzinfo=zone, fold=0).astimezone(UTC)
        if candidate.astimezone(zone).replace(tzinfo=None) == wall:
            return candidate
        wall += timedelta(minutes=1)
    raise ValueError("invalid_calendar_day")


def due_window(schedule: Schedule, now: datetime) -> tuple[datetime | None, datetime]:
    """Return latest due occurrence and strictly future occurrence, without catch-up jobs."""
    if now.utcoffset() != timedelta(0):
        raise ValueError("utc_required")
    day = now.astimezone(ZoneInfo(schedule.timezone)).date()
    steps = max(0, (day - schedule.anchor).days // schedule.interval_days)
    candidate = occurrence(
        schedule, schedule.anchor + timedelta(days=steps * schedule.interval_days)
    )
    if candidate > now:
        if steps == 0:
            return None, candidate
        steps -= 1
        candidate = occurrence(
            schedule, schedule.anchor + timedelta(days=steps * schedule.interval_days)
        )
    following = occurrence(
        schedule, schedule.anchor + timedelta(days=(steps + 1) * schedule.interval_days)
    )
    return candidate, following
