from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from trading.contexts.market_data.application.dto import RestFillTask
from trading.contexts.market_data.application.services.minute_utils import floor_to_minute_utc
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class SchedulerBackfillPlanner:
    """
    Plan scheduler REST fill tasks for bootstrap, historical backfill, and tail insurance.

    Parameters:
    - tail_lookback_minutes: insurance lookback window for tail task planning.

    Assumptions/Invariants:
    - All ranges follow half-open semantics `[start, end)`.
    - Inputs are UTC timestamps normalized to minute boundaries or normalizable to them.
    """

    tail_lookback_minutes: int

    def __post_init__(self) -> None:
        """
        Validate planner construction arguments.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - `tail_lookback_minutes` must be positive.

        Errors/Exceptions:
        - Raises `ValueError` when lookback is not positive.

        Side effects:
        - None.
        """
        if self.tail_lookback_minutes <= 0:
            raise ValueError("tail_lookback_minutes must be > 0")

    def plan_for_instrument(
        self,
        *,
        instrument_id: InstrumentId,
        earliest_market_ts: UtcTimestamp,
        bounds_1m: tuple[UtcTimestamp | None, UtcTimestamp | None],
        now_floor: UtcTimestamp,
    ) -> list[RestFillTask]:
        """
        Build non-overlapping maintenance tasks for one instrument.

        Parameters:
        - instrument_id: instrument identity.
        - earliest_market_ts: configured earliest available exchange minute for this market.
        - bounds_1m: canonical bounds `(min_ts_open, max_ts_open)` before `now_floor`.
        - now_floor: current minute floor in UTC, excluded from fills.

        Returns:
        - Ordered list of planned tasks for this instrument.

        Assumptions/Invariants:
        - `bounds_1m` values are minute-level values or `None`.
        - No overlap is produced between historical and tail tasks.

        Errors/Exceptions:
        - Raises `ValueError` indirectly when produced ranges violate `TimeRange` invariants.

        Side effects:
        - None.
        """
        earliest = UtcTimestamp(floor_to_minute_utc(earliest_market_ts.value))
        now_minute = UtcTimestamp(floor_to_minute_utc(now_floor.value))
        if earliest.value >= now_minute.value:
            return []

        first_ts, last_ts = bounds_1m
        if first_ts is None or last_ts is None:
            return [
                RestFillTask(
                    instrument_id=instrument_id,
                    time_range=TimeRange(start=earliest, end=now_minute),
                    reason="scheduler_bootstrap",
                )
            ]

        tasks: list[RestFillTask] = []

        historical_threshold = earliest.value + timedelta(minutes=1)
        if first_ts.value > historical_threshold and earliest.value < first_ts.value:
            tasks.append(
                RestFillTask(
                    instrument_id=instrument_id,
                    time_range=TimeRange(start=earliest, end=first_ts),
                    reason="historical_backfill",
                )
            )

        lookback_start = UtcTimestamp(
            now_minute.value - timedelta(minutes=self.tail_lookback_minutes)
        )
        tail_start_dt = max(last_ts.value + timedelta(minutes=1), lookback_start.value)
        if tail_start_dt < now_minute.value:
            tasks.append(
                RestFillTask(
                    instrument_id=instrument_id,
                    time_range=TimeRange(start=UtcTimestamp(tail_start_dt), end=now_minute),
                    reason="scheduler_tail",
                )
            )

        return tasks
