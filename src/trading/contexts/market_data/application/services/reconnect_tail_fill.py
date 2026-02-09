from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from trading.contexts.market_data.application.dto import RestFillTask
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    CanonicalCandleIndexReader,
)
from trading.contexts.market_data.application.services.minute_utils import floor_to_minute_utc
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class ReconnectTailFillPlanner:
    """
    Build REST fill tasks for reconnect/restart recovery.

    Parameters:
    - index_reader: canonical index reader used for last known minute lookup.
    - clock: UTC clock.
    - bootstrap_start: lower bound for bootstrap fills when canonical is empty.

    Assumptions/Invariants:
    - Tail range semantics are half-open: `[start, now_floor)`.
    - Current `now_floor` minute is considered not closed and therefore excluded.
    """

    index_reader: CanonicalCandleIndexReader
    clock: Clock
    bootstrap_start: UtcTimestamp = UtcTimestamp(datetime(2017, 1, 1, tzinfo=timezone.utc))

    def __post_init__(self) -> None:
        """
        Validate planner collaborators and bootstrap boundary.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Collaborators are non-null.

        Errors/Exceptions:
        - Raises `ValueError` on invalid constructor arguments.

        Side effects:
        - None.
        """
        if self.index_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("ReconnectTailFillPlanner requires index_reader")
        if self.clock is None:  # type: ignore[truthy-bool]
            raise ValueError("ReconnectTailFillPlanner requires clock")
        if self.bootstrap_start is None:  # type: ignore[truthy-bool]
            raise ValueError("ReconnectTailFillPlanner requires bootstrap_start")

    def plan(self, instruments: Iterable[InstrumentId]) -> list[RestFillTask]:
        """
        Plan reconnect-tail tasks for a collection of instruments.

        Parameters:
        - instruments: iterable of instrument ids bound to one WS connection.

        Returns:
        - List of fill tasks for bootstrap or missing tail minutes.

        Assumptions/Invariants:
        - Instrument ids are unique or duplicates are acceptable in caller context.

        Errors/Exceptions:
        - Propagates index reader errors.

        Side effects:
        - Reads canonical index storage.
        """
        tasks: list[RestFillTask] = []
        for instrument_id in instruments:
            task = self.plan_for_instrument(instrument_id)
            if task is not None:
                tasks.append(task)
        return tasks

    def plan_for_instrument(self, instrument_id: InstrumentId) -> RestFillTask | None:
        """
        Plan reconnect-tail task for one instrument.

        Parameters:
        - instrument_id: target instrument.

        Returns:
        - `RestFillTask` when fill is needed, otherwise `None`.

        Assumptions/Invariants:
        - `now_floor` is minute-floor of current UTC clock.

        Errors/Exceptions:
        - Propagates index reader errors.

        Side effects:
        - Reads canonical index storage.
        """
        now_floor = UtcTimestamp(floor_to_minute_utc(self.clock.now().value))
        last = self.index_reader.max_ts_open_lt(
            instrument_id=instrument_id,
            before=now_floor,
        )

        if last is None:
            if self.bootstrap_start.value >= now_floor.value:
                return None
            return RestFillTask(
                instrument_id=instrument_id,
                time_range=TimeRange(start=self.bootstrap_start, end=now_floor),
                reason="bootstrap",
            )

        threshold = now_floor.value - timedelta(minutes=1)
        if last.value >= threshold:
            return None

        start = UtcTimestamp(last.value + timedelta(minutes=1))
        if start.value >= now_floor.value:
            return None
        return RestFillTask(
            instrument_id=instrument_id,
            time_range=TimeRange(start=start, end=now_floor),
            reason="reconnect_tail",
        )

