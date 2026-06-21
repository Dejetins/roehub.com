from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Mapping

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
    - bootstrap_start_by_market: market-specific bootstrap lower bounds from runtime config.
    - tail_lookback_minutes: bounded recent lookback for reconnect recovery.

    Assumptions/Invariants:
    - Tail range semantics are half-open: `[start, now_floor)`.
    - Current `now_floor` minute is considered not closed and therefore excluded.
    - Reconnect recovery is a recent-tail concern; full historical scans belong to scheduler
      maintenance, not websocket reconnect callbacks.
    """

    index_reader: CanonicalCandleIndexReader
    clock: Clock
    bootstrap_start_by_market: Mapping[int, UtcTimestamp]
    tail_lookback_minutes: int = 720

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
        if self.bootstrap_start_by_market is None:  # type: ignore[truthy-bool]
            raise ValueError("ReconnectTailFillPlanner requires bootstrap_start_by_market")
        if self.tail_lookback_minutes <= 0:
            raise ValueError("tail_lookback_minutes must be > 0")

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
        lookback_start = UtcTimestamp(
            now_floor.value - timedelta(minutes=self.tail_lookback_minutes)
        )
        last = self.index_reader.max_ts_open_lt(
            instrument_id=instrument_id,
            before=now_floor,
            after=lookback_start,
        )

        if last is None:
            bootstrap_start = self._bootstrap_start(instrument_id)
            start = max(bootstrap_start.value, lookback_start.value)
            if start >= now_floor.value:
                return None
            return RestFillTask(
                instrument_id=instrument_id,
                time_range=TimeRange(start=UtcTimestamp(start), end=now_floor),
                reason="reconnect_tail",
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

    def _bootstrap_start(self, instrument_id: InstrumentId) -> UtcTimestamp:
        """
        Resolve bootstrap lower bound for one market from runtime mapping.

        Parameters:
        - instrument_id: instrument used to extract market id key.

        Returns:
        - Configured earliest timestamp for that market.

        Assumptions/Invariants:
        - Mapping contains entries for all configured markets.

        Errors/Exceptions:
        - Raises `KeyError` when market id is missing in mapping.

        Side effects:
        - None.
        """
        market_id = int(instrument_id.market_id.value)
        try:
            return self.bootstrap_start_by_market[market_id]
        except KeyError as exc:
            raise KeyError(
                f"missing bootstrap_start for market_id={market_id} in reconnect planner"
            ) from exc
