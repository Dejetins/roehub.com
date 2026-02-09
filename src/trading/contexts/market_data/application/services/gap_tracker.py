from __future__ import annotations

from datetime import timedelta
from typing import Callable

from trading.contexts.market_data.application.dto import CandleWithMeta, RestFillTask
from trading.contexts.market_data.application.services.minute_utils import floor_to_minute_utc
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


class WsMinuteGapTracker:
    """
    Track minute continuity per instrument and build REST fill tasks on gaps.

    Parameters:
    - on_duplicate: optional callback invoked when duplicate minute is observed.
    - on_out_of_order: optional callback invoked when older minute arrives.

    Assumptions/Invariants:
    - Input candles are normalized to 1m closed candles.
    - Time comparisons are done on minute buckets in UTC.
    """

    def __init__(
        self,
        *,
        on_duplicate: Callable[[], None] | None = None,
        on_out_of_order: Callable[[], None] | None = None,
    ) -> None:
        """
        Create tracker with optional metrics callbacks.

        Parameters:
        - on_duplicate: callback for duplicate minute events.
        - on_out_of_order: callback for out-of-order minute events.

        Returns:
        - None.

        Assumptions/Invariants:
        - Callbacks are side-effect functions (metrics/logging) and may be absent.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        self._last_seen_by_instrument: dict[InstrumentId, UtcTimestamp] = {}
        self._on_duplicate = on_duplicate
        self._on_out_of_order = on_out_of_order

    def observe(self, row: CandleWithMeta) -> RestFillTask | None:
        """
        Observe one closed candle and detect minute gaps.

        Parameters:
        - row: closed 1m candle with metadata from WS path.

        Returns:
        - `RestFillTask` for gap `[expected, current)` or `None` when no gap exists.

        Assumptions/Invariants:
        - `row.candle.ts_open` can be normalized to minute bucket.

        Errors/Exceptions:
        - None.

        Side effects:
        - Updates in-memory last-seen minute cache.
        - Triggers duplicate/out-of-order callbacks when applicable.
        """
        instrument_id = row.candle.instrument_id
        minute_dt = floor_to_minute_utc(row.candle.ts_open.value)
        minute_ts = UtcTimestamp(minute_dt)

        last_seen = self._last_seen_by_instrument.get(instrument_id)
        if last_seen is None:
            self._last_seen_by_instrument[instrument_id] = minute_ts
            return None

        expected_dt = last_seen.value + timedelta(minutes=1)
        expected_ts = UtcTimestamp(expected_dt)

        if minute_ts.value == expected_ts.value:
            self._last_seen_by_instrument[instrument_id] = minute_ts
            return None

        if minute_ts.value <= last_seen.value:
            if minute_ts.value == last_seen.value:
                _invoke_callback(self._on_duplicate)
            else:
                _invoke_callback(self._on_out_of_order)
            return None

        self._last_seen_by_instrument[instrument_id] = minute_ts
        return RestFillTask(
            instrument_id=instrument_id,
            time_range=TimeRange(start=expected_ts, end=minute_ts),
            reason="gap",
        )


def _invoke_callback(callback: Callable[[], None] | None) -> None:
    """
    Safely call optional no-arg callback.

    Parameters:
    - callback: callback function or `None`.

    Returns:
    - None.

    Assumptions/Invariants:
    - Callback side effects are controlled by caller.

    Errors/Exceptions:
    - None.

    Side effects:
    - Executes callback when provided.
    """
    if callback is None:
        return
    callback()

