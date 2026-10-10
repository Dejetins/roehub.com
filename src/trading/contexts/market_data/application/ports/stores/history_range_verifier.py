"""Confirm durable candles and record authoritative source omissions separately."""

from datetime import datetime
from typing import Callable, Protocol

from trading.shared_kernel.primitives import InstrumentId, TimeRange


class HistoryRangeVerifier(Protocol):
    def confirm(
        self, *, instrument_id: InstrumentId, time_range: TimeRange,
        observed_minutes: set[int], observed_at: datetime, accept_absence: bool,
        check_lease: Callable[[], None],
    ) -> None: ...
