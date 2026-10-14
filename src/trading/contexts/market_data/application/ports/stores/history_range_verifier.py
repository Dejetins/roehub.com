"""Confirm durable candles and record authoritative source omissions separately."""

from datetime import datetime
from typing import Any, Callable, Protocol

from trading.shared_kernel.primitives import InstrumentId, TimeRange


class HistoryRangeVerifier(Protocol):
    def acquire(self, *, instrument_id: InstrumentId, ttl_seconds: int = 120) -> Any: ...
    def renew(self, *, instrument_id: InstrumentId, fence: Any, ttl_seconds: int = 120) -> None: ...
    def release(self, *, instrument_id: InstrumentId, fence: Any) -> Any: ...
    def origin(self, *, instrument_id: InstrumentId) -> datetime | None: ...

    def confirm(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
        observed_minutes: set[int],
        observed_at: datetime,
        accept_absence: bool,
        check_lease: Callable[[], None],
        fence: Any,
        origin: datetime | None,
    ) -> None: ...
