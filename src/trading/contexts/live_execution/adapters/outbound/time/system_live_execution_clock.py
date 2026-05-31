from __future__ import annotations

from datetime import UTC, datetime


class SystemLiveExecutionClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

