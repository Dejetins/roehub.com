from __future__ import annotations

from datetime import datetime
from typing import Protocol


class LiveExecutionClock(Protocol):
    def now(self) -> datetime: ...

