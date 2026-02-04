from __future__ import annotations

from typing import Protocol

from trading.shared_kernel.primitives import UtcTimestamp


class Clock(Protocol):
    """
    Clock — источник "текущего времени" для application-слоя в UTC.

    Contract:
    - now() -> UtcTimestamp
    """

    def now(self) -> UtcTimestamp:
        ...
