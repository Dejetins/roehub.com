from __future__ import annotations

from datetime import datetime, timezone

from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.shared_kernel.primitives import UtcTimestamp


class SystemClock(Clock):
    """
    SystemClock — platform реализация Clock: "сейчас" из системного времени.

    Возвращает UtcTimestamp(datetime.now(timezone.utc)).
    """

    def now(self) -> UtcTimestamp:
        return UtcTimestamp(datetime.now(timezone.utc))
