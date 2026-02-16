from __future__ import annotations

from datetime import datetime, timezone

from trading.contexts.strategy.application.ports.clock import StrategyClock


class SystemStrategyClock(StrategyClock):
    """
    SystemStrategyClock — platform StrategyClock adapter returning current UTC datetime.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/clock.py
      - apps/api/wiring/modules/strategy.py
      - src/trading/platform/time/system_clock.py
    """

    def now(self) -> datetime:
        """
        Return current timezone-aware UTC datetime.

        Args:
            None.
        Returns:
            datetime: Current UTC datetime.
        Assumptions:
            `datetime.now(timezone.utc)` returns offset-aware UTC value.
        Raises:
            None.
        Side Effects:
            None.
        """
        return datetime.now(timezone.utc)
