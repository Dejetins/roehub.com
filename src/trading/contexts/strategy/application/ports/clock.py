from __future__ import annotations

from datetime import datetime
from typing import Protocol


class StrategyClock(Protocol):
    """
    StrategyClock — application port providing timezone-aware UTC timestamps for Strategy use-cases.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/use_cases
      - apps/api/wiring/modules/strategy.py
      - src/trading/platform/time/system_clock.py
    """

    def now(self) -> datetime:
        """
        Return current timezone-aware UTC timestamp.

        Args:
            None.
        Returns:
            datetime: Current UTC timestamp.
        Assumptions:
            Returned datetime is timezone-aware with zero UTC offset.
        Raises:
            ValueError: If implementation cannot provide valid UTC datetime.
        Side Effects:
            None.
        """
        ...
