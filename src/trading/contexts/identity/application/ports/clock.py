from __future__ import annotations

from datetime import datetime
from typing import Protocol


class IdentityClock(Protocol):
    """
    IdentityClock — порт источника текущего времени для identity use-cases.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/create_exchange_key.py
      - src/trading/contexts/identity/application/use_cases/delete_exchange_key.py
      - src/trading/contexts/identity/adapters/outbound/time/system_identity_clock.py
    """

    def now(self) -> datetime:
        """
        Return current UTC timestamp used in identity flow.

        Args:
            None.
        Returns:
            datetime: Timezone-aware UTC datetime.
        Assumptions:
            Implementations return monotonic wall-clock progression for request flow.
        Raises:
            None.
        Side Effects:
            None.
        """
        ...
