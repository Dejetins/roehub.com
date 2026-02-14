from __future__ import annotations

from datetime import datetime, timezone

from trading.contexts.identity.application.ports.clock import IdentityClock


class SystemIdentityClock(IdentityClock):
    """
    SystemIdentityClock — platform реализация `IdentityClock` на системном UTC времени.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/clock.py
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - apps/api/wiring/modules/identity.py
    """

    def now(self) -> datetime:
        """
        Return current timezone-aware UTC datetime.

        Args:
            None.
        Returns:
            datetime: Current UTC datetime.
        Assumptions:
            System clock is reasonably synchronized.
        Raises:
            None.
        Side Effects:
            Reads system wall clock.
        """
        return datetime.now(timezone.utc)
