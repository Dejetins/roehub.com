from __future__ import annotations

from datetime import datetime
from typing import Protocol

from trading.contexts.identity.domain.entities import User
from trading.contexts.identity.domain.value_objects import TelegramUserId
from trading.shared_kernel.primitives import UserId


class UserRepository(Protocol):
    """
    UserRepository — порт хранения пользователей identity v1.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/in_memory/user_repository.py
    """

    def find_by_user_id(self, *, user_id: UserId) -> User | None:
        """
        Find user by stable cross-context user identifier.

        Args:
            user_id: Identity user identifier.
        Returns:
            User | None: Active or deleted user snapshot, or `None` when missing.
        Assumptions:
            Lookup is deterministic and unique by `user_id`.
        Raises:
            ValueError: If repository implementation rejects invalid inputs.
        Side Effects:
            None.
        """
        ...

    def upsert_telegram_login(
        self,
        *,
        telegram_user_id: TelegramUserId,
        login_at: datetime,
    ) -> User:
        """
        Create-or-update user by Telegram identity during successful login.

        Args:
            telegram_user_id: Validated Telegram user identifier.
            login_at: Current UTC login timestamp.
        Returns:
            User: Persisted user snapshot after create/update/reactivate.
        Assumptions:
            `telegram_user_id` is unique across users.
        Raises:
            ValueError: If repository cannot persist or map domain state.
        Side Effects:
            Writes one record in identity storage.
        """
        ...
