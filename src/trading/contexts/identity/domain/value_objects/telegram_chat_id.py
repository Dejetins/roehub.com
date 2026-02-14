from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TelegramChatId:
    """
    TelegramChatId — идентификатор Telegram-чата для будущих уведомлений.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - migrations/postgres/0001_identity_v1.sql
      - src/trading/contexts/identity/domain/entities/user.py
      - src/trading/contexts/identity/domain/value_objects/telegram_user_id.py
    """

    value: int

    def __post_init__(self) -> None:
        """
        Validate Telegram chat identifier boundary.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Chat identifier must be non-zero integer.
        Raises:
            ValueError: If value is zero or bool.
        Side Effects:
            None.
        """
        if type(self.value) is bool:  # noqa: E721
            raise ValueError("TelegramChatId must be int, not bool")
        if self.value == 0:
            raise ValueError("TelegramChatId must be non-zero")

    def __str__(self) -> str:
        """
        Return string representation for deterministic serialization.

        Args:
            None.
        Returns:
            str: Decimal chat id.
        Assumptions:
            Value already validated.
        Raises:
            None.
        Side Effects:
            None.
        """
        return str(self.value)
