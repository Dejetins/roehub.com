from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TelegramUserId:
    """
    TelegramUserId — идентификатор пользователя Telegram в login payload.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/domain/entities/user.py
      - src/trading/contexts/identity/adapters/outbound/security/telegram/
        telegram_login_widget_payload_validator.py
      - migrations/postgres/0001_identity_v1.sql
    """

    value: int

    def __post_init__(self) -> None:
        """
        Validate Telegram user identifier boundaries.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Telegram Login Widget returns positive integer user identifiers.
        Raises:
            ValueError: If value is not positive integer or is bool.
        Side Effects:
            None.
        """
        if type(self.value) is bool:  # noqa: E721
            raise ValueError("TelegramUserId must be int, not bool")
        if self.value <= 0:
            raise ValueError(f"TelegramUserId must be > 0, got {self.value}")

    def __str__(self) -> str:
        """
        Return string representation for logging and deterministic serialization.

        Args:
            None.
        Returns:
            str: Decimal id representation.
        Assumptions:
            Value was validated in constructor.
        Raises:
            None.
        Side Effects:
            None.
        """
        return str(self.value)
