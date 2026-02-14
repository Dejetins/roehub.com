from __future__ import annotations

from datetime import datetime
from typing import Mapping, Protocol

from trading.contexts.identity.domain.value_objects import TelegramUserId


class TelegramAuthValidationError(ValueError):
    """
    TelegramAuthValidationError — ошибка проверки payload Telegram Login Widget.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/contexts/identity/adapters/outbound/security/telegram/
        telegram_login_widget_payload_validator.py
      - apps/api/routes/identity.py
    """

    def __init__(self, *, code: str, message: str) -> None:
        """
        Initialize validator error with stable code and message.

        Args:
            code: Machine-readable deterministic error code.
            message: Human-readable deterministic message.
        Returns:
            None.
        Assumptions:
            API layer exposes `code` as part of deterministic error payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(message)
        self.code = code
        self.message = message


class TelegramAuthPayloadValidator(Protocol):
    """
    TelegramAuthPayloadValidator — порт проверки Variant A payload от Telegram Widget.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/contexts/identity/adapters/outbound/security/telegram/
        telegram_login_widget_payload_validator.py
      - apps/api/routes/identity.py
    """

    def validate(self, *, payload: Mapping[str, str], now: datetime) -> TelegramUserId:
        """
        Validate Telegram payload signature/freshness and return Telegram user id.

        Args:
            payload: Raw payload fields from Telegram widget.
            now: Current UTC datetime for replay-window checks.
        Returns:
            TelegramUserId: Validated Telegram user identifier.
        Assumptions:
            Payload contains `hash`, `auth_date`, and `id` fields.
        Raises:
            TelegramAuthValidationError: If payload is missing fields, stale, or has bad hash.
        Side Effects:
            None.
        """
        ...
