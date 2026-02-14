from __future__ import annotations

import hashlib
import hmac
from datetime import datetime
from typing import Mapping

from trading.contexts.identity.application.ports.telegram_auth_payload_validator import (
    TelegramAuthPayloadValidator,
    TelegramAuthValidationError,
)
from trading.contexts.identity.domain.value_objects import TelegramUserId


class TelegramLoginWidgetPayloadValidator(TelegramAuthPayloadValidator):
    """
    TelegramLoginWidgetPayloadValidator — проверка Variant A payload Telegram Login Widget.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/telegram_auth_payload_validator.py
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - apps/api/routes/identity.py
    """

    def __init__(
        self,
        *,
        bot_token: str,
        auth_max_age_seconds: int = 86_400,
        clock_skew_seconds: int = 60,
    ) -> None:
        """
        Initialize validator with Telegram bot token and replay-window policy.

        Args:
            bot_token: Telegram bot token used to derive HMAC secret.
            auth_max_age_seconds: Maximum accepted age of `auth_date`.
            clock_skew_seconds: Allowed future skew for `auth_date`.
        Returns:
            None.
        Assumptions:
            Bot token is non-empty and belongs to Telegram Login Widget app.
        Raises:
            ValueError: If constructor arguments are invalid.
        Side Effects:
            None.
        """
        token = bot_token.strip()
        if not token:
            raise ValueError("TelegramLoginWidgetPayloadValidator requires non-empty bot_token")
        if auth_max_age_seconds <= 0:
            raise ValueError(
                "TelegramLoginWidgetPayloadValidator requires auth_max_age_seconds > 0"
            )
        if clock_skew_seconds < 0:
            raise ValueError("TelegramLoginWidgetPayloadValidator requires clock_skew_seconds >= 0")

        self._bot_token = token
        self._auth_max_age_seconds = auth_max_age_seconds
        self._clock_skew_seconds = clock_skew_seconds

    def validate(self, *, payload: Mapping[str, str], now: datetime) -> TelegramUserId:
        """
        Validate Telegram payload hash and freshness and return Telegram user id.

        Args:
            payload: Telegram widget payload fields including `hash`.
            now: Current UTC datetime.
        Returns:
            TelegramUserId: Verified Telegram user identifier.
        Assumptions:
            Payload values were normalized before validator invocation.
        Raises:
            TelegramAuthValidationError: If payload is malformed, stale, or hash mismatches.
        Side Effects:
            None.
        """
        now_offset = now.utcoffset()
        if now.tzinfo is None or now_offset is None:
            raise ValueError("TelegramLoginWidgetPayloadValidator requires UTC 'now' datetime")
        if now_offset.total_seconds() != 0:
            raise ValueError("TelegramLoginWidgetPayloadValidator requires UTC 'now' datetime")

        received_hash = payload.get("hash", "").strip().lower()
        if not received_hash:
            raise TelegramAuthValidationError(
                code="missing_hash",
                message="Telegram payload must include non-empty hash",
            )

        auth_date_raw = payload.get("auth_date", "").strip()
        if not auth_date_raw:
            raise TelegramAuthValidationError(
                code="missing_auth_date",
                message="Telegram payload must include auth_date",
            )

        try:
            auth_date_seconds = int(auth_date_raw)
        except ValueError as error:
            raise TelegramAuthValidationError(
                code="invalid_auth_date",
                message="Telegram payload auth_date must be integer seconds",
            ) from error

        now_seconds = int(now.timestamp())
        if auth_date_seconds < now_seconds - self._auth_max_age_seconds:
            raise TelegramAuthValidationError(
                code="stale_auth_date",
                message="Telegram payload auth_date is stale",
            )
        if auth_date_seconds > now_seconds + self._clock_skew_seconds:
            raise TelegramAuthValidationError(
                code="future_auth_date",
                message="Telegram payload auth_date is in the future",
            )

        telegram_user_raw = payload.get("id", "").strip()
        if not telegram_user_raw:
            raise TelegramAuthValidationError(
                code="missing_telegram_user_id",
                message="Telegram payload must include id",
            )

        try:
            telegram_user_id = TelegramUserId(int(telegram_user_raw))
        except ValueError as error:
            raise TelegramAuthValidationError(
                code="invalid_telegram_user_id",
                message="Telegram payload id must be positive integer",
            ) from error

        expected_hash = _build_expected_hash(bot_token=self._bot_token, payload=payload)
        if not hmac.compare_digest(received_hash, expected_hash):
            raise TelegramAuthValidationError(
                code="invalid_hash",
                message="Telegram payload hash verification failed",
            )

        return telegram_user_id



def _build_expected_hash(*, bot_token: str, payload: Mapping[str, str]) -> str:
    """
    Build expected lowercase hash using Telegram Login Widget algorithm.

    Args:
        bot_token: Telegram bot token.
        payload: Raw payload fields including hash.
    Returns:
        str: Lowercase hex digest for comparison with payload hash.
    Assumptions:
        Payload mapping preserves textual values exactly as received.
    Raises:
        None.
    Side Effects:
        None.
    """
    data_check_string = _build_data_check_string(payload=payload)
    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
    return hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest().lower()



def _build_data_check_string(*, payload: Mapping[str, str]) -> str:
    """
    Build deterministic Telegram `data_check_string` from payload.

    Args:
        payload: Telegram widget payload mapping.
    Returns:
        str: Newline-delimited `key=value` rows sorted by key.
    Assumptions:
        `hash` field must be excluded from data-check string.
    Raises:
        None.
    Side Effects:
        None.
    """
    items = [
        (key, value)
        for key, value in payload.items()
        if key != "hash" and value is not None and str(value).strip() != ""
    ]
    items.sort(key=lambda item: item[0])
    return "\n".join(f"{key}={value}" for key, value in items)
