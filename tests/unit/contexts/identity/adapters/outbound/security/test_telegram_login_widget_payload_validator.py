from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone

import pytest

from trading.contexts.identity.adapters.outbound.security.telegram import (
    TelegramLoginWidgetPayloadValidator,
)
from trading.contexts.identity.application.ports.telegram_auth_payload_validator import (
    TelegramAuthValidationError,
)

BOT_TOKEN = "test-telegram-bot-token"


def _build_valid_payload(*, auth_date: int, user_id: int) -> dict[str, str]:
    """
    Build deterministic Telegram payload with correct Variant A hash.

    Args:
        auth_date: Unix timestamp in seconds.
        user_id: Telegram user identifier.
    Returns:
        dict[str, str]: Signed payload ready for validator.
    Assumptions:
        Hash is calculated exactly as Telegram Login Widget algorithm defines.
    Raises:
        None.
    Side Effects:
        None.
    """
    payload = {
        "auth_date": str(auth_date),
        "first_name": "Roe",
        "id": str(user_id),
        "username": "roehub_user",
    }
    payload["hash"] = _build_hash(payload=payload)
    return payload



def _build_hash(*, payload: dict[str, str]) -> str:
    """
    Compute Telegram Widget hash from payload and bot token.

    Args:
        payload: Payload without `hash` field.
    Returns:
        str: Lowercase hex hash.
    Assumptions:
        `BOT_TOKEN` is stable for test runtime.
    Raises:
        None.
    Side Effects:
        None.
    """
    data_check_rows = [
        f"{key}={value}"
        for key, value in sorted(payload.items(), key=lambda item: item[0])
        if key != "hash"
    ]
    data_check_string = "\n".join(data_check_rows)
    secret_key = hashlib.sha256(BOT_TOKEN.encode("utf-8")).digest()
    return hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()



def test_validate_accepts_valid_payload() -> None:
    """
    Verify validator accepts payload with correct hash and fresh auth_date.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fresh payload within replay window should be accepted.
    Raises:
        AssertionError: If validator rejects valid payload.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
    payload = _build_valid_payload(auth_date=int(now.timestamp()), user_id=900001)
    validator = TelegramLoginWidgetPayloadValidator(bot_token=BOT_TOKEN)

    telegram_user_id = validator.validate(payload=payload, now=now)

    assert telegram_user_id.value == 900001



def test_validate_rejects_wrong_hash() -> None:
    """
    Verify validator rejects payload with modified hash.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Any hash mismatch must fail deterministically.
    Raises:
        AssertionError: If wrong hash is accepted.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
    payload = _build_valid_payload(auth_date=int(now.timestamp()), user_id=900002)
    payload["hash"] = "0" * 64
    validator = TelegramLoginWidgetPayloadValidator(bot_token=BOT_TOKEN)

    with pytest.raises(TelegramAuthValidationError) as error_info:
        validator.validate(payload=payload, now=now)

    assert error_info.value.code == "invalid_hash"



def test_validate_rejects_stale_auth_date() -> None:
    """
    Verify validator rejects stale auth_date values outside replay window.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Replay window defaults to 24 hours.
    Raises:
        AssertionError: If stale payload is accepted.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
    stale_auth_date = int((now - timedelta(days=2)).timestamp())
    payload = _build_valid_payload(auth_date=stale_auth_date, user_id=900003)
    validator = TelegramLoginWidgetPayloadValidator(bot_token=BOT_TOKEN)

    with pytest.raises(TelegramAuthValidationError) as error_info:
        validator.validate(payload=payload, now=now)

    assert error_info.value.code == "stale_auth_date"
