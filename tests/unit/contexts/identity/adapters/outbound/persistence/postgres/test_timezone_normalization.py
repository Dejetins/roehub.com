from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from trading.contexts.identity.adapters.outbound.persistence.postgres import (
    exchange_keys_repository,
    user_repository,
)


def test_map_user_row_normalizes_non_utc_timestamps() -> None:
    local_tz = timezone(timedelta(hours=3))
    created_at_local = datetime(2026, 3, 22, 21, 0, 0, tzinfo=local_tz)
    last_login_at_local = datetime(2026, 3, 22, 21, 5, 0, tzinfo=local_tz)

    mapped = user_repository._map_user_row(
        row={
            "user_id": str(uuid4()),
            "paid_level": "free",
            "created_at": created_at_local,
            "last_login_at": last_login_at_local,
            "is_deleted": False,
        }
    )

    assert mapped.created_at.tzinfo is not None
    assert mapped.created_at.utcoffset() == timedelta(0)
    assert mapped.created_at == created_at_local.astimezone(timezone.utc)
    assert mapped.last_login_at is not None
    assert mapped.last_login_at.utcoffset() == timedelta(0)
    assert mapped.last_login_at == last_login_at_local.astimezone(timezone.utc)

def test_map_exchange_key_row_normalizes_non_utc_timestamps() -> None:
    local_tz = timezone(timedelta(hours=3))
    created_at_local = datetime(2026, 3, 22, 21, 15, 0, tzinfo=local_tz)
    updated_at_local = datetime(2026, 3, 22, 21, 16, 0, tzinfo=local_tz)

    mapped = exchange_keys_repository._map_exchange_key_row(
        row={
            "key_id": str(uuid4()),
            "user_id": str(uuid4()),
            "exchange_name": "binance",
            "market_type": "spot",
            "label": "main",
            "permissions": "trade",
            "api_key_enc": b"api-key-enc",
            "api_key_hash": b"x" * 32,
            "api_key_last4": "1234",
            "api_secret_enc": b"api-secret-enc",
            "passphrase_enc": None,
            "created_at": created_at_local,
            "updated_at": updated_at_local,
            "is_deleted": False,
            "deleted_at": None,
        }
    )

    assert mapped.created_at.utcoffset() == timedelta(0)
    assert mapped.created_at == created_at_local.astimezone(timezone.utc)
    assert mapped.updated_at.utcoffset() == timedelta(0)
    assert mapped.updated_at == updated_at_local.astimezone(timezone.utc)
