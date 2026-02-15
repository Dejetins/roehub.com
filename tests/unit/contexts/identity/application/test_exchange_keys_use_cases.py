from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentityExchangeKeysRepository,
)
from trading.contexts.identity.adapters.outbound.security.exchange_keys import (
    AesGcmEnvelopeExchangeKeysSecretCipher,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.use_cases import (
    CreateExchangeKeyUseCase,
    DeleteExchangeKeyUseCase,
    ExchangeKeyAlreadyExistsError,
    ExchangeKeyNotFoundError,
    ListExchangeKeysUseCase,
)
from trading.shared_kernel.primitives import UserId


class _MutableClock(IdentityClock):
    """
    Mutable deterministic UTC clock for exchange keys application use-case tests.
    """

    def __init__(self, *, now_value: datetime) -> None:
        """
        Initialize mutable clock with initial timezone-aware UTC datetime.

        Args:
            now_value: Initial UTC datetime value.
        Returns:
            None.
        Assumptions:
            Timestamp changes are controlled only through `set_now`.
        Raises:
            ValueError: If datetime is naive or non-UTC.
        Side Effects:
            None.
        """
        self._now_value = _ensure_utc_datetime(value=now_value, field_name="now_value")

    def set_now(self, *, now_value: datetime) -> None:
        """
        Update deterministic clock value.

        Args:
            now_value: New UTC datetime value.
        Returns:
            None.
        Assumptions:
            Test controls all time changes deterministically.
        Raises:
            ValueError: If datetime is naive or non-UTC.
        Side Effects:
            Mutates internal clock state.
        """
        self._now_value = _ensure_utc_datetime(value=now_value, field_name="now_value")

    def now(self) -> datetime:
        """
        Return current deterministic UTC timestamp.

        Args:
            None.
        Returns:
            datetime: Current test-controlled timestamp.
        Assumptions:
            Time does not auto-progress between calls.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._now_value


def test_create_exchange_key_encrypts_secret_and_returns_masked_projection() -> None:
    """
    Verify create use-case encrypts secret fields and returns API-safe masked projection.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        In-memory repository stores encrypted blobs without plaintext exposure.
    Raises:
        AssertionError: If create flow violates encryption or projection contract.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=now)
    repository = InMemoryIdentityExchangeKeysRepository()
    cipher = _build_cipher()
    create_use_case = CreateExchangeKeyUseCase(
        repository=repository,
        secret_cipher=cipher,
        clock=clock,
    )
    user_id = UserId.from_string("00000000-0000-0000-0000-000000001001")

    result = create_use_case.create(
        user_id=user_id,
        exchange_name="binance",
        market_type="spot",
        label="main",
        permissions="read",
        api_key="ABCD12345678",
        api_secret="secret-value-1",
        passphrase="passphrase-value-1",
    )

    assert result.exchange_name == "binance"
    assert result.market_type == "spot"
    assert result.permissions == "read"
    assert result.api_key == "****5678"
    assert result.created_at == now
    assert result.updated_at == now

    stored = repository.list_active_for_user(user_id=user_id)
    assert len(stored) == 1
    stored_row = stored[0]
    assert stored_row.api_secret_enc != b"secret-value-1"
    assert cipher.decrypt_secret(secret_enc=stored_row.api_secret_enc) == "secret-value-1"
    assert stored_row.passphrase_enc is not None
    assert cipher.decrypt_secret(secret_enc=stored_row.passphrase_enc) == "passphrase-value-1"


def test_create_exchange_key_rejects_active_duplicate() -> None:
    """
    Verify create use-case returns deterministic conflict for active duplicate key tuple.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Duplicate uniqueness is `(user_id, exchange_name, market_type, api_key)`.
    Raises:
        AssertionError: If duplicate is unexpectedly accepted.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 15, 10, 5, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=now)
    repository = InMemoryIdentityExchangeKeysRepository()
    create_use_case = CreateExchangeKeyUseCase(
        repository=repository,
        secret_cipher=_build_cipher(),
        clock=clock,
    )
    user_id = UserId.from_string("00000000-0000-0000-0000-000000001002")

    create_use_case.create(
        user_id=user_id,
        exchange_name="bybit",
        market_type="futures",
        label=None,
        permissions="trade",
        api_key="DUPLICATE-KEY-0001",
        api_secret="secret-value-a",
        passphrase=None,
    )

    with pytest.raises(ExchangeKeyAlreadyExistsError):
        create_use_case.create(
            user_id=user_id,
            exchange_name="bybit",
            market_type="futures",
            label="duplicate",
            permissions="trade",
            api_key="DUPLICATE-KEY-0001",
            api_secret="secret-value-b",
            passphrase=None,
        )


def test_list_exchange_keys_returns_deterministic_order_and_skips_deleted_rows() -> None:
    """
    Verify list use-case returns active rows in deterministic order and excludes soft-deleted.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        List ordering must remain deterministic as `created_at ASC, key_id ASC`.
    Raises:
        AssertionError: If ordering or delete filtering is incorrect.
    Side Effects:
        None.
    """
    base_time = datetime(2026, 2, 15, 11, 0, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=base_time)
    repository = InMemoryIdentityExchangeKeysRepository()
    create_use_case = CreateExchangeKeyUseCase(
        repository=repository,
        secret_cipher=_build_cipher(),
        clock=clock,
    )
    list_use_case = ListExchangeKeysUseCase(repository=repository)
    delete_use_case = DeleteExchangeKeyUseCase(repository=repository, clock=clock)
    user_id = UserId.from_string("00000000-0000-0000-0000-000000001003")

    first = create_use_case.create(
        user_id=user_id,
        exchange_name="binance",
        market_type="spot",
        label="first",
        permissions="read",
        api_key="ORDER-KEY-0001",
        api_secret="secret-order-1",
        passphrase=None,
    )

    clock.set_now(now_value=base_time.replace(minute=1))
    second = create_use_case.create(
        user_id=user_id,
        exchange_name="bybit",
        market_type="futures",
        label="second",
        permissions="trade",
        api_key="ORDER-KEY-0002",
        api_secret="secret-order-2",
        passphrase=None,
    )

    listed_before_delete = list_use_case.list_for_user(user_id=user_id)
    assert [item.key_id for item in listed_before_delete] == [first.key_id, second.key_id]

    clock.set_now(now_value=base_time.replace(minute=2))
    delete_use_case.delete(user_id=user_id, key_id=first.key_id)

    listed_after_delete = list_use_case.list_for_user(user_id=user_id)
    assert [item.key_id for item in listed_after_delete] == [second.key_id]


def test_delete_exchange_key_soft_deletes_and_missing_key_raises_not_found() -> None:
    """
    Verify delete use-case performs soft-delete and raises deterministic not-found on repeat.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repeated delete of same key is treated as not-found by contract.
    Raises:
        AssertionError: If soft-delete semantics or not-found behavior is broken.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 15, 11, 30, 0, tzinfo=timezone.utc)
    clock = _MutableClock(now_value=now)
    repository = InMemoryIdentityExchangeKeysRepository()
    create_use_case = CreateExchangeKeyUseCase(
        repository=repository,
        secret_cipher=_build_cipher(),
        clock=clock,
    )
    delete_use_case = DeleteExchangeKeyUseCase(repository=repository, clock=clock)
    user_id = UserId.from_string("00000000-0000-0000-0000-000000001004")

    created = create_use_case.create(
        user_id=user_id,
        exchange_name="binance",
        market_type="futures",
        label=None,
        permissions="trade",
        api_key="DELETE-KEY-0001",
        api_secret="secret-delete-1",
        passphrase=None,
    )

    clock.set_now(now_value=now.replace(minute=31))
    delete_use_case.delete(user_id=user_id, key_id=created.key_id)

    stored_row = repository._rows[str(created.key_id)]
    assert stored_row.is_deleted is True
    assert stored_row.deleted_at == now.replace(minute=31)

    with pytest.raises(ExchangeKeyNotFoundError):
        delete_use_case.delete(user_id=user_id, key_id=created.key_id)


def _build_cipher() -> AesGcmEnvelopeExchangeKeysSecretCipher:
    """
    Build deterministic exchange keys envelope cipher for tests.

    Args:
        None.
    Returns:
        AesGcmEnvelopeExchangeKeysSecretCipher: Cipher instance with test KEK.
    Assumptions:
        Test KEK is static and valid 32-byte base64 material.
    Raises:
        ValueError: If test KEK constant is malformed.
    Side Effects:
        None.
    """
    return AesGcmEnvelopeExchangeKeysSecretCipher(
        kek_b64="cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    )



def _ensure_utc_datetime(*, value: datetime, field_name: str) -> datetime:
    """
    Validate datetime is timezone-aware UTC and return same value.

    Args:
        value: Datetime value to validate.
        field_name: Label for deterministic error messages.
    Returns:
        datetime: Same validated datetime.
    Assumptions:
        UTC datetimes have zero UTC offset.
    Raises:
        ValueError: If datetime is naive or not UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{field_name} must be UTC datetime")
    return value
