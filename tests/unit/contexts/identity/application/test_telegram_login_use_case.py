from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping

from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.security.jwt import Hs256JwtCodec
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.ports.telegram_auth_payload_validator import (
    TelegramAuthPayloadValidator,
)
from trading.contexts.identity.application.use_cases import TelegramLoginUseCase
from trading.contexts.identity.domain.value_objects import TelegramUserId


class _ClockStub(IdentityClock):
    """
    Deterministic clock stub returning predefined UTC timestamps sequentially.
    """

    def __init__(self, *, values: tuple[datetime, ...]) -> None:
        """
        Initialize clock stub with finite timestamp sequence.

        Args:
            values: Ordered UTC timestamps to return on `now()` calls.
        Returns:
            None.
        Assumptions:
            All values are timezone-aware UTC datetimes.
        Raises:
            ValueError: If sequence is empty.
        Side Effects:
            None.
        """
        if not values:
            raise ValueError("_ClockStub requires at least one datetime value")
        self._values = list(values)

    def now(self) -> datetime:
        """
        Return next predefined timestamp, keeping last value for extra calls.

        Args:
            None.
        Returns:
            datetime: Predefined UTC datetime.
        Assumptions:
            Caller performs a small bounded number of reads.
        Raises:
            None.
        Side Effects:
            Consumes one element from internal list until one value remains.
        """
        if len(self._values) == 1:
            return self._values[0]
        return self._values.pop(0)


class _TelegramValidatorStub(TelegramAuthPayloadValidator):
    """
    Telegram validator stub returning fixed telegram user id for use-case tests.
    """

    def __init__(self, *, telegram_user_id: TelegramUserId) -> None:
        """
        Store deterministic Telegram user id returned by validate().

        Args:
            telegram_user_id: Telegram user id returned for each validation.
        Returns:
            None.
        Assumptions:
            Stub bypasses hash/auth_date checks to isolate upsert behavior.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._telegram_user_id = telegram_user_id

    def validate(self, *, payload: Mapping[str, str], now: datetime) -> TelegramUserId:
        """
        Return configured Telegram user id regardless of payload.

        Args:
            payload: Raw payload mapping.
            now: Current UTC datetime.
        Returns:
            TelegramUserId: Configured fixed id.
        Assumptions:
            Payload non-emptiness is validated by use-case itself.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = payload
        _ = now
        return self._telegram_user_id



def test_telegram_login_use_case_upserts_user_and_updates_last_login_at() -> None:
    """
    Verify first login creates user with free level and re-login updates last_login_at.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Same Telegram user id maps to same stable UserId.
    Raises:
        AssertionError: If upsert invariants are broken.
    Side Effects:
        None.
    """
    first_login = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
    second_login = first_login + timedelta(hours=3)
    use_case_clock = _ClockStub(values=(first_login, second_login))
    jwt_clock = _ClockStub(values=(second_login,))

    repository = InMemoryIdentityUserRepository()
    validator = _TelegramValidatorStub(telegram_user_id=TelegramUserId(777001))
    jwt_codec = Hs256JwtCodec(secret_key="identity-test-secret", clock=jwt_clock)
    use_case = TelegramLoginUseCase(
        validator=validator,
        user_repository=repository,
        jwt_codec=jwt_codec,
        clock=use_case_clock,
        jwt_ttl_days=7,
    )

    first_result = use_case.login(payload={"id": "777001"})
    second_result = use_case.login(payload={"id": "777001"})

    assert str(first_result.paid_level) == "free"
    assert first_result.user_id == second_result.user_id
    assert int((first_result.jwt_expires_at - first_login).total_seconds()) == 7 * 24 * 3600

    stored_user = repository.find_by_user_id(user_id=first_result.user_id)
    assert stored_user is not None
    assert stored_user.created_at == first_login
    assert stored_user.last_login_at == second_login
    assert str(stored_user.paid_level) == "free"
