from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading.contexts.identity.domain.value_objects.telegram_user_id import TelegramUserId
from trading.shared_kernel.primitives import PaidLevel, UserId


@dataclass(frozen=True, slots=True)
class User:
    """
    User — минимальная доменная модель пользователя identity v1.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/user_repository.py
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - migrations/postgres/0001_identity_v1.sql
    """

    user_id: UserId
    telegram_user_id: TelegramUserId
    paid_level: PaidLevel
    created_at: datetime
    last_login_at: datetime | None
    is_deleted: bool = False

    def __post_init__(self) -> None:
        """
        Validate identity user invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `created_at` and `last_login_at` are UTC timezone-aware datetimes.
        Raises:
            ValueError: If datetime invariants are violated.
        Side Effects:
            None.
        """
        _ensure_utc_datetime(name="created_at", value=self.created_at)
        if self.last_login_at is not None:
            _ensure_utc_datetime(name="last_login_at", value=self.last_login_at)
            if self.last_login_at < self.created_at:
                raise ValueError("User.last_login_at cannot be before created_at")

    def reactivated(self, *, login_at: datetime) -> User:
        """
        Return immutable copy reflecting successful Telegram login.

        Args:
            login_at: Current login timestamp in UTC.
        Returns:
            User: Updated user with `last_login_at` set and `is_deleted=False`.
        Assumptions:
            Login timestamp is timezone-aware UTC datetime.
        Raises:
            ValueError: If `login_at` breaks datetime invariants.
        Side Effects:
            None.
        """
        _ensure_utc_datetime(name="login_at", value=login_at)
        if login_at < self.created_at:
            raise ValueError("User.reactivated login_at cannot be before created_at")
        return User(
            user_id=self.user_id,
            telegram_user_id=self.telegram_user_id,
            paid_level=self.paid_level,
            created_at=self.created_at,
            last_login_at=login_at,
            is_deleted=False,
        )



def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone awareness and UTC offset for datetime fields.

    Args:
        name: Field name for deterministic error messages.
        value: Datetime value to validate.
    Returns:
        None.
    Assumptions:
        UTC datetimes are represented with timezone info and zero offset.
    Raises:
        ValueError: If datetime is naive or not in UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{name} must be UTC datetime")
