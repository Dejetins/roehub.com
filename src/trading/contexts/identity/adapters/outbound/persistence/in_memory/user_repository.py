from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from trading.contexts.identity.application.ports.user_repository import UserRepository
from trading.contexts.identity.domain.entities import User
from trading.contexts.identity.domain.value_objects import TelegramUserId
from trading.shared_kernel.primitives import PaidLevel, UserId


class InMemoryIdentityUserRepository(UserRepository):
    """
    InMemoryIdentityUserRepository — deterministic in-memory identity repository for dev/test.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/user_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py
      - tests/unit/contexts/identity/application/test_telegram_login_use_case.py
    """

    def __init__(self) -> None:
        """
        Initialize empty in-memory repository state.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Repository instance is process-local and not shared between tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._by_telegram_id: dict[int, User] = {}
        self._by_user_id: dict[str, User] = {}

    def find_by_user_id(self, *, user_id: UserId) -> User | None:
        """
        Find user snapshot by stable user id in local dictionary.

        Args:
            user_id: Stable user identifier.
        Returns:
            User | None: Stored user snapshot or None.
        Assumptions:
            User id dictionary key uses canonical UUID string format.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._by_user_id.get(str(user_id))

    def upsert_telegram_login(
        self,
        *,
        telegram_user_id: TelegramUserId,
        login_at: datetime,
    ) -> User:
        """
        Create or update in-memory user snapshot for successful Telegram login.

        Args:
            telegram_user_id: Telegram identity key.
            login_at: Current UTC login timestamp.
        Returns:
            User: Upserted user snapshot.
        Assumptions:
            Input datetime is timezone-aware UTC.
        Raises:
            ValueError: If domain entity invariants fail.
        Side Effects:
            Mutates in-memory dictionaries.
        """
        existing = self._by_telegram_id.get(telegram_user_id.value)
        if existing is None:
            created = User(
                user_id=UserId(UUID(str(uuid4()))),
                telegram_user_id=telegram_user_id,
                paid_level=PaidLevel.free(),
                created_at=login_at,
                last_login_at=login_at,
                is_deleted=False,
            )
            self._by_telegram_id[telegram_user_id.value] = created
            self._by_user_id[str(created.user_id)] = created
            return created

        updated = existing.reactivated(login_at=login_at)
        self._by_telegram_id[telegram_user_id.value] = updated
        self._by_user_id[str(updated.user_id)] = updated
        return updated
