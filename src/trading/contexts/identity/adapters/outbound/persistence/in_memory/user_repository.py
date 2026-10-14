from __future__ import annotations

from datetime import datetime

from trading.contexts.identity.application.ports.user_repository import UserRepository
from trading.contexts.identity.domain.entities import User
from trading.shared_kernel.primitives import PaidLevel, UserId


class InMemoryIdentityUserRepository(UserRepository):
    """
    InMemoryIdentityUserRepository — deterministic in-memory identity repository for dev/test.

    Docs:
      - docs/architecture/identity/oidc-authentication-provider-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/user_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py
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



    def create_local_user(
        self,
        *,
        user_id: UserId,
        created_at: datetime,
    ) -> User:
        if str(user_id) in self._by_user_id:
            raise ValueError("local user already exists")
        user = User(
            user_id=user_id,
            paid_level=PaidLevel.free(),
            created_at=created_at,
            last_login_at=created_at,
            is_deleted=False,
        )
        self._by_user_id[str(user_id)] = user
        return user

    def record_local_login(
        self,
        *,
        user_id: UserId,
        login_at: datetime,
    ) -> User:
        existing = self._by_user_id.get(str(user_id))
        if existing is None:
            raise ValueError("local user is not found")
        updated = existing.reactivated(login_at=login_at)
        self._by_user_id[str(user_id)] = updated
        return updated
