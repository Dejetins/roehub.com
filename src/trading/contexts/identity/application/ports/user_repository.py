from __future__ import annotations

from datetime import datetime
from typing import Protocol

from trading.contexts.identity.domain.entities import User
from trading.shared_kernel.primitives import UserId


class UserRepository(Protocol):
    """
    UserRepository — порт хранения пользователей identity.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/in_memory/user_repository.py
    """

    def find_by_user_id(self, *, user_id: UserId) -> User | None:
        """
        Find user by stable cross-context user identifier.

        Args:
            user_id: Identity user identifier.
        Returns:
            User | None: Active or deleted user snapshot, or `None` when missing.
        Assumptions:
            Lookup is deterministic and unique by `user_id`.
        Raises:
            ValueError: If repository implementation rejects invalid inputs.
        Side Effects:
            None.
        """
        ...



    def create_local_user(
        self,
        *,
        user_id: UserId,
        created_at: datetime,
    ) -> User:
        """Create a greenfield local identity user with a caller-assigned stable id."""

        ...

    def record_local_login(
        self,
        *,
        user_id: UserId,
        login_at: datetime,
    ) -> User:
        """Record a successful local authentication without changing identity bindings."""

        ...
