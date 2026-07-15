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

    def find_by_keycloak_subject(self, *, keycloak_subject: str) -> User | None:
        """
        Find user by external Keycloak subject bound to local Roehub identity.

        Args:
            keycloak_subject: Opaque external subject from Keycloak `sub`.
        Returns:
            User | None: Active or deleted user snapshot, or `None` when missing.
        Assumptions:
            Keycloak subject is unique across all identity users.
        Raises:
            ValueError: If repository implementation rejects blank or malformed subject.
        Side Effects:
            None.
        """
        ...

    def upsert_keycloak_login(
        self,
        *,
        keycloak_subject: str,
        login_at: datetime,
    ) -> User:
        """
        Create-or-update user by Keycloak subject during successful login.

        Args:
            keycloak_subject: Opaque external subject from Keycloak `sub`.
            login_at: Current UTC login timestamp.
        Returns:
            User: Persisted local Roehub user snapshot after create/update/reactivate.
        Assumptions:
            Repository preserves stable local `user_id` for repeat logins of one subject.
            `paid_level` source of truth remains Roehub identity storage, not provider claims.
        Raises:
            ValueError: If repository cannot persist or map domain state.
        Side Effects:
            Writes one record in identity storage.
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
