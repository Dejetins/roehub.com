from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from trading.contexts.identity.application.ports.user_repository import UserRepository
from trading.contexts.identity.domain.entities import User
from trading.shared_kernel.primitives import PaidLevel, UserId


class InMemoryIdentityUserRepository(UserRepository):
    """
    InMemoryIdentityUserRepository — deterministic in-memory identity repository for dev/test.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
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
        self._user_id_by_keycloak_subject: dict[str, str] = {}

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

    def find_by_keycloak_subject(self, *, keycloak_subject: str) -> User | None:
        """
        Find user snapshot by bound Keycloak subject in local dictionaries.

        Args:
            keycloak_subject: Opaque external subject from Keycloak.
        Returns:
            User | None: Stored user snapshot or `None` when subject is unknown.
        Assumptions:
            Subject is normalized as non-empty stripped string before dictionary lookup.
        Raises:
            ValueError: If subject is blank after normalization.
        Side Effects:
            None.
        """
        normalized_keycloak_subject = _normalize_keycloak_subject(
            keycloak_subject=keycloak_subject
        )
        stored_user_id = self._user_id_by_keycloak_subject.get(normalized_keycloak_subject)
        if stored_user_id is None:
            return None
        return self._by_user_id.get(stored_user_id)

    def upsert_keycloak_login(
        self,
        *,
        keycloak_subject: str,
        login_at: datetime,
    ) -> User:
        """
        Create or update in-memory user snapshot for successful Keycloak login.

        Args:
            keycloak_subject: Opaque external subject from Keycloak.
            login_at: Current UTC login timestamp.
        Returns:
            User: Upserted user snapshot.
        Assumptions:
            Input datetime is timezone-aware UTC.
        Raises:
            ValueError: If domain entity invariants fail.
        Side Effects:
            Mutates in-memory dictionary.
        """
        normalized_keycloak_subject = _normalize_keycloak_subject(
            keycloak_subject=keycloak_subject
        )
        existing = self.find_by_keycloak_subject(keycloak_subject=normalized_keycloak_subject)
        if existing is None:
            new_user_id = UserId(uuid4())
            created = User(
                user_id=new_user_id,
                paid_level=PaidLevel.free(),
                created_at=login_at,
                last_login_at=login_at,
                is_deleted=False,
            )
            self._by_user_id[str(created.user_id)] = created
            self._user_id_by_keycloak_subject[normalized_keycloak_subject] = str(
                created.user_id
            )
            return created

        updated = existing.reactivated(login_at=login_at)
        self._by_user_id[str(updated.user_id)] = updated
        return updated


def _normalize_keycloak_subject(*, keycloak_subject: str) -> str:
    """
    Normalize Keycloak subject string for repository dictionary keys.

    Args:
        keycloak_subject: Raw external subject value.
    Returns:
        str: Non-empty stripped subject string.
    Assumptions:
        Keycloak subject is opaque and should not be transformed beyond whitespace trim.
    Raises:
        ValueError: If subject is blank after normalization.
    Side Effects:
        None.
    """
    normalized_subject = keycloak_subject.strip()
    if not normalized_subject:
        raise ValueError("keycloak_subject must be non-empty")
    return normalized_subject
