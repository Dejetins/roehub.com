from __future__ import annotations

from uuid import UUID

from trading.contexts.identity.application.ports import IdentityClock
from trading.contexts.identity.application.ports.current_user import (
    CurrentUser,
    CurrentUserPrincipal,
    CurrentUserUnauthorizedError,
)
from trading.contexts.identity.application.ports.session_repository import SessionRepository
from trading.contexts.identity.application.ports.user_repository import UserRepository
from trading.shared_kernel.primitives import PaidLevel


class RoehubSessionCurrentUser(CurrentUser):
    """
    RoehubSessionCurrentUser resolves authenticated principal from local Roehub session storage.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/current_user.py
      - src/trading/contexts/identity/application/ports/session_repository.py
      - src/trading/contexts/identity/application/ports/user_repository.py
    """

    def __init__(
        self,
        *,
        session_repository: SessionRepository,
        user_repository: UserRepository,
        clock: IdentityClock,
    ) -> None:
        """
        Initialize resolver with session storage, user storage, and current-time source.

        Args:
            session_repository: Persisted Roehub session storage port.
            user_repository: Local Roehub user storage port.
            clock: Current UTC clock for expiry evaluation.
        Returns:
            None.
        Assumptions:
            Repositories share one consistent local Roehub identity model.
        Raises:
            ValueError: If dependencies are missing.
        Side Effects:
            None.
        """
        if session_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RoehubSessionCurrentUser requires session_repository")
        if user_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RoehubSessionCurrentUser requires user_repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("RoehubSessionCurrentUser requires clock")

        self._session_repository = session_repository
        self._user_repository = user_repository
        self._clock = clock

    def require(self, *, session_id: str | None) -> CurrentUserPrincipal:
        """
        Resolve authenticated principal from persisted Roehub session id.

        Args:
            session_id: Opaque Roehub session identifier from browser cookie.
        Returns:
            CurrentUserPrincipal: Authenticated local Roehub user context.
        Assumptions:
            Session id is UUID-compatible and addresses local session persistence.
        Raises:
            CurrentUserUnauthorizedError: If session is missing, invalid, inactive, or user is unavailable.
        Side Effects:
            None.
        """
        normalized_session_id = "" if session_id is None else session_id.strip()
        if not normalized_session_id:
            raise CurrentUserUnauthorizedError(
                code="missing_session_id",
                message="Session id is required",
            )

        try:
            parsed_session_id = UUID(normalized_session_id)
        except ValueError as error:
            raise CurrentUserUnauthorizedError(
                code="invalid_session_id",
                message="Session id must be UUID",
            ) from error

        session = self._session_repository.find_by_session_id(session_id=parsed_session_id)
        if session is None:
            raise CurrentUserUnauthorizedError(
                code="session_not_found",
                message="Session is not found",
            )

        now = self._clock.now()
        if not session.is_active_at(at=now):
            raise CurrentUserUnauthorizedError(
                code="inactive_session",
                message="Session is inactive",
            )

        user = self._user_repository.find_by_user_id(user_id=session.user_id)
        if user is None:
            raise CurrentUserUnauthorizedError(
                code="user_not_found",
                message="Session user is not found",
            )
        if user.is_deleted:
            raise CurrentUserUnauthorizedError(
                code="inactive_user",
                message="Session user is inactive",
            )

        paid_level = PaidLevel(str(user.paid_level))
        return CurrentUserPrincipal(
            user_id=user.user_id,
            paid_level=paid_level,
        )
