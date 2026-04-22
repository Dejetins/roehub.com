from fastapi import HTTPException
from starlette.requests import Request

from trading.contexts.identity.application.ports.current_user import (
    CurrentUser,
    CurrentUserPrincipal,
    CurrentUserUnauthorizedError,
)


class RequireCurrentUserDependency:
    """
    RequireCurrentUserDependency — FastAPI dependency resolving authenticated identity user.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/current_user.py
      - src/trading/contexts/identity/adapters/outbound/security/current_user/
        roehub_session_current_user.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py
    """

    def __init__(self, *, current_user: CurrentUser, cookie_name: str) -> None:
        """
        Initialize dependency with current-user port and cookie key.

        Args:
            current_user: Port resolving user principal from Roehub session id.
            cookie_name: Cookie key where opaque Roehub session id is stored.
        Returns:
            None.
        Assumptions:
            Cookie name is deterministic and shared with login route writer.
        Raises:
            ValueError: If dependencies are invalid.
        Side Effects:
            None.
        """
        normalized_cookie_name = cookie_name.strip()
        if current_user is None:  # type: ignore[truthy-bool]
            raise ValueError("RequireCurrentUserDependency requires current_user")
        if not normalized_cookie_name:
            raise ValueError("RequireCurrentUserDependency requires non-empty cookie_name")

        self._current_user = current_user
        self._cookie_name = normalized_cookie_name

    def __call__(self, request: Request) -> CurrentUserPrincipal:
        """
        Resolve authenticated principal from Roehub session cookie.

        Args:
            request: FastAPI HTTP request.
        Returns:
            CurrentUserPrincipal: Verified user context.
        Assumptions:
            Browser-authenticated requests carry one opaque Roehub session cookie.
        Raises:
            HTTPException: 401 with deterministic payload for unauthorized requests.
        Side Effects:
            None.
        """
        session_id = _resolve_session_id(
            request=request,
            cookie_name=self._cookie_name,
        )
        try:
            return self._current_user.require(session_id=session_id)
        except CurrentUserUnauthorizedError as error:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": error.code,
                    "message": error.message,
                },
            ) from error


def _resolve_session_id(*, request: Request, cookie_name: str) -> str | None:
    """
    Resolve Roehub session id from configured browser cookie.

    Args:
        request: FastAPI HTTP request.
        cookie_name: Auth cookie key configured for API.
    Returns:
        str | None: Resolved session id value or `None` when absent.
    Assumptions:
        Browser/web auth path is cookie-only and ignores Authorization header.
    Raises:
        None.
    Side Effects:
        None.
    """
    cookie_session_id = request.cookies.get(cookie_name)
    if cookie_session_id is None:
        return None
    normalized_cookie_session_id = cookie_session_id.strip()
    if not normalized_cookie_session_id:
        return None
    return normalized_cookie_session_id
