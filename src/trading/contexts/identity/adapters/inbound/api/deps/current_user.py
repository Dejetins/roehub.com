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
        jwt_cookie_current_user.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py
    """

    def __init__(self, *, current_user: CurrentUser, cookie_name: str) -> None:
        """
        Initialize dependency with current-user port and cookie key.

        Args:
            current_user: Port resolving user principal from JWT token.
            cookie_name: Cookie key where JWT token is stored.
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
        Resolve authenticated principal from incoming request cookies.

        Args:
            request: FastAPI HTTP request.
        Returns:
            CurrentUserPrincipal: Verified user context.
        Assumptions:
            JWT token is stored in configured cookie key.
        Raises:
            HTTPException: 401 with deterministic payload for unauthorized requests.
        Side Effects:
            None.
        """
        token = request.cookies.get(self._cookie_name)
        try:
            return self._current_user.require(token=token)
        except CurrentUserUnauthorizedError as error:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": error.code,
                    "message": error.message,
                },
            ) from error
