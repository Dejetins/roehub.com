from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trading.shared_kernel.primitives import PaidLevel, UserId


@dataclass(frozen=True, slots=True)
class CurrentUserPrincipal:
    """
    CurrentUserPrincipal — стабильный user context для protected API endpoints.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
      - src/trading/contexts/identity/adapters/outbound/security/current_user/
        jwt_cookie_current_user.py
      - apps/api/routes/identity.py
    """

    user_id: UserId
    paid_level: PaidLevel


class CurrentUserUnauthorizedError(ValueError):
    """
    CurrentUserUnauthorizedError — детерминированная ошибка авторизации CurrentUser.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/current_user.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
      - src/trading/contexts/identity/adapters/outbound/security/current_user/
        jwt_cookie_current_user.py
    """

    def __init__(self, *, code: str, message: str) -> None:
        """
        Initialize authorization error with stable code and message.

        Args:
            code: Machine-readable deterministic error code.
            message: Human-readable deterministic description.
        Returns:
            None.
        Assumptions:
            Error code is consumed by API layer in 401 payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(message)
        self.code = code
        self.message = message


class CurrentUser(Protocol):
    """
    CurrentUser — порт извлечения `CurrentUserPrincipal` из JWT cookie.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/current_user.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
      - src/trading/contexts/identity/adapters/outbound/security/current_user/
        jwt_cookie_current_user.py
    """

    def require(self, *, token: str | None) -> CurrentUserPrincipal:
        """
        Resolve authenticated user principal or raise unauthorized error.

        Args:
            token: JWT token from HttpOnly cookie; may be missing.
        Returns:
            CurrentUserPrincipal: Authenticated user context.
        Assumptions:
            Token is signed by identity JWT key and not expired.
        Raises:
            CurrentUserUnauthorizedError: If token is missing, invalid, or user is unavailable.
        Side Effects:
            None.
        """
        ...
