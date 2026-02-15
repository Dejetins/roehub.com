from typing import cast

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse

from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.application.ports.two_factor_policy_gate import (
    TwoFactorPolicyGate,
    TwoFactorRequiredError,
)


class TwoFactorRequiredHttpError(PermissionError):
    """
    TwoFactorRequiredHttpError — HTTP-facing deterministic 403 error for 2FA policy gate.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/two_factor_policy_gate.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/two_factor_enabled.py
      - apps/api/main/app.py
    """

    def __init__(self, *, code: str, message: str) -> None:
        """
        Initialize HTTP 403 error payload fields with deterministic values.

        Args:
            code: Machine-readable deterministic error code.
            message: Human-readable deterministic error message.
        Returns:
            None.
        Assumptions:
            Handler maps this error to exact top-level JSON payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(message)
        self.code = code
        self.message = message

    def payload(self) -> dict[str, str]:
        """
        Build deterministic 403 payload with stable key order.

        Args:
            None.
        Returns:
            dict[str, str]: `{"error": "...", "message": "..."}` payload.
        Assumptions:
            Payload keys order remains deterministic across responses.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "error": self.code,
            "message": self.message,
        }


def two_factor_required_http_error_handler(
    _request: Request,
    error: Exception,
) -> JSONResponse:
    """
    Map `TwoFactorRequiredHttpError` to exact 403 JSON payload for API responses.

    Args:
        _request: Starlette request object (unused).
        error: Deterministic 2FA-required HTTP error.
    Returns:
        JSONResponse: HTTP 403 with exact payload shape.
    Assumptions:
        Payload must remain exactly `{"error":"two_factor_required","message":"..."}`.
    Raises:
        None.
    Side Effects:
        None.
    """
    typed_error = cast(TwoFactorRequiredHttpError, error)
    return JSONResponse(
        status_code=403,
        content=typed_error.payload(),
    )


def register_two_factor_required_exception_handler(*, app: FastAPI) -> None:
    """
    Register deterministic 2FA-required exception handler on FastAPI app instance.

    Args:
        app: FastAPI application where the handler should be installed.
    Returns:
        None.
    Assumptions:
        Handler is idempotent and may be registered once during app startup.
    Raises:
        ValueError: If app reference is missing.
    Side Effects:
        Updates app-level exception handler registry.
    """
    if app is None:  # type: ignore[truthy-bool]
        raise ValueError("register_two_factor_required_exception_handler requires app")
    app.add_exception_handler(
        TwoFactorRequiredHttpError,
        two_factor_required_http_error_handler,
    )


class RequireTwoFactorEnabledDependency:
    """
    RequireTwoFactorEnabledDependency — reusable FastAPI dependency enforcing enabled 2FA.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/two_factor_policy_gate.py
      - src/trading/contexts/identity/adapters/outbound/policy/two_factor_policy_gate.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
    """

    def __init__(
        self,
        *,
        current_user_dependency: RequireCurrentUserDependency,
        policy_gate: TwoFactorPolicyGate,
    ) -> None:
        """
        Initialize dependency with authenticated principal resolver and policy gate.

        Args:
            current_user_dependency: Dependency resolving authenticated current user.
            policy_gate: Gate checking whether user has enabled 2FA.
        Returns:
            None.
        Assumptions:
            `current_user_dependency` raises deterministic 401 errors when unauthorized.
        Raises:
            ValueError: If dependency arguments are missing.
        Side Effects:
            None.
        """
        if current_user_dependency is None:  # type: ignore[truthy-bool]
            raise ValueError("RequireTwoFactorEnabledDependency requires current_user_dependency")
        if policy_gate is None:  # type: ignore[truthy-bool]
            raise ValueError("RequireTwoFactorEnabledDependency requires policy_gate")
        self._current_user_dependency = current_user_dependency
        self._policy_gate = policy_gate

    def __call__(self, request: Request) -> CurrentUserPrincipal:
        """
        Resolve authenticated principal and enforce enabled 2FA policy.

        Args:
            request: FastAPI request object containing authentication cookie.
        Returns:
            CurrentUserPrincipal: Authenticated principal with guaranteed enabled 2FA.
        Assumptions:
            Request carries JWT cookie used by current user dependency.
        Raises:
            TwoFactorRequiredHttpError: If 2FA policy gate rejects request.
            HTTPException: 401 errors propagated from current-user dependency.
        Side Effects:
            None.
        """
        principal = self._current_user_dependency(request)
        try:
            self._policy_gate.require_enabled(user_id=principal.user_id)
        except TwoFactorRequiredError as error:
            raise TwoFactorRequiredHttpError(
                code=error.code,
                message=error.message,
            ) from error
        return principal
