from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.application.use_cases import (
    SetupTwoFactorTotpUseCase,
    TwoFactorOperationError,
    VerifyTwoFactorTotpUseCase,
)


class TwoFactorSetupResponse(BaseModel):
    """
    TwoFactorSetupResponse — API response payload for identity `POST /2fa/setup`.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
      - apps/api/routes/identity.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
    """

    otpauth_uri: str


class TwoFactorVerifyRequest(BaseModel):
    """
    TwoFactorVerifyRequest — API request payload for identity `POST /2fa/verify`.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
      - apps/api/routes/identity.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
    """

    code: str


class TwoFactorVerifyResponse(BaseModel):
    """
    TwoFactorVerifyResponse — API response payload for successful 2FA enable flow.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
      - apps/api/routes/identity.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
    """

    enabled: bool


def build_two_factor_totp_router(
    *,
    setup_use_case: SetupTwoFactorTotpUseCase,
    verify_use_case: VerifyTwoFactorTotpUseCase,
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    """
    Build router exposing identity 2FA TOTP setup and verify endpoints.

    Args:
        setup_use_case: 2FA setup use-case dependency.
        verify_use_case: 2FA verify use-case dependency.
        current_user_dependency: Auth dependency for current user principal.
    Returns:
        APIRouter: Configured router with `/2fa/setup` and `/2fa/verify`.
    Assumptions:
        Current user dependency enforces authenticated JWT cookie access.
    Raises:
        ValueError: If required dependencies are missing.
    Side Effects:
        None.
    """
    if setup_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_two_factor_totp_router requires setup_use_case")
    if verify_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_two_factor_totp_router requires verify_use_case")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_two_factor_totp_router requires current_user_dependency")

    router = APIRouter(tags=["identity"])

    @router.post("/2fa/setup", response_model=TwoFactorSetupResponse)
    def post_two_factor_setup(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> TwoFactorSetupResponse:
        """
        Create or refresh pending TOTP setup and return otpauth URI only.

        Args:
            principal: Authenticated current user.
        Returns:
            TwoFactorSetupResponse: Otpauth URI for UI QR generation.
        Assumptions:
            Option 1 policy rejects setup when 2FA already enabled.
        Raises:
            HTTPException: Deterministic 4xx payload on policy/input errors.
        Side Effects:
            Persists encrypted TOTP secret in 2FA repository.
        """
        try:
            result = setup_use_case.setup(user_id=principal.user_id)
        except TwoFactorOperationError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.payload(),
            ) from error
        return TwoFactorSetupResponse(otpauth_uri=result.otpauth_uri)

    @router.post("/2fa/verify", response_model=TwoFactorVerifyResponse)
    def post_two_factor_verify(
        request: TwoFactorVerifyRequest,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> TwoFactorVerifyResponse:
        """
        Verify submitted TOTP code and enable 2FA for authenticated user.

        Args:
            request: Submitted 6-digit TOTP code payload.
            principal: Authenticated current user.
        Returns:
            TwoFactorVerifyResponse: Enabled marker payload.
        Assumptions:
            Setup flow was completed and encrypted secret exists in storage.
        Raises:
            HTTPException: Deterministic 4xx payload on policy/input errors.
        Side Effects:
            Updates persisted 2FA state to `enabled=true` on success.
        """
        try:
            result = verify_use_case.verify(
                user_id=principal.user_id,
                code=request.code,
            )
        except TwoFactorOperationError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.payload(),
            ) from error
        return TwoFactorVerifyResponse(enabled=result.enabled)

    return router
