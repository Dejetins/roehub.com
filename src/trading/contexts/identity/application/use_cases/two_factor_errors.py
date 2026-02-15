from __future__ import annotations


class TwoFactorOperationError(ValueError):
    """
    TwoFactorOperationError — base deterministic application error for 2FA setup/verify flows.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
    """

    def __init__(self, *, code: str, message: str, status_code: int) -> None:
        """
        Initialize stable operation error attributes for HTTP mapping.

        Args:
            code: Machine-readable deterministic error code.
            message: Human-readable deterministic message.
            status_code: HTTP status expected by inbound adapter.
        Returns:
            None.
        Assumptions:
            Status code is final and does not require additional adapter mapping logic.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def payload(self) -> dict[str, str]:
        """
        Build deterministic HTTP error payload with stable key order.

        Args:
            None.
        Returns:
            dict[str, str]: `{"error": "...", "message": "..."}` payload.
        Assumptions:
            Payload is consumed by FastAPI HTTPException `detail`.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "error": self.code,
            "message": self.message,
        }


class TwoFactorAlreadyEnabledError(TwoFactorOperationError):
    """
    TwoFactorAlreadyEnabledError — Option 1 guard blocking setup/verify when 2FA is enabled.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic 409 conflict error for already-enabled 2FA state.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Conflict status is fixed by identity 2FA policy v1.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(
            code="two_factor_already_enabled",
            message="Two-factor authentication is already enabled.",
            status_code=409,
        )


class TwoFactorSetupRequiredError(TwoFactorOperationError):
    """
    TwoFactorSetupRequiredError — verify was requested before setup stored a secret.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
      - migrations/postgres/0002_identity_2fa_totp_v1.sql
    """

    def __init__(self) -> None:
        """
        Initialize deterministic setup-required error for verify flow.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Verify endpoint should return 422 when setup row is absent.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(
            code="two_factor_setup_required",
            message="Two-factor setup must be completed first.",
            status_code=422,
        )


class TwoFactorInvalidCodeError(TwoFactorOperationError):
    """
    TwoFactorInvalidCodeError — submitted TOTP code was malformed or verification failed.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
      - src/trading/contexts/identity/adapters/outbound/security/two_factor/pyotp_totp_provider.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic invalid-code error for verify flow.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Invalid codes are user-input errors mapped to HTTP 422.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(
            code="invalid_two_factor_code",
            message="Invalid two-factor authentication code.",
            status_code=422,
        )
