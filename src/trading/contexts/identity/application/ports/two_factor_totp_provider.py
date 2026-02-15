from __future__ import annotations

from datetime import datetime
from typing import Protocol

from trading.shared_kernel.primitives import UserId


class TwoFactorTotpProvider(Protocol):
    """
    TwoFactorTotpProvider — порт операций RFC 6238 TOTP для identity 2FA v1.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
      - src/trading/contexts/identity/adapters/outbound/security/two_factor/pyotp_totp_provider.py
    """

    def create_secret(self) -> str:
        """
        Generate new TOTP base32 secret for setup flow.

        Args:
            None.
        Returns:
            str: New base32 secret.
        Assumptions:
            Secret must have enough entropy for TOTP security.
        Raises:
            ValueError: If provider cannot generate valid secret.
        Side Effects:
            Uses cryptographically secure random source.
        """
        ...

    def build_otpauth_uri(self, *, secret: str, user_id: UserId, issuer: str) -> str:
        """
        Build standard otpauth URI for UI QR rendering.

        Args:
            secret: Base32 TOTP secret.
            user_id: Stable user identifier used as account label.
            issuer: Issuer label shown in authenticator apps.
        Returns:
            str: URI starting with `otpauth://totp`.
        Assumptions:
            URI string is returned to UI but should never be logged server-side.
        Raises:
            ValueError: If secret or metadata is invalid.
        Side Effects:
            None.
        """
        ...

    def verify_code(self, *, secret: str, code: str, at_time: datetime) -> bool:
        """
        Verify submitted TOTP code against plaintext secret at given UTC time.

        Args:
            secret: Base32 TOTP secret in plaintext form.
            code: User-provided code.
            at_time: Current UTC timestamp for verification.
        Returns:
            bool: `True` when code is valid, `False` otherwise.
        Assumptions:
            Caller controls code format normalization and UTC timestamp validity.
        Raises:
            ValueError: If provider receives malformed arguments.
        Side Effects:
            None.
        """
        ...
