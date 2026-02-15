from __future__ import annotations

from typing import Protocol

from trading.shared_kernel.primitives import UserId

_TWO_FACTOR_REQUIRED_CODE = "two_factor_required"
_TWO_FACTOR_REQUIRED_MESSAGE = "Two-factor authentication must be enabled."


class TwoFactorRequiredError(PermissionError):
    """
    TwoFactorRequiredError — детерминированный отказ policy gate при выключенной 2FA.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/two_factor_policy_gate.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/two_factor_enabled.py
      - src/trading/contexts/identity/adapters/outbound/policy/two_factor_policy_gate.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic 2FA-required error payload fields.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Error payload is reused directly in HTTP 403 responses.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(_TWO_FACTOR_REQUIRED_MESSAGE)
        self.code = _TWO_FACTOR_REQUIRED_CODE
        self.message = _TWO_FACTOR_REQUIRED_MESSAGE

    def payload(self) -> dict[str, str]:
        """
        Return deterministic policy error payload with stable key order.

        Args:
            None.
        Returns:
            dict[str, str]: `{"error": "...", "message": "..."}` payload.
        Assumptions:
            Keys order remains deterministic for API responses.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "error": self.code,
            "message": self.message,
        }


class TwoFactorPolicyGate(Protocol):
    """
    TwoFactorPolicyGate — порт проверки policy `exchange keys require 2FA`.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/adapters/outbound/policy/two_factor_policy_gate.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/two_factor_enabled.py
      - apps/api/wiring/modules/identity.py
    """

    def require_enabled(self, *, user_id: UserId) -> None:
        """
        Enforce that user has enabled 2FA, otherwise raise deterministic error.

        Args:
            user_id: Stable identity user identifier.
        Returns:
            None.
        Assumptions:
            Gate is reused by future exchange-keys endpoints.
        Raises:
            TwoFactorRequiredError: If 2FA is not enabled for user.
        Side Effects:
            None.
        """
        ...
