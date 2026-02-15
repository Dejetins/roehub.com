from __future__ import annotations

from trading.contexts.identity.application.ports.two_factor_policy_gate import (
    TwoFactorPolicyGate,
    TwoFactorRequiredError,
)
from trading.contexts.identity.application.ports.two_factor_repository import TwoFactorRepository
from trading.shared_kernel.primitives import UserId


class RepositoryTwoFactorPolicyGate(TwoFactorPolicyGate):
    """
    RepositoryTwoFactorPolicyGate — policy gate enforcing enabled 2FA via repository state.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/two_factor_policy_gate.py
      - src/trading/contexts/identity/application/ports/two_factor_repository.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/two_factor_enabled.py
    """

    def __init__(self, *, repository: TwoFactorRepository) -> None:
        """
        Initialize gate with repository dependency for 2FA state checks.

        Args:
            repository: 2FA repository port.
        Returns:
            None.
        Assumptions:
            Repository returns deterministic snapshots for given `user_id`.
        Raises:
            ValueError: If repository dependency is missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RepositoryTwoFactorPolicyGate requires repository")
        self._repository = repository

    def require_enabled(self, *, user_id: UserId) -> None:
        """
        Enforce that user has enabled 2FA and raise deterministic gate error otherwise.

        Args:
            user_id: Stable identity user identifier.
        Returns:
            None.
        Assumptions:
            Missing row is treated as 2FA disabled per policy.
        Raises:
            TwoFactorRequiredError: If user has no 2FA row or 2FA is not enabled.
        Side Effects:
            Reads one 2FA state snapshot from repository.
        """
        state = self._repository.find_by_user_id(user_id=user_id)
        if state is None or not state.enabled:
            raise TwoFactorRequiredError()
