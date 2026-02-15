from __future__ import annotations

from datetime import datetime

from trading.contexts.identity.application.ports.two_factor_repository import TwoFactorRepository
from trading.contexts.identity.domain.entities import TwoFactorAuth
from trading.shared_kernel.primitives import UserId


class InMemoryIdentityTwoFactorRepository(TwoFactorRepository):
    """
    InMemoryIdentityTwoFactorRepository — deterministic in-memory 2FA state storage.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/two_factor_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/
        two_factor_repository.py
      - tests/unit/contexts/identity/application/test_two_factor_totp_use_cases.py
    """

    def __init__(self) -> None:
        """
        Initialize empty in-memory 2FA storage.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Repository instance is process-local and isolated per test run.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._rows: dict[str, TwoFactorAuth] = {}

    def find_by_user_id(self, *, user_id: UserId) -> TwoFactorAuth | None:
        """
        Find 2FA state by stable user identifier.

        Args:
            user_id: Identity user identifier.
        Returns:
            TwoFactorAuth | None: Stored state snapshot or `None`.
        Assumptions:
            Dictionary key uses canonical string representation of `UserId`.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._rows.get(str(user_id))

    def upsert_pending_secret(
        self,
        *,
        user_id: UserId,
        totp_secret_enc: bytes,
        updated_at: datetime,
    ) -> TwoFactorAuth:
        """
        Store or replace pending encrypted secret and mark state as not enabled.

        Args:
            user_id: Identity user identifier.
            totp_secret_enc: Encrypted opaque TOTP secret blob.
            updated_at: UTC update timestamp.
        Returns:
            TwoFactorAuth: Stored state snapshot.
        Assumptions:
            Caller applies policy checks for already-enabled rows.
        Raises:
            ValueError: If resulting state violates domain invariants.
        Side Effects:
            Mutates in-memory dictionary row for the user.
        """
        row = TwoFactorAuth(
            user_id=user_id,
            totp_secret_enc=bytes(totp_secret_enc),
            enabled=False,
            enabled_at=None,
            updated_at=updated_at,
        )
        self._rows[str(user_id)] = row
        return row

    def enable(
        self,
        *,
        user_id: UserId,
        enabled_at: datetime,
        updated_at: datetime,
    ) -> TwoFactorAuth:
        """
        Mark existing setup row as enabled and persist enable timestamps.

        Args:
            user_id: Identity user identifier.
            enabled_at: UTC timestamp when 2FA became enabled.
            updated_at: UTC update timestamp.
        Returns:
            TwoFactorAuth: Updated enabled state snapshot.
        Assumptions:
            Pending setup row exists for the user.
        Raises:
            ValueError: If setup row is missing.
        Side Effects:
            Mutates in-memory dictionary row for the user.
        """
        existing = self._rows.get(str(user_id))
        if existing is None:
            raise ValueError("InMemoryIdentityTwoFactorRepository missing setup row")

        enabled_row = TwoFactorAuth(
            user_id=user_id,
            totp_secret_enc=existing.totp_secret_enc,
            enabled=True,
            enabled_at=enabled_at,
            updated_at=updated_at,
        )
        self._rows[str(user_id)] = enabled_row
        return enabled_row
