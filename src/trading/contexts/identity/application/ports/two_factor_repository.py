from __future__ import annotations

from datetime import datetime
from typing import Protocol

from trading.contexts.identity.domain.entities import TwoFactorAuth
from trading.shared_kernel.primitives import UserId


class TwoFactorRepository(Protocol):
    """
    TwoFactorRepository — порт хранения 2FA состояния identity TOTP policy v1.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/domain/entities/two_factor_auth.py
      - src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/
        two_factor_repository.py
    """

    def find_by_user_id(self, *, user_id: UserId) -> TwoFactorAuth | None:
        """
        Find 2FA state snapshot by stable user identifier.

        Args:
            user_id: Identity user identifier.
        Returns:
            TwoFactorAuth | None: Stored state snapshot or `None` when setup is absent.
        Assumptions:
            `user_id` uniquely identifies one 2FA row.
        Raises:
            ValueError: If adapter cannot map storage row to domain state.
        Side Effects:
            Reads one storage record.
        """
        ...

    def upsert_pending_secret(
        self,
        *,
        user_id: UserId,
        totp_secret_enc: bytes,
        updated_at: datetime,
    ) -> TwoFactorAuth:
        """
        Create or replace pending TOTP secret while 2FA is not enabled.

        Args:
            user_id: Identity user identifier.
            totp_secret_enc: Encrypted opaque TOTP secret blob.
            updated_at: UTC timestamp of this write operation.
        Returns:
            TwoFactorAuth: Persisted 2FA state after upsert.
        Assumptions:
            Caller enforces policy that enabled 2FA cannot be reset in v1.
        Raises:
            ValueError: If adapter cannot persist or map the resulting state.
        Side Effects:
            Writes one storage record.
        """
        ...

    def enable(
        self,
        *,
        user_id: UserId,
        enabled_at: datetime,
        updated_at: datetime,
    ) -> TwoFactorAuth:
        """
        Mark existing 2FA setup as enabled and persist enable timestamps.

        Args:
            user_id: Identity user identifier.
            enabled_at: UTC timestamp when 2FA becomes enabled.
            updated_at: UTC timestamp of this write operation.
        Returns:
            TwoFactorAuth: Persisted enabled 2FA state.
        Assumptions:
            A pending encrypted secret exists for the target user.
        Raises:
            ValueError: If setup does not exist or adapter cannot map resulting state.
        Side Effects:
            Writes one storage record.
        """
        ...
