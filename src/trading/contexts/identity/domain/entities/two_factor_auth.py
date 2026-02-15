from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class TwoFactorAuth:
    """
    TwoFactorAuth — immutable 2FA state snapshot for identity TOTP policy v1.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/two_factor_repository.py
      - migrations/postgres/0002_identity_2fa_totp_v1.sql
      - src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
    """

    user_id: UserId
    totp_secret_enc: bytes
    enabled: bool
    enabled_at: datetime | None
    updated_at: datetime

    def __post_init__(self) -> None:
        """
        Validate 2FA state invariants for enabled flag and UTC timestamps.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `updated_at` and `enabled_at` (when present) are timezone-aware UTC datetimes.
        Raises:
            ValueError: If encrypted secret is empty, timestamps are non-UTC, or enabled
                invariants are violated.
        Side Effects:
            None.
        """
        if not self.totp_secret_enc:
            raise ValueError("TwoFactorAuth.totp_secret_enc must be non-empty")
        _ensure_utc_datetime(name="updated_at", value=self.updated_at)
        if self.enabled:
            if self.enabled_at is None:
                raise ValueError("TwoFactorAuth.enabled_at must be set when enabled is true")
            _ensure_utc_datetime(name="enabled_at", value=self.enabled_at)
            if self.updated_at < self.enabled_at:
                raise ValueError("TwoFactorAuth.updated_at cannot be before enabled_at")
            return
        if self.enabled_at is not None:
            raise ValueError("TwoFactorAuth.enabled_at must be None when enabled is false")


def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone awareness and UTC offset for datetime fields.

    Args:
        name: Field name for deterministic error messages.
        value: Datetime value to validate.
    Returns:
        None.
    Assumptions:
        UTC datetimes are represented with timezone info and zero offset.
    Raises:
        ValueError: If datetime is naive or not in UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{name} must be UTC datetime")
