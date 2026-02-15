from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading.contexts.identity.application.ports import IdentityClock
from trading.contexts.identity.application.ports.two_factor_repository import TwoFactorRepository
from trading.contexts.identity.application.ports.two_factor_secret_cipher import (
    TwoFactorSecretCipher,
)
from trading.contexts.identity.application.ports.two_factor_totp_provider import (
    TwoFactorTotpProvider,
)
from trading.contexts.identity.application.use_cases.two_factor_errors import (
    TwoFactorAlreadyEnabledError,
)
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class SetupTwoFactorTotpResult:
    """
    SetupTwoFactorTotpResult — output model for identity `/2fa/setup` flow.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    """

    otpauth_uri: str

    def __post_init__(self) -> None:
        """
        Validate that result contains standard otpauth URI expected by UI.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            UI generates QR code from returned URI and API returns no binary QR payload.
        Raises:
            ValueError: If URI is empty or does not match expected scheme prefix.
        Side Effects:
            None.
        """
        normalized = self.otpauth_uri.strip()
        if not normalized:
            raise ValueError("SetupTwoFactorTotpResult.otpauth_uri must be non-empty")
        if not normalized.startswith("otpauth://totp"):
            raise ValueError(
                "SetupTwoFactorTotpResult.otpauth_uri must start with 'otpauth://totp'"
            )


class SetupTwoFactorTotpUseCase:
    """
    SetupTwoFactorTotpUseCase — setup flow creating encrypted pending TOTP secret.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/two_factor_repository.py
      - src/trading/contexts/identity/application/ports/two_factor_secret_cipher.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
    """

    def __init__(
        self,
        *,
        repository: TwoFactorRepository,
        secret_cipher: TwoFactorSecretCipher,
        totp_provider: TwoFactorTotpProvider,
        clock: IdentityClock,
        issuer: str = "Roehub",
    ) -> None:
        """
        Initialize setup use-case dependencies and immutable issuer policy.

        Args:
            repository: 2FA persistence port.
            secret_cipher: Envelope encryption port for TOTP secret.
            totp_provider: Provider generating secrets and otpauth URI.
            clock: UTC time source for `updated_at`.
            issuer: Issuer label used in authenticator apps.
        Returns:
            None.
        Assumptions:
            All dependencies are initialized and non-null.
        Raises:
            ValueError: If dependencies are missing or issuer is empty.
        Side Effects:
            None.
        """
        normalized_issuer = issuer.strip()
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("SetupTwoFactorTotpUseCase requires repository")
        if secret_cipher is None:  # type: ignore[truthy-bool]
            raise ValueError("SetupTwoFactorTotpUseCase requires secret_cipher")
        if totp_provider is None:  # type: ignore[truthy-bool]
            raise ValueError("SetupTwoFactorTotpUseCase requires totp_provider")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("SetupTwoFactorTotpUseCase requires clock")
        if not normalized_issuer:
            raise ValueError("SetupTwoFactorTotpUseCase requires non-empty issuer")

        self._repository = repository
        self._secret_cipher = secret_cipher
        self._totp_provider = totp_provider
        self._clock = clock
        self._issuer = normalized_issuer

    def setup(self, *, user_id: UserId) -> SetupTwoFactorTotpResult:
        """
        Generate pending TOTP secret, persist encrypted blob, and return otpauth URI.

        Args:
            user_id: Authenticated identity user id.
        Returns:
            SetupTwoFactorTotpResult: Otpauth URI for UI QR rendering.
        Assumptions:
            Option 1 policy forbids setup reset once 2FA is enabled.
        Raises:
            TwoFactorAlreadyEnabledError: If user already enabled 2FA.
            ValueError: If dependencies return invalid data.
        Side Effects:
            Persists encrypted pending secret in identity 2FA storage.
        """
        existing = self._repository.find_by_user_id(user_id=user_id)
        if existing is not None and existing.enabled:
            raise TwoFactorAlreadyEnabledError()

        plaintext_secret = self._totp_provider.create_secret()
        secret_enc = self._secret_cipher.encrypt_secret(secret=plaintext_secret)
        now = _ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
        upserted = self._repository.upsert_pending_secret(
            user_id=user_id,
            totp_secret_enc=secret_enc,
            updated_at=now,
        )
        if upserted.enabled:
            raise TwoFactorAlreadyEnabledError()
        otpauth_uri = self._totp_provider.build_otpauth_uri(
            secret=plaintext_secret,
            user_id=user_id,
            issuer=self._issuer,
        )
        return SetupTwoFactorTotpResult(otpauth_uri=otpauth_uri)


def _ensure_utc_datetime(*, value: datetime, field_name: str) -> datetime:
    """
    Validate datetime is timezone-aware UTC and return same value.

    Args:
        value: Datetime value to validate.
        field_name: Field label for deterministic error message.
    Returns:
        datetime: Same validated datetime.
    Assumptions:
        UTC datetimes have zero offset.
    Raises:
        ValueError: If datetime is naive or non-UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{field_name} must be UTC datetime")
    return value
