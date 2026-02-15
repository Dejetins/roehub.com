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
    TwoFactorInvalidCodeError,
    TwoFactorSetupRequiredError,
)
from trading.shared_kernel.primitives import UserId

_TOTP_CODE_DIGITS = 6


@dataclass(frozen=True, slots=True)
class VerifyTwoFactorTotpResult:
    """
    VerifyTwoFactorTotpResult — output model for identity `/2fa/verify` flow.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
      - migrations/postgres/0002_identity_2fa_totp_v1.sql
    """

    enabled: bool

    def __post_init__(self) -> None:
        """
        Validate that verify result always reflects enabled 2FA state.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Successful verify flow always flips `enabled=true`.
        Raises:
            ValueError: If result is created with `enabled=False`.
        Side Effects:
            None.
        """
        if not self.enabled:
            raise ValueError("VerifyTwoFactorTotpResult.enabled must be true")


class VerifyTwoFactorTotpUseCase:
    """
    VerifyTwoFactorTotpUseCase — verify TOTP code and enable 2FA.

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
    ) -> None:
        """
        Initialize verify use-case dependencies.

        Args:
            repository: 2FA persistence port.
            secret_cipher: Secret decryption port.
            totp_provider: TOTP verification provider.
            clock: UTC time source for verification and timestamps.
        Returns:
            None.
        Assumptions:
            Dependencies are initialized and non-null.
        Raises:
            ValueError: If required dependency is missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("VerifyTwoFactorTotpUseCase requires repository")
        if secret_cipher is None:  # type: ignore[truthy-bool]
            raise ValueError("VerifyTwoFactorTotpUseCase requires secret_cipher")
        if totp_provider is None:  # type: ignore[truthy-bool]
            raise ValueError("VerifyTwoFactorTotpUseCase requires totp_provider")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("VerifyTwoFactorTotpUseCase requires clock")

        self._repository = repository
        self._secret_cipher = secret_cipher
        self._totp_provider = totp_provider
        self._clock = clock

    def verify(self, *, user_id: UserId, code: str) -> VerifyTwoFactorTotpResult:
        """
        Validate submitted code against decrypted secret and enable 2FA on success.

        Args:
            user_id: Authenticated identity user id.
            code: User-submitted TOTP verification code.
        Returns:
            VerifyTwoFactorTotpResult: Successful enablement marker.
        Assumptions:
            Option 1 policy forbids verify attempts when 2FA already enabled.
        Raises:
            TwoFactorSetupRequiredError: If setup record is missing.
            TwoFactorAlreadyEnabledError: If user already enabled 2FA.
            TwoFactorInvalidCodeError: If code is malformed or verification fails.
        Side Effects:
            Reads encrypted secret, decrypts it in-memory, and updates persisted state.
        """
        normalized_code = _normalize_totp_code(code=code)
        state = self._repository.find_by_user_id(user_id=user_id)
        if state is None:
            raise TwoFactorSetupRequiredError()
        if state.enabled:
            raise TwoFactorAlreadyEnabledError()

        now = _ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
        plaintext_secret = self._secret_cipher.decrypt_secret(secret_enc=state.totp_secret_enc)
        if not self._totp_provider.verify_code(
            secret=plaintext_secret,
            code=normalized_code,
            at_time=now,
        ):
            raise TwoFactorInvalidCodeError()

        self._repository.enable(
            user_id=user_id,
            enabled_at=now,
            updated_at=now,
        )
        return VerifyTwoFactorTotpResult(enabled=True)


def _normalize_totp_code(*, code: str) -> str:
    """
    Normalize and validate TOTP code format for deterministic verification behavior.

    Args:
        code: Raw code from API payload.
    Returns:
        str: Normalized six-digit numeric code.
    Assumptions:
        Identity 2FA v1 uses six-digit TOTP codes.
    Raises:
        TwoFactorInvalidCodeError: If code is empty, not numeric, or wrong length.
    Side Effects:
        None.
    """
    normalized = code.strip()
    if len(normalized) != _TOTP_CODE_DIGITS:
        raise TwoFactorInvalidCodeError()
    if not normalized.isdigit():
        raise TwoFactorInvalidCodeError()
    return normalized


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
