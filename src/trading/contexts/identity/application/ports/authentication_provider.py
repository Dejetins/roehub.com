from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId

OidcAttemptPurpose = Literal["login", "link"]


class AuthenticationProviderError(RuntimeError):
    """Sanitized provider failure safe for bounded application handling."""

    def __init__(
        self,
        *,
        code: str,
        retryable: bool = False,
        token_result_unknown: bool = False,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.token_result_unknown = token_result_unknown


@dataclass(frozen=True, slots=True)
class VerifiedExternalIdentity:
    """Claims accepted only after issuer, signature, audience, time and nonce validation."""

    issuer: str
    subject: str
    email: str | None
    email_verified: bool


class AuthenticationProvider(Protocol):
    """`AuthenticationProvider/v1` outbound port for one configured OIDC provider."""

    @property
    def provider_id(self) -> str: ...

    @property
    def issuer(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    def authorization_url(
        self,
        *,
        state: str,
        nonce: str,
        code_challenge: str,
    ) -> str: ...

    def exchange_code(
        self,
        *,
        code: str,
        code_verifier: str,
        expected_nonce_sha256: str,
    ) -> VerifiedExternalIdentity: ...


@dataclass(frozen=True, slots=True)
class OidcLoginAttempt:
    attempt_id: UUID
    provider_id: str
    issuer: str
    purpose: OidcAttemptPurpose
    state_sha256: str
    nonce_sha256: str
    code_verifier: str
    linking_user_id: UserId | None
    next_path: str
    created_at: datetime
    expires_at: datetime
    exchange_started_at: datetime | None = None
    consumed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class OidcIdentityCompletion:
    user_id: UserId
    provisioned: bool
    linked: bool
    accepted_invitation_count: int


class OidcIdentityRepositoryError(RuntimeError):
    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


class OidcIdentityRepository(Protocol):
    """Persistence port for one-time OIDC attempts and stable external identity links."""

    def create_attempt(
        self,
        *,
        provider_id: str,
        issuer: str,
        purpose: OidcAttemptPurpose,
        state_sha256: str,
        nonce_sha256: str,
        code_verifier: str,
        linking_user_id: UserId | None,
        next_path: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> OidcLoginAttempt: ...

    def find_attempt(self, *, attempt_id: UUID, now: datetime) -> OidcLoginAttempt | None: ...

    def claim_attempt(
        self, *, attempt_id: UUID, claimed_at: datetime
    ) -> OidcLoginAttempt | None: ...

    def reject_attempt(
        self,
        *,
        attempt_id: UUID,
        reason_code: str,
        rejected_at: datetime,
    ) -> None: ...

    def complete_attempt(
        self,
        *,
        attempt_id: UUID,
        provider_id: str,
        issuer: str,
        subject_sha256: str,
        email_sha256: str | None,
        email_verified: bool,
        callback_user_id: UserId | None,
        completed_at: datetime,
    ) -> OidcIdentityCompletion: ...
