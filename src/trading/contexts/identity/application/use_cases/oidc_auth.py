from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from trading.contexts.identity.application.ports import (
    AuthenticationProvider,
    AuthenticationProviderError,
    IdentityClock,
    IdentitySession,
    OidcIdentityRepository,
    OidcIdentityRepositoryError,
    SessionRepository,
)
from trading.shared_kernel.primitives import UserId

_ATTEMPT_TTL = timedelta(minutes=10)


class OidcAuthenticationError(ValueError):
    """Sanitized OIDC failure that never carries provider payloads or credentials."""

    def __init__(self, *, code: str, provider_unavailable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.provider_unavailable = provider_unavailable


@dataclass(frozen=True, slots=True)
class OidcAuthorizationStart:
    attempt_id: UUID
    state: str
    authorization_url: str


@dataclass(frozen=True, slots=True)
class OidcAuthenticationResult:
    next_path: str
    session: IdentitySession | None
    linked: bool
    provisioned: bool
    accepted_invitation_count: int


class OidcAuthenticationService:
    """Provider-neutral OIDC authorization-code orchestration with invitation-first provisioning."""

    def __init__(
        self,
        *,
        provider: AuthenticationProvider,
        repository: OidcIdentityRepository,
        session_repository: SessionRepository,
        clock: IdentityClock,
        session_idle_ttl_seconds: int,
        session_absolute_ttl_seconds: int,
    ) -> None:
        if session_idle_ttl_seconds <= 0:
            raise ValueError("OIDC idle session TTL must be positive")
        if session_absolute_ttl_seconds < session_idle_ttl_seconds:
            raise ValueError("OIDC absolute session TTL must cover idle TTL")
        self._provider = provider
        self._repository = repository
        self._sessions = session_repository
        self._clock = clock
        self._idle_ttl = session_idle_ttl_seconds
        self._absolute_ttl = session_absolute_ttl_seconds

    @property
    def provider_id(self) -> str:
        return self._provider.provider_id

    @property
    def provider_display_name(self) -> str:
        return self._provider.display_name

    def begin_login(self, *, next_path: str) -> OidcAuthorizationStart:
        return self._begin(purpose="login", linking_user_id=None, next_path=next_path)

    def begin_link(
        self, *, user_id: UserId, next_path: str
    ) -> OidcAuthorizationStart:
        return self._begin(purpose="link", linking_user_id=user_id, next_path=next_path)

    def complete(
        self,
        *,
        attempt_id: UUID,
        state: str,
        code: str,
        callback_user_id: UserId | None,
    ) -> OidcAuthenticationResult:
        now = self._clock.now()
        attempt = self._repository.find_attempt(attempt_id=attempt_id, now=now)
        if attempt is None:
            raise OidcAuthenticationError(code="oidc_attempt_invalid")
        if (
            attempt.provider_id != self._provider.provider_id
            or attempt.issuer != self._provider.issuer
        ):
            self._reject(attempt_id=attempt_id, reason_code="provider_binding_mismatch", now=now)
            raise OidcAuthenticationError(code="oidc_attempt_invalid")
        if not hmac.compare_digest(attempt.state_sha256, _sha256_text(state)):
            self._reject(attempt_id=attempt_id, reason_code="state_mismatch", now=now)
            raise OidcAuthenticationError(code="oidc_state_mismatch")
        if attempt.purpose == "link" and callback_user_id != attempt.linking_user_id:
            self._reject(attempt_id=attempt_id, reason_code="link_session_mismatch", now=now)
            raise OidcAuthenticationError(code="oidc_link_session_required")

        claimed_attempt = self._repository.claim_attempt(
            attempt_id=attempt_id,
            claimed_at=now,
        )
        if claimed_attempt is None:
            raise OidcAuthenticationError(code="oidc_attempt_invalid")
        attempt = claimed_attempt

        try:
            external = self._provider.exchange_code(
                code=code,
                code_verifier=attempt.code_verifier,
                expected_nonce_sha256=attempt.nonce_sha256,
            )
        except AuthenticationProviderError as error:
            self._reject(
                attempt_id=attempt_id,
                reason_code=error.code,
                now=self._clock.now(),
            )
            raise OidcAuthenticationError(
                code="oidc_provider_unavailable"
                if error.retryable or error.token_result_unknown
                else "oidc_authentication_failed",
                provider_unavailable=error.retryable or error.token_result_unknown,
            ) from error

        completed_at = self._clock.now()
        if external.issuer != attempt.issuer:
            self._reject(
                attempt_id=attempt_id,
                reason_code="issuer_mismatch",
                now=completed_at,
            )
            raise OidcAuthenticationError(code="oidc_authentication_failed")
        try:
            completion = self._repository.complete_attempt(
                attempt_id=attempt_id,
                provider_id=attempt.provider_id,
                issuer=attempt.issuer,
                subject_sha256=_sha256_text(external.subject),
                email_sha256=(
                    _sha256_text(external.email.strip().lower()) if external.email else None
                ),
                email_verified=external.email_verified,
                callback_user_id=callback_user_id,
                completed_at=completed_at,
            )
        except OidcIdentityRepositoryError as error:
            self._reject(
                attempt_id=attempt_id,
                reason_code=error.code,
                now=completed_at,
            )
            raise OidcAuthenticationError(code=error.code) from error

        session: IdentitySession | None = None
        if attempt.purpose == "login":
            session = self._sessions.create_session(
                user_id=completion.user_id,
                now=completed_at,
                idle_ttl_seconds=self._idle_ttl,
                absolute_ttl_seconds=self._absolute_ttl,
            )
        return OidcAuthenticationResult(
            next_path=attempt.next_path,
            session=session,
            linked=completion.linked,
            provisioned=completion.provisioned,
            accepted_invitation_count=completion.accepted_invitation_count,
        )

    def cancel(self, *, attempt_id: UUID, reason_code: str = "provider_rejected") -> None:
        self._reject(
            attempt_id=attempt_id,
            reason_code=reason_code,
            now=self._clock.now(),
        )

    def _begin(
        self,
        *,
        purpose: str,
        linking_user_id: UserId | None,
        next_path: str,
    ) -> OidcAuthorizationStart:
        now = self._clock.now()
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        attempt = self._repository.create_attempt(
            provider_id=self._provider.provider_id,
            issuer=self._provider.issuer,
            purpose="link" if purpose == "link" else "login",
            state_sha256=_sha256_text(state),
            nonce_sha256=_sha256_text(nonce),
            code_verifier=verifier,
            linking_user_id=linking_user_id,
            next_path=_safe_next_path(next_path),
            created_at=now,
            expires_at=now + _ATTEMPT_TTL,
        )
        try:
            url = self._provider.authorization_url(
                state=state,
                nonce=nonce,
                code_challenge=_pkce_challenge(verifier),
            )
        except AuthenticationProviderError as error:
            self._reject(attempt_id=attempt.attempt_id, reason_code=error.code, now=now)
            raise OidcAuthenticationError(
                code="oidc_provider_unavailable", provider_unavailable=True
            ) from error
        return OidcAuthorizationStart(
            attempt_id=attempt.attempt_id,
            state=state,
            authorization_url=url,
        )

    def _reject(self, *, attempt_id: UUID, reason_code: str, now: datetime) -> None:
        self._repository.reject_attempt(
            attempt_id=attempt_id,
            reason_code=reason_code,
            rejected_at=now,
        )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _safe_next_path(value: str) -> str:
    normalized = value.strip()
    if not normalized.startswith("/") or normalized.startswith("//") or "\\" in normalized:
        return "/"
    return normalized[:1024]
