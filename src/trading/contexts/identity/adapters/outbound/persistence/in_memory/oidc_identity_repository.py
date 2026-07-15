from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from threading import RLock
from uuid import uuid4

from trading.contexts.identity.application.ports.authentication_provider import (
    OidcAttemptPurpose,
    OidcIdentityCompletion,
    OidcIdentityRepository,
    OidcIdentityRepositoryError,
    OidcLoginAttempt,
)
from trading.contexts.identity.application.ports.user_repository import UserRepository
from trading.shared_kernel.primitives import UserId

from .organization_repository import InMemoryOrganizationRepository


class InMemoryOidcIdentityRepository(OidcIdentityRepository):
    """Process-local adapter preserving the same takeover protections as PostgreSQL."""

    def __init__(
        self,
        *,
        user_repository: UserRepository,
        organization_repository: InMemoryOrganizationRepository,
    ) -> None:
        self._users = user_repository
        self._organizations = organization_repository
        self._attempts: dict[object, OidcLoginAttempt] = {}
        self._identity_user: dict[tuple[str, str, str], UserId] = {}
        self._user_provider_identity: dict[tuple[UserId, str, str], str] = {}
        self._lock = RLock()

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
    ) -> OidcLoginAttempt:
        attempt = OidcLoginAttempt(
            attempt_id=uuid4(),
            provider_id=provider_id,
            issuer=issuer,
            purpose=purpose,
            state_sha256=state_sha256,
            nonce_sha256=nonce_sha256,
            code_verifier=code_verifier,
            linking_user_id=linking_user_id,
            next_path=next_path,
            created_at=created_at,
            expires_at=expires_at,
        )
        with self._lock:
            self._attempts[attempt.attempt_id] = attempt
        return attempt

    def find_attempt(self, *, attempt_id: object, now: datetime) -> OidcLoginAttempt | None:
        attempt = self._attempts.get(attempt_id)
        if (
            attempt is None
            or attempt.exchange_started_at is not None
            or attempt.consumed_at is not None
            or attempt.expires_at <= now
        ):
            return None
        return attempt

    def claim_attempt(
        self, *, attempt_id: object, claimed_at: datetime
    ) -> OidcLoginAttempt | None:
        with self._lock:
            attempt = self._attempts.get(attempt_id)
            if (
                attempt is None
                or attempt.exchange_started_at is not None
                or attempt.consumed_at is not None
                or attempt.expires_at <= claimed_at
            ):
                return None
            claimed = replace(attempt, exchange_started_at=claimed_at)
            self._attempts[attempt_id] = claimed
            return claimed

    def reject_attempt(
        self,
        *,
        attempt_id: object,
        reason_code: str,
        rejected_at: datetime,
    ) -> None:
        del reason_code
        with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is not None and attempt.consumed_at is None:
                self._attempts[attempt_id] = replace(attempt, consumed_at=rejected_at)

    def complete_attempt(
        self,
        *,
        attempt_id: object,
        provider_id: str,
        issuer: str,
        subject_sha256: str,
        email_sha256: str | None,
        email_verified: bool,
        callback_user_id: UserId | None,
        completed_at: datetime,
    ) -> OidcIdentityCompletion:
        with self._lock:
            attempt = self.find_attempt(attempt_id=attempt_id, now=completed_at)
            if attempt is None:
                attempt = self._attempts.get(attempt_id)
            if (
                attempt is None
                or attempt.exchange_started_at is None
                or attempt.consumed_at is not None
                or attempt.expires_at <= completed_at
            ):
                raise OidcIdentityRepositoryError(code="oidc_attempt_invalid")
            if attempt.provider_id != provider_id or attempt.issuer != issuer:
                raise OidcIdentityRepositoryError(code="oidc_attempt_invalid")
            identity_key = (provider_id, issuer, subject_sha256)
            linked_user_id = self._identity_user.get(identity_key)
            if linked_user_id is not None:
                if attempt.purpose == "link" and linked_user_id != attempt.linking_user_id:
                    raise OidcIdentityRepositoryError(code="oidc_identity_conflict")
                self._users.record_local_login(user_id=linked_user_id, login_at=completed_at)
                self._attempts[attempt_id] = replace(attempt, consumed_at=completed_at)
                return OidcIdentityCompletion(
                    user_id=linked_user_id,
                    provisioned=False,
                    linked=False,
                    accepted_invitation_count=0,
                )

            if attempt.purpose == "link":
                target = attempt.linking_user_id
                if target is None or callback_user_id != target:
                    raise OidcIdentityRepositoryError(code="oidc_link_session_required")
                if self._users.find_by_user_id(user_id=target) is None:
                    raise OidcIdentityRepositoryError(code="oidc_link_session_required")
                user_key = (target, provider_id, issuer)
                if user_key in self._user_provider_identity:
                    raise OidcIdentityRepositoryError(code="oidc_identity_conflict")
                self._identity_user[identity_key] = target
                self._user_provider_identity[user_key] = subject_sha256
                self._attempts[attempt_id] = replace(attempt, consumed_at=completed_at)
                return OidcIdentityCompletion(
                    user_id=target,
                    provisioned=False,
                    linked=True,
                    accepted_invitation_count=0,
                )

            if not email_verified or email_sha256 is None:
                raise OidcIdentityRepositoryError(code="oidc_verified_email_required")
            user_id = UserId(uuid4())
            accepted = self._organizations.accept_pending_invitations(
                user_id=user_id,
                recipient_email_sha256=email_sha256,
                accepted_at=completed_at,
            )
            if accepted == 0:
                raise OidcIdentityRepositoryError(code="oidc_invitation_required")
            self._users.create_local_user(user_id=user_id, created_at=completed_at)
            self._identity_user[identity_key] = user_id
            self._user_provider_identity[(user_id, provider_id, issuer)] = subject_sha256
            self._attempts[attempt_id] = replace(attempt, consumed_at=completed_at)
            return OidcIdentityCompletion(
                user_id=user_id,
                provisioned=True,
                linked=True,
                accepted_invitation_count=accepted,
            )
