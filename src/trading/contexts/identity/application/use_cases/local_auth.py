from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, NoReturn
from uuid import UUID, uuid4

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers.exceptions import WebAuthnException
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from trading.contexts.identity.application.ports import (
    IdentityClock,
    IdentitySession,
    SessionRepository,
    UserRepository,
)
from trading.contexts.identity.application.ports.local_auth_repository import (
    LocalAccount,
    LocalAuthChallenge,
    LocalAuthPurpose,
    LocalAuthRepository,
    LocalAuthRepositoryError,
    LocalPasskey,
)
from trading.shared_kernel.primitives import UserId

_BOOTSTRAP_TTL = timedelta(minutes=15)
_CHALLENGE_TTL = timedelta(minutes=5)
_USERNAME_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{2,63}$")
_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,62}[a-z0-9]$")
_MIN_PASSWORD_LENGTH = 14
_RECOVERY_CODE_COUNT = 8


class LocalAuthError(ValueError):
    """Public local-auth failure that does not disclose account existence."""

    def __init__(self, *, code: str = "authentication_failed") -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class LocalAuthStatus:
    bootstrap_required: bool
    passkey_available: bool
    password_available: bool
    registration_open: bool = False


@dataclass(frozen=True, slots=True)
class LocalAuthOptions:
    challenge_id: UUID
    public_key: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LocalAuthResult:
    session: IdentitySession
    recovery_codes: tuple[str, ...] = ()


class LocalAuthService:
    """Passkey-first local auth with opaque sessions and one-time recovery."""

    def __init__(
        self,
        *,
        repository: LocalAuthRepository,
        user_repository: UserRepository,
        session_repository: SessionRepository,
        clock: IdentityClock,
        rp_id: str,
        rp_name: str,
        expected_origin: str,
        session_idle_ttl_seconds: int,
        session_absolute_ttl_seconds: int,
    ) -> None:
        normalized_rp_id = rp_id.strip().lower()
        normalized_rp_name = rp_name.strip()
        normalized_origin = expected_origin.strip().rstrip("/")
        if not normalized_rp_id or not normalized_rp_name or not normalized_origin:
            raise ValueError("local auth WebAuthn settings must be non-empty")
        if session_idle_ttl_seconds <= 0:
            raise ValueError("local auth idle session TTL must be positive")
        if session_absolute_ttl_seconds < session_idle_ttl_seconds:
            raise ValueError("local auth absolute session TTL must cover idle TTL")
        self._repository = repository
        self._users = user_repository
        self._sessions = session_repository
        self._clock = clock
        self._rp_id = normalized_rp_id
        self._rp_name = normalized_rp_name
        self._expected_origin = normalized_origin
        self._idle_ttl = session_idle_ttl_seconds
        self._absolute_ttl = session_absolute_ttl_seconds
        self._hasher = PasswordHasher()
        self._fallback_digest = self._hasher.hash(secrets.token_urlsafe(32))

    def status(self) -> LocalAuthStatus:
        bootstrap_required = self._repository.bootstrap_required()
        return LocalAuthStatus(
            bootstrap_required=bootstrap_required,
            passkey_available=not bootstrap_required,
            password_available=not bootstrap_required,
        )

    def issue_bootstrap_ticket(self) -> str:
        now = self._clock.now()
        one_time_value = secrets.token_urlsafe(32)
        self._repository.issue_bootstrap_ticket(
            token_sha256=_sha256_text(one_time_value),
            created_at=now,
            expires_at=now + _BOOTSTRAP_TTL,
        )
        return one_time_value

    def begin_bootstrap(
        self,
        *,
        ticket: str,
        username: str,
        display_name: str,
        installation_name: str,
        organization_slug: str,
        organization_name: str,
        password: str | None,
    ) -> LocalAuthOptions:
        now = self._clock.now()
        ticket_id = self._repository.find_bootstrap_ticket(
            token_sha256=_sha256_text(ticket.strip()), now=now
        )
        if ticket_id is None:
            raise LocalAuthError(code="bootstrap_unavailable")
        normalized_username = _normalize_username(username)
        normalized_slug = organization_slug.strip().lower()
        if not _SLUG_PATTERN.fullmatch(normalized_slug):
            raise LocalAuthError(code="invalid_bootstrap_input")
        password_digest: str | None = None
        if password is not None and password:
            if len(password) < _MIN_PASSWORD_LENGTH:
                raise LocalAuthError(code="weak_password")
            password_digest = self._hasher.hash(password)

        user_id = UserId(uuid4())
        context = {
            "bootstrap_user_id": str(user_id),
            "ticket_id": str(ticket_id),
            "username": normalized_username,
            "display_name": _bounded_text(display_name, 2, 120),
            "installation_name": _bounded_text(installation_name, 2, 120),
            "organization_slug": normalized_slug,
            "organization_name": _bounded_text(organization_name, 2, 120),
        }
        if password_digest is not None:
            context["password_hash"] = password_digest

        challenge_bytes = secrets.token_bytes(32)
        challenge = self._repository.create_challenge(
            purpose="bootstrap",
            challenge_sha256=_sha256_bytes(challenge_bytes),
            user_id=None,
            context=context,
            created_at=now,
            expires_at=now + _CHALLENGE_TTL,
        )
        options = generate_registration_options(
            rp_id=self._rp_id,
            rp_name=self._rp_name,
            user_id=user_id.value.bytes,
            user_name=normalized_username,
            user_display_name=context["display_name"],
            challenge=challenge_bytes,
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.REQUIRED,
                user_verification=UserVerificationRequirement.REQUIRED,
            ),
        )
        return _options(challenge_id=challenge.challenge_id, value=options)

    def complete_bootstrap(
        self, *, challenge_id: UUID, credential: Mapping[str, Any]
    ) -> LocalAuthResult:
        now = self._clock.now()
        challenge, challenge_bytes = self._validated_challenge(
            challenge_id=challenge_id,
            purpose="bootstrap",
            credential=credential,
        )
        try:
            verification = verify_registration_response(
                credential=dict(credential),
                expected_challenge=challenge_bytes,
                expected_rp_id=self._rp_id,
                expected_origin=self._expected_origin,
                require_user_verification=True,
            )
            user_id = _bootstrap_user_id(challenge)
            ticket_id = UUID(challenge.context["ticket_id"])
            recovery_codes = tuple(_new_recovery_code() for _ in range(_RECOVERY_CODE_COUNT))
            self._repository.complete_bootstrap(
                challenge_id=challenge_id,
                ticket_id=ticket_id,
                user_id=user_id,
                username=challenge.context["username"],
                display_name=challenge.context["display_name"],
                password_hash=challenge.context.get("password_hash"),
                installation_name=challenge.context["installation_name"],
                organization_slug=challenge.context["organization_slug"],
                organization_name=challenge.context["organization_name"],
                passkey=LocalPasskey(
                    credential_id=_credential_id(verification.credential_id),
                    user_id=user_id,
                    public_key=verification.credential_public_key,
                    sign_count=verification.sign_count,
                    transports=_credential_transports(credential),
                    created_at=now,
                ),
                recovery_code_hashes=tuple(self._hasher.hash(code) for code in recovery_codes),
                completed_at=now,
            )
        except (KeyError, ValueError, LocalAuthRepositoryError, WebAuthnException) as error:
            raise LocalAuthError(code="bootstrap_unavailable") from error
        return LocalAuthResult(
            session=self._issue_session(user_id=user_id, now=now),
            recovery_codes=recovery_codes,
        )

    def begin_passkey_login(self) -> LocalAuthOptions:
        if self._repository.bootstrap_required():
            raise LocalAuthError()
        now = self._clock.now()
        challenge_bytes = secrets.token_bytes(32)
        challenge = self._repository.create_challenge(
            purpose="login",
            challenge_sha256=_sha256_bytes(challenge_bytes),
            user_id=None,
            context={},
            created_at=now,
            expires_at=now + _CHALLENGE_TTL,
        )
        options = generate_authentication_options(
            rp_id=self._rp_id,
            challenge=challenge_bytes,
            user_verification=UserVerificationRequirement.REQUIRED,
        )
        return _options(challenge_id=challenge.challenge_id, value=options)

    def complete_passkey_login(
        self, *, challenge_id: UUID, credential: Mapping[str, Any]
    ) -> LocalAuthResult:
        return self._complete_passkey_authentication(
            challenge_id=challenge_id,
            credential=credential,
            purpose="login",
            expected_user_id=None,
        )

    def password_login(self, *, username: str, password: str) -> LocalAuthResult:
        normalized_username = _normalize_username_for_login(username)
        subject_hash = _subject_hash("password", normalized_username)
        now = self._clock.now()
        self._enforce_rate_limit(
            user_id=None,
            subject_hash=subject_hash,
            action="local_auth.password",
            now=now,
        )
        account = self._repository.find_account_by_username(username=normalized_username)
        stored_digest = (
            self._fallback_digest
            if account is None or account.password_hash is None
            else account.password_hash
        )
        try:
            credential_valid = self._hasher.verify(stored_digest, password)
        except VerificationError:
            credential_valid = False
        if account is None or account.password_hash is None or not credential_valid:
            self._record_failure(
                user_id=None if account is None else account.user_id,
                subject_hash=subject_hash,
                action="local_auth.password",
                now=now,
            )
        self._repository.clear_auth_failures(subject_sha256=subject_hash)
        self._users.record_local_login(user_id=account.user_id, login_at=now)
        self._repository.record_auth_event(
            user_id=account.user_id,
            subject_sha256=subject_hash,
            action="local_auth.password",
            outcome="succeeded",
            reason_code="completed",
            created_at=now,
        )
        return LocalAuthResult(session=self._issue_session(user_id=account.user_id, now=now))

    def recovery_login(self, *, username: str, recovery_code: str) -> LocalAuthResult:
        normalized_username = _normalize_username_for_login(username)
        subject_hash = _subject_hash("recovery", normalized_username)
        now = self._clock.now()
        self._enforce_rate_limit(
            user_id=None,
            subject_hash=subject_hash,
            action="local_auth.recovery",
            now=now,
        )
        account = self._repository.find_account_by_username(username=normalized_username)
        matching_code_id: UUID | None = None
        hashes = (
            ()
            if account is None
            else self._repository.list_recovery_code_hashes(user_id=account.user_id)
        )
        for stored in hashes:
            try:
                if self._hasher.verify(stored.code_hash, recovery_code):
                    matching_code_id = stored.recovery_code_id
                    break
            except VerificationError:
                continue
        if account is None:
            try:
                self._hasher.verify(self._fallback_digest, recovery_code)
            except VerificationError:
                pass
        if account is None or matching_code_id is None:
            self._record_failure(
                user_id=None if account is None else account.user_id,
                subject_hash=subject_hash,
                action="local_auth.recovery",
                now=now,
            )
        if not self._repository.consume_recovery_code(
            recovery_code_id=matching_code_id,
            user_id=account.user_id,
            consumed_at=now,
        ):
            self._reject(
                user_id=account.user_id,
                subject_hash=subject_hash,
                action="local_auth.recovery",
                reason="already_consumed",
                now=now,
            )
        self._sessions.revoke_user_sessions(user_id=account.user_id, revoked_at=now)
        self._users.record_local_login(user_id=account.user_id, login_at=now)
        self._repository.clear_auth_failures(subject_sha256=subject_hash)
        self._repository.record_auth_event(
            user_id=account.user_id,
            subject_sha256=subject_hash,
            action="local_auth.recovery",
            outcome="succeeded",
            reason_code="completed",
            created_at=now,
        )
        return LocalAuthResult(session=self._issue_session(user_id=account.user_id, now=now))

    def begin_passkey_registration(self, *, user_id: UserId) -> LocalAuthOptions:
        now = self._clock.now()
        account = self._require_account(user_id=user_id)
        existing = self._repository.list_passkeys(user_id=user_id)
        challenge_bytes = secrets.token_bytes(32)
        challenge = self._repository.create_challenge(
            purpose="register",
            challenge_sha256=_sha256_bytes(challenge_bytes),
            user_id=user_id,
            context={},
            created_at=now,
            expires_at=now + _CHALLENGE_TTL,
        )
        options = generate_registration_options(
            rp_id=self._rp_id,
            rp_name=self._rp_name,
            user_id=user_id.value.bytes,
            user_name=account.username,
            user_display_name=account.display_name,
            challenge=challenge_bytes,
            exclude_credentials=[
                PublicKeyCredentialDescriptor(id=_decode_base64url(item.credential_id))
                for item in existing
            ],
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.REQUIRED,
                user_verification=UserVerificationRequirement.REQUIRED,
            ),
        )
        return _options(challenge_id=challenge.challenge_id, value=options)

    def complete_passkey_registration(
        self,
        *,
        challenge_id: UUID,
        user_id: UserId,
        credential: Mapping[str, Any],
    ) -> None:
        now = self._clock.now()
        challenge, challenge_bytes = self._validated_challenge(
            challenge_id=challenge_id,
            purpose="register",
            credential=credential,
        )
        if challenge.user_id != user_id:
            raise LocalAuthError()
        try:
            verification = verify_registration_response(
                credential=dict(credential),
                expected_challenge=challenge_bytes,
                expected_rp_id=self._rp_id,
                expected_origin=self._expected_origin,
                require_user_verification=True,
            )
            self._repository.add_passkey_and_consume_challenge(
                challenge_id=challenge_id,
                passkey=LocalPasskey(
                    credential_id=_credential_id(verification.credential_id),
                    user_id=user_id,
                    public_key=verification.credential_public_key,
                    sign_count=verification.sign_count,
                    transports=_credential_transports(credential),
                    created_at=now,
                ),
                completed_at=now,
            )
        except (LocalAuthRepositoryError, WebAuthnException) as error:
            raise LocalAuthError() from error
        self._repository.record_auth_event(
            user_id=user_id,
            subject_sha256=_subject_hash("passkey", str(user_id)),
            action="local_auth.passkey_registered",
            outcome="succeeded",
            reason_code="completed",
            created_at=now,
        )

    def begin_recent_auth(self, *, user_id: UserId) -> LocalAuthOptions:
        now = self._clock.now()
        passkeys = self._repository.list_passkeys(user_id=user_id)
        if not passkeys:
            raise LocalAuthError()
        challenge_bytes = secrets.token_bytes(32)
        challenge = self._repository.create_challenge(
            purpose="recent_auth",
            challenge_sha256=_sha256_bytes(challenge_bytes),
            user_id=user_id,
            context={},
            created_at=now,
            expires_at=now + _CHALLENGE_TTL,
        )
        options = generate_authentication_options(
            rp_id=self._rp_id,
            challenge=challenge_bytes,
            allow_credentials=[
                PublicKeyCredentialDescriptor(id=_decode_base64url(item.credential_id))
                for item in passkeys
            ],
            user_verification=UserVerificationRequirement.REQUIRED,
        )
        return _options(challenge_id=challenge.challenge_id, value=options)

    def complete_recent_auth(
        self,
        *,
        challenge_id: UUID,
        user_id: UserId,
        credential: Mapping[str, Any],
        session_id_to_rotate: UUID,
    ) -> LocalAuthResult:
        result = self._complete_passkey_authentication(
            challenge_id=challenge_id,
            credential=credential,
            purpose="recent_auth",
            expected_user_id=user_id,
        )
        rotated_at = self._clock.now()
        try:
            revoked = self._sessions.revoke_session(
                session_id=session_id_to_rotate,
                revoked_at=rotated_at,
            )
        except Exception:
            self._sessions.revoke_session(
                session_id=result.session.session_id,
                revoked_at=rotated_at,
            )
            raise
        if revoked is None:
            self._sessions.revoke_session(
                session_id=result.session.session_id,
                revoked_at=rotated_at,
            )
            raise LocalAuthError()
        return result

    def _complete_passkey_authentication(
        self,
        *,
        challenge_id: UUID,
        credential: Mapping[str, Any],
        purpose: LocalAuthPurpose,
        expected_user_id: UserId | None,
    ) -> LocalAuthResult:
        now = self._clock.now()
        credential_id = str(credential.get("id", "")).strip()
        subject_hash = _subject_hash("passkey", credential_id)
        self._enforce_rate_limit(
            user_id=expected_user_id,
            subject_hash=subject_hash,
            action=f"local_auth.{purpose}",
            now=now,
        )
        passkey = self._repository.find_passkey(credential_id=credential_id)
        if passkey is None or (
            expected_user_id is not None and passkey.user_id != expected_user_id
        ):
            self._record_failure(
                user_id=expected_user_id,
                subject_hash=subject_hash,
                action=f"local_auth.{purpose}",
                now=now,
            )
        try:
            challenge, challenge_bytes = self._validated_challenge(
                challenge_id=challenge_id,
                purpose=purpose,
                credential=credential,
            )
            if challenge.user_id is not None and challenge.user_id != passkey.user_id:
                raise LocalAuthError()
            verification = verify_authentication_response(
                credential=dict(credential),
                expected_challenge=challenge_bytes,
                expected_rp_id=self._rp_id,
                expected_origin=self._expected_origin,
                credential_public_key=passkey.public_key,
                credential_current_sign_count=passkey.sign_count,
                require_user_verification=True,
            )
            self._repository.finish_passkey_authentication(
                challenge_id=challenge_id,
                credential_id=credential_id,
                new_sign_count=verification.new_sign_count,
                completed_at=now,
            )
        except (LocalAuthRepositoryError, WebAuthnException, LocalAuthError) as error:
            self._repository.record_auth_failure(subject_sha256=subject_hash, now=now)
            self._reject(
                user_id=passkey.user_id,
                subject_hash=subject_hash,
                action=f"local_auth.{purpose}",
                reason="invalid_credential",
                now=now,
                cause=error,
            )
        self._repository.clear_auth_failures(subject_sha256=subject_hash)
        self._users.record_local_login(user_id=passkey.user_id, login_at=now)
        self._repository.record_auth_event(
            user_id=passkey.user_id,
            subject_sha256=subject_hash,
            action=f"local_auth.{purpose}",
            outcome="succeeded",
            reason_code="completed",
            created_at=now,
        )
        return LocalAuthResult(session=self._issue_session(user_id=passkey.user_id, now=now))

    def _validated_challenge(
        self,
        *,
        challenge_id: UUID,
        purpose: LocalAuthPurpose,
        credential: Mapping[str, Any],
    ) -> tuple[LocalAuthChallenge, bytes]:
        challenge = self._repository.find_challenge(
            challenge_id=challenge_id,
            purpose=purpose,
            now=self._clock.now(),
        )
        if challenge is None:
            raise LocalAuthError()
        challenge_bytes = _client_challenge(credential)
        if not secrets.compare_digest(challenge.challenge_sha256, _sha256_bytes(challenge_bytes)):
            raise LocalAuthError()
        return challenge, challenge_bytes

    def _require_account(self, *, user_id: UserId) -> LocalAccount:
        account = self._repository.find_account_by_user_id(user_id=user_id)
        if account is None:
            raise LocalAuthError()
        return account

    def _issue_session(self, *, user_id: UserId, now: datetime) -> IdentitySession:
        return self._sessions.create_session(
            user_id=user_id,
            now=now,
            idle_ttl_seconds=self._idle_ttl,
            absolute_ttl_seconds=self._absolute_ttl,
        )

    def _enforce_rate_limit(
        self,
        *,
        user_id: UserId | None,
        subject_hash: str,
        action: str,
        now: datetime,
    ) -> None:
        if self._repository.is_rate_limited(subject_sha256=subject_hash, now=now):
            self._reject(
                user_id=user_id,
                subject_hash=subject_hash,
                action=action,
                reason="rate_limited",
                now=now,
            )

    def _record_failure(
        self,
        *,
        user_id: UserId | None,
        subject_hash: str,
        action: str,
        now: datetime,
    ) -> NoReturn:
        self._repository.record_auth_failure(subject_sha256=subject_hash, now=now)
        self._reject(
            user_id=user_id,
            subject_hash=subject_hash,
            action=action,
            reason="invalid_credential",
            now=now,
        )

    def _reject(
        self,
        *,
        user_id: UserId | None,
        subject_hash: str,
        action: str,
        reason: str,
        now: datetime,
        cause: Exception | None = None,
    ) -> NoReturn:
        self._repository.record_auth_event(
            user_id=user_id,
            subject_sha256=subject_hash,
            action=action,
            outcome="rejected",
            reason_code=reason,
            created_at=now,
        )
        if cause is None:
            raise LocalAuthError()
        raise LocalAuthError() from cause


def _options(*, challenge_id: UUID, value: Any) -> LocalAuthOptions:
    payload = json.loads(options_to_json(value))
    if not isinstance(payload, dict):
        raise ValueError("WebAuthn options must be an object")
    return LocalAuthOptions(challenge_id=challenge_id, public_key=payload)


def _client_challenge(credential: Mapping[str, Any]) -> bytes:
    try:
        response = credential["response"]
        if not isinstance(response, Mapping):
            raise ValueError
        client_data = json.loads(_decode_base64url(str(response["clientDataJSON"])))
        if not isinstance(client_data, dict):
            raise ValueError
        return _decode_base64url(str(client_data["challenge"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LocalAuthError() from error


def _credential_transports(credential: Mapping[str, Any]) -> tuple[str, ...]:
    response = credential.get("response")
    if not isinstance(response, Mapping):
        return ()
    transports = response.get("transports")
    if not isinstance(transports, list):
        return ()
    return tuple(value for value in transports if isinstance(value, str) and value.strip())


def _bootstrap_user_id(challenge: LocalAuthChallenge) -> UserId:
    if challenge.user_id is not None:
        return challenge.user_id
    return UserId(UUID(challenge.context["bootstrap_user_id"]))


def _normalize_username(value: str) -> str:
    normalized = value.strip().lower()
    if not _USERNAME_PATTERN.fullmatch(normalized):
        raise LocalAuthError(code="invalid_bootstrap_input")
    return normalized


def _normalize_username_for_login(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if _USERNAME_PATTERN.fullmatch(normalized) else "invalid"


def _bounded_text(value: str, minimum: int, maximum: int) -> str:
    normalized = " ".join(value.split())
    if not minimum <= len(normalized) <= maximum:
        raise LocalAuthError(code="invalid_bootstrap_input")
    return normalized


def _new_recovery_code() -> str:
    raw = secrets.token_hex(12).upper()
    return "-".join(raw[index : index + 4] for index in range(0, len(raw), 4))


def _credential_id(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_base64url(value: str) -> bytes:
    encoded = value.encode("ascii")
    return base64.urlsafe_b64decode(encoded + b"=" * (-len(encoded) % 4))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _subject_hash(kind: str, value: str) -> str:
    return _sha256_text(f"{kind}:{value}")
