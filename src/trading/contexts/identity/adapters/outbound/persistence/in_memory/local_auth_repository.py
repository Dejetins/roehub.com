from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from threading import RLock
from typing import Literal, Mapping
from uuid import UUID, uuid4

from trading.contexts.identity.adapters.outbound.persistence.in_memory.user_repository import (
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.application.ports.local_auth_repository import (
    LocalAccount,
    LocalAuthChallenge,
    LocalAuthPurpose,
    LocalAuthRepository,
    LocalAuthRepositoryError,
    LocalPasskey,
    RecoveryCodeHash,
)
from trading.shared_kernel.primitives import UserId

from .organization_repository import InMemoryOrganizationRepository

_RATE_WINDOW = timedelta(minutes=15)
_LOCKOUT = timedelta(minutes=15)
_MAX_FAILURES = 5


class InMemoryLocalAuthRepository(LocalAuthRepository):
    def __init__(
        self,
        *,
        user_repository: InMemoryIdentityUserRepository,
        organization_repository: InMemoryOrganizationRepository,
    ) -> None:
        self._users = user_repository
        self._organizations = organization_repository
        self._lock = RLock()
        self._tickets: dict[UUID, tuple[str, datetime, datetime | None]] = {}
        self._challenges: dict[UUID, LocalAuthChallenge] = {}
        self._accounts: dict[str, LocalAccount] = {}
        self._passkeys: dict[str, LocalPasskey] = {}
        self._recovery_codes: dict[UUID, tuple[UserId, str, datetime | None]] = {}
        self._failures: dict[str, tuple[int, datetime, datetime | None]] = {}
        self._events: list[dict[str, str]] = []

    def bootstrap_required(self) -> bool:
        return self._organizations.get_installation() is None

    def issue_bootstrap_ticket(
        self, *, token_sha256: str, created_at: datetime, expires_at: datetime
    ) -> UUID:
        with self._lock:
            if not self.bootstrap_required():
                raise LocalAuthRepositoryError(code="bootstrap_unavailable")
            for ticket_id, (stored_hash, _expiry, consumed_at) in tuple(
                self._tickets.items()
            ):
                if consumed_at is None:
                    self._tickets[ticket_id] = (stored_hash, _expiry, created_at)
            ticket_id = uuid4()
            self._tickets[ticket_id] = (token_sha256, expires_at, None)
            return ticket_id

    def find_bootstrap_ticket(self, *, token_sha256: str, now: datetime) -> UUID | None:
        with self._lock:
            for ticket_id, (stored_hash, expires_at, consumed_at) in self._tickets.items():
                if (
                    consumed_at is None
                    and expires_at > now
                    and stored_hash == token_sha256
                ):
                    return ticket_id
        return None

    def create_challenge(
        self,
        *,
        purpose: LocalAuthPurpose,
        challenge_sha256: str,
        user_id: UserId | None,
        context: Mapping[str, str],
        created_at: datetime,
        expires_at: datetime,
    ) -> LocalAuthChallenge:
        challenge = LocalAuthChallenge(
            challenge_id=uuid4(),
            purpose=purpose,
            challenge_sha256=challenge_sha256,
            user_id=user_id,
            context=dict(context),
            created_at=created_at,
            expires_at=expires_at,
        )
        with self._lock:
            self._challenges[challenge.challenge_id] = challenge
        return challenge

    def find_challenge(
        self, *, challenge_id: UUID, purpose: LocalAuthPurpose, now: datetime
    ) -> LocalAuthChallenge | None:
        challenge = self._challenges.get(challenge_id)
        if (
            challenge is None
            or challenge.purpose != purpose
            or challenge.consumed_at is not None
            or challenge.expires_at <= now
        ):
            return None
        return challenge

    def complete_bootstrap(
        self,
        *,
        challenge_id: UUID,
        ticket_id: UUID,
        user_id: UserId,
        username: str,
        display_name: str,
        password_hash: str | None,
        installation_name: str,
        organization_slug: str,
        organization_name: str,
        passkey: LocalPasskey,
        recovery_code_hashes: tuple[str, ...],
        completed_at: datetime,
    ) -> None:
        with self._lock:
            challenge = self.find_challenge(
                challenge_id=challenge_id,
                purpose="bootstrap",
                now=completed_at,
            )
            ticket = self._tickets.get(ticket_id)
            if challenge is None or ticket is None or ticket[2] is not None:
                raise LocalAuthRepositoryError(code="bootstrap_invalid")
            if not self.bootstrap_required() or username in self._accounts:
                raise LocalAuthRepositoryError(code="bootstrap_unavailable")
            self._users.create_local_user(user_id=user_id, created_at=completed_at)
            self._organizations.bootstrap_installation(
                owner_user_id=user_id,
                installation_name=installation_name,
                organization_slug=organization_slug,
                organization_name=organization_name,
                created_at=completed_at,
            )
            self._accounts[username] = LocalAccount(
                user_id=user_id,
                username=username,
                display_name=display_name,
                password_hash=password_hash,
                created_at=completed_at,
            )
            self._passkeys[passkey.credential_id] = passkey
            for code_hash in recovery_code_hashes:
                self._recovery_codes[uuid4()] = (user_id, code_hash, None)
            self._tickets[ticket_id] = (ticket[0], ticket[1], completed_at)
            self._challenges[challenge_id] = replace(
                challenge, consumed_at=completed_at
            )

    def find_account_by_username(self, *, username: str) -> LocalAccount | None:
        return self._accounts.get(username)

    def find_account_by_user_id(self, *, user_id: UserId) -> LocalAccount | None:
        return next(
            (
                account
                for account in self._accounts.values()
                if account.user_id == user_id
            ),
            None,
        )

    def find_passkey(self, *, credential_id: str) -> LocalPasskey | None:
        return self._passkeys.get(credential_id)

    def list_passkeys(self, *, user_id: UserId) -> tuple[LocalPasskey, ...]:
        return tuple(
            passkey
            for passkey in self._passkeys.values()
            if passkey.user_id == user_id
        )

    def add_passkey_and_consume_challenge(
        self,
        *,
        challenge_id: UUID,
        passkey: LocalPasskey,
        completed_at: datetime,
    ) -> None:
        with self._lock:
            challenge = self._challenges.get(challenge_id)
            if (
                challenge is None
                or challenge.consumed_at is not None
                or challenge.user_id != passkey.user_id
                or passkey.credential_id in self._passkeys
            ):
                raise LocalAuthRepositoryError(code="challenge_invalid")
            self._passkeys[passkey.credential_id] = passkey
            self._challenges[challenge_id] = replace(
                challenge, consumed_at=completed_at
            )

    def finish_passkey_authentication(
        self,
        *,
        challenge_id: UUID,
        credential_id: str,
        new_sign_count: int,
        completed_at: datetime,
    ) -> None:
        with self._lock:
            challenge = self._challenges.get(challenge_id)
            passkey = self._passkeys.get(credential_id)
            if challenge is None or challenge.consumed_at is not None or passkey is None:
                raise LocalAuthRepositoryError(code="challenge_invalid")
            self._passkeys[credential_id] = replace(
                passkey, sign_count=new_sign_count
            )
            self._challenges[challenge_id] = replace(
                challenge, consumed_at=completed_at
            )

    def list_recovery_code_hashes(
        self, *, user_id: UserId
    ) -> tuple[RecoveryCodeHash, ...]:
        return tuple(
            RecoveryCodeHash(recovery_code_id=code_id, code_hash=code_hash)
            for code_id, (owner_id, code_hash, consumed_at) in self._recovery_codes.items()
            if owner_id == user_id and consumed_at is None
        )

    def consume_recovery_code(
        self,
        *,
        recovery_code_id: UUID,
        user_id: UserId,
        consumed_at: datetime,
    ) -> bool:
        with self._lock:
            row = self._recovery_codes.get(recovery_code_id)
            if row is None or row[0] != user_id or row[2] is not None:
                return False
            self._recovery_codes[recovery_code_id] = (row[0], row[1], consumed_at)
            return True

    def is_rate_limited(self, *, subject_sha256: str, now: datetime) -> bool:
        row = self._failures.get(subject_sha256)
        return row is not None and row[2] is not None and row[2] > now

    def record_auth_failure(self, *, subject_sha256: str, now: datetime) -> None:
        with self._lock:
            count, window_started, locked_until = self._failures.get(
                subject_sha256, (0, now, None)
            )
            if now - window_started >= _RATE_WINDOW:
                count, window_started, locked_until = 0, now, None
            count += 1
            if count >= _MAX_FAILURES:
                locked_until = now + _LOCKOUT
            self._failures[subject_sha256] = (count, window_started, locked_until)

    def clear_auth_failures(self, *, subject_sha256: str) -> None:
        self._failures.pop(subject_sha256, None)

    def record_auth_event(
        self,
        *,
        user_id: UserId | None,
        subject_sha256: str,
        action: str,
        outcome: Literal["succeeded", "rejected"],
        reason_code: str,
        created_at: datetime,
    ) -> None:
        self._events.append(
            {
                "user_id": "" if user_id is None else str(user_id),
                "subject_sha256": subject_sha256,
                "action": action,
                "outcome": outcome,
                "reason_code": reason_code,
                "created_at": created_at.isoformat(),
            }
        )
