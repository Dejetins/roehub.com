from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId

LocalAuthPurpose = Literal["bootstrap", "login", "register", "recent_auth"]


@dataclass(frozen=True, slots=True)
class LocalAccount:
    user_id: UserId
    username: str
    display_name: str
    password_hash: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class LocalAuthChallenge:
    challenge_id: UUID
    purpose: LocalAuthPurpose
    challenge_sha256: str
    user_id: UserId | None
    context: Mapping[str, str]
    created_at: datetime
    expires_at: datetime
    consumed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class LocalPasskey:
    credential_id: str
    user_id: UserId
    public_key: bytes
    sign_count: int
    transports: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RecoveryCodeHash:
    recovery_code_id: UUID
    code_hash: str


class LocalAuthRepositoryError(ValueError):
    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


class LocalAuthRepository(Protocol):
    def bootstrap_required(self) -> bool: ...

    def issue_bootstrap_ticket(
        self, *, token_sha256: str, created_at: datetime, expires_at: datetime
    ) -> UUID: ...

    def find_bootstrap_ticket(
        self, *, token_sha256: str, now: datetime
    ) -> UUID | None: ...

    def create_challenge(
        self,
        *,
        purpose: LocalAuthPurpose,
        challenge_sha256: str,
        user_id: UserId | None,
        context: Mapping[str, str],
        created_at: datetime,
        expires_at: datetime,
    ) -> LocalAuthChallenge: ...

    def find_challenge(
        self, *, challenge_id: UUID, purpose: LocalAuthPurpose, now: datetime
    ) -> LocalAuthChallenge | None: ...

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
    ) -> None: ...

    def find_account_by_username(self, *, username: str) -> LocalAccount | None: ...

    def find_account_by_user_id(self, *, user_id: UserId) -> LocalAccount | None: ...

    def find_passkey(self, *, credential_id: str) -> LocalPasskey | None: ...

    def list_passkeys(self, *, user_id: UserId) -> tuple[LocalPasskey, ...]: ...

    def add_passkey_and_consume_challenge(
        self,
        *,
        challenge_id: UUID,
        passkey: LocalPasskey,
        completed_at: datetime,
    ) -> None: ...

    def finish_passkey_authentication(
        self,
        *,
        challenge_id: UUID,
        credential_id: str,
        new_sign_count: int,
        completed_at: datetime,
    ) -> None: ...

    def list_recovery_code_hashes(
        self, *, user_id: UserId
    ) -> tuple[RecoveryCodeHash, ...]: ...

    def consume_recovery_code(
        self,
        *,
        recovery_code_id: UUID,
        user_id: UserId,
        consumed_at: datetime,
    ) -> bool: ...

    def is_rate_limited(self, *, subject_sha256: str, now: datetime) -> bool: ...

    def record_auth_failure(self, *, subject_sha256: str, now: datetime) -> None: ...

    def clear_auth_failures(self, *, subject_sha256: str) -> None: ...

    def record_auth_event(
        self,
        *,
        user_id: UserId | None,
        subject_sha256: str,
        action: str,
        outcome: Literal["succeeded", "rejected"],
        reason_code: str,
        created_at: datetime,
    ) -> None: ...
