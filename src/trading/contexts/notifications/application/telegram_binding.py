from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from hashlib import sha256
from secrets import choice
from string import ascii_uppercase, digits
from typing import Protocol
from uuid import UUID, uuid4

from trading.shared_kernel.primitives import UserId

_CODE_ALPHABET = ascii_uppercase + digits


class NotificationTelegramBindingError(ValueError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class NotificationTelegramBindingCode:
    binding_code_id: UUID
    owner_user_id: UserId
    code_hash: str
    expires_at: datetime
    created_at: datetime
    consumed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class NotificationTelegramBindingCodeView:
    code: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class NotificationTelegramBindingStatus:
    owner_user_id: UserId
    is_confirmed: bool
    chat_id_ref: str | None
    confirmed_at: datetime | None


class NotificationTelegramBindingStore(Protocol):
    def save_binding_code(
        self, *, binding_code: NotificationTelegramBindingCode
    ) -> NotificationTelegramBindingCode: ...

    def get_active_binding_code_by_hash(
        self, *, code_hash: str, now: datetime
    ) -> NotificationTelegramBindingCode | None: ...

    def consume_binding_code(
        self, *, binding_code_id: UUID, consumed_at: datetime
    ) -> NotificationTelegramBindingCode | None: ...

    def confirm_chat(
        self, *, owner_user_id: UserId, chat_id_ref: str, confirmed_at: datetime
    ) -> NotificationTelegramBindingStatus: ...

    def get_binding_status(
        self, *, owner_user_id: UserId
    ) -> NotificationTelegramBindingStatus: ...

    def find_owner_by_chat_ref(self, *, chat_id_ref: str) -> UserId | None: ...


@dataclass(slots=True)
class InMemoryNotificationTelegramBindingStore:
    binding_codes: dict[UUID, NotificationTelegramBindingCode]
    bindings_by_owner: dict[UserId, NotificationTelegramBindingStatus]
    owners_by_chat_ref: dict[str, UserId]

    def __init__(self) -> None:
        self.binding_codes = {}
        self.bindings_by_owner = {}
        self.owners_by_chat_ref = {}

    def save_binding_code(
        self, *, binding_code: NotificationTelegramBindingCode
    ) -> NotificationTelegramBindingCode:
        self.binding_codes[binding_code.binding_code_id] = binding_code
        return binding_code

    def get_active_binding_code_by_hash(
        self, *, code_hash: str, now: datetime
    ) -> NotificationTelegramBindingCode | None:
        for binding_code in self.binding_codes.values():
            if (
                binding_code.code_hash == code_hash
                and binding_code.consumed_at is None
                and binding_code.expires_at >= now
            ):
                return binding_code
        return None

    def consume_binding_code(
        self, *, binding_code_id: UUID, consumed_at: datetime
    ) -> NotificationTelegramBindingCode | None:
        binding_code = self.binding_codes.get(binding_code_id)
        if binding_code is None or binding_code.consumed_at is not None:
            return None
        consumed = replace(binding_code, consumed_at=consumed_at)
        self.binding_codes[binding_code_id] = consumed
        return consumed

    def confirm_chat(
        self, *, owner_user_id: UserId, chat_id_ref: str, confirmed_at: datetime
    ) -> NotificationTelegramBindingStatus:
        status = NotificationTelegramBindingStatus(
            owner_user_id=owner_user_id,
            is_confirmed=True,
            chat_id_ref=chat_id_ref,
            confirmed_at=confirmed_at,
        )
        self.bindings_by_owner[owner_user_id] = status
        self.owners_by_chat_ref[chat_id_ref] = owner_user_id
        return status

    def get_binding_status(
        self, *, owner_user_id: UserId
    ) -> NotificationTelegramBindingStatus:
        return self.bindings_by_owner.get(
            owner_user_id,
            NotificationTelegramBindingStatus(
                owner_user_id=owner_user_id,
                is_confirmed=False,
                chat_id_ref=None,
                confirmed_at=None,
            ),
        )

    def find_owner_by_chat_ref(self, *, chat_id_ref: str) -> UserId | None:
        return self.owners_by_chat_ref.get(chat_id_ref)


class NotificationTelegramBindingService:
    def __init__(
        self,
        *,
        store: NotificationTelegramBindingStore,
        ttl_seconds: int = 600,
        code_length: int = 8,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("NotificationTelegramBindingService.ttl_seconds must be > 0")
        if not 6 <= code_length <= 32:
            raise ValueError("NotificationTelegramBindingService.code_length must be 6..32")
        self._store = store
        self._ttl_seconds = ttl_seconds
        self._code_length = code_length

    def create_binding_code(
        self, *, owner_user_id: UserId, now: datetime
    ) -> NotificationTelegramBindingCodeView:
        code = "".join(choice(_CODE_ALPHABET) for _ in range(self._code_length))
        expires_at = now + timedelta(seconds=self._ttl_seconds)
        self._store.save_binding_code(
            binding_code=NotificationTelegramBindingCode(
                binding_code_id=uuid4(),
                owner_user_id=owner_user_id,
                code_hash=_hash_binding_code(code=code),
                expires_at=expires_at,
                created_at=now,
            )
        )
        return NotificationTelegramBindingCodeView(code=code, expires_at=expires_at)

    def confirm_binding_code(
        self, *, code: str, chat_id_ref: str, now: datetime
    ) -> NotificationTelegramBindingStatus:
        normalized = _normalize_binding_code(code=code)
        binding_code = self._store.get_active_binding_code_by_hash(
            code_hash=_hash_binding_code(code=normalized),
            now=now,
        )
        if binding_code is None:
            raise NotificationTelegramBindingError(reason="binding_code_invalid_or_expired")
        consumed = self._store.consume_binding_code(
            binding_code_id=binding_code.binding_code_id,
            consumed_at=now,
        )
        if consumed is None:
            raise NotificationTelegramBindingError(reason="binding_code_already_used")
        return self._store.confirm_chat(
            owner_user_id=binding_code.owner_user_id,
            chat_id_ref=chat_id_ref,
            confirmed_at=now,
        )

    def get_binding_status(
        self, *, owner_user_id: UserId
    ) -> NotificationTelegramBindingStatus:
        return self._store.get_binding_status(owner_user_id=owner_user_id)

    def owner_for_chat_ref(self, *, chat_id_ref: str) -> UserId | None:
        return self._store.find_owner_by_chat_ref(chat_id_ref=chat_id_ref)


def _normalize_binding_code(*, code: str) -> str:
    normalized = code.strip().upper()
    if not 6 <= len(normalized) <= 32:
        raise NotificationTelegramBindingError(reason="binding_code_invalid_length")
    if any(char not in _CODE_ALPHABET for char in normalized):
        raise NotificationTelegramBindingError(reason="binding_code_invalid_charset")
    return normalized


def _hash_binding_code(*, code: str) -> str:
    return sha256(_normalize_binding_code(code=code).encode()).hexdigest()
