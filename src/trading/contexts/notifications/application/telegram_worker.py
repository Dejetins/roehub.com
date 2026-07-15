from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol
from uuid import UUID

from trading.contexts.notifications.application.ports import NotificationProviderRepository
from trading.contexts.notifications.application.telegram_commands import (
    TelegramCommandHandler,
    TelegramInboundCommand,
)
from trading.shared_kernel.primitives import OrganizationId


class TelegramUpdateSource(Protocol):
    def fetch_updates(
        self, *, offset: int, long_poll_timeout_seconds: int
    ) -> tuple[Mapping[str, Any], ...]: ...


class TelegramRecipientScopeResolver(Protocol):
    def resolve_organization(
        self,
        *,
        provider_instance_id: UUID,
        chat_id_ref: str,
        command_text: str | None,
        now: datetime,
    ) -> OrganizationId | None: ...


class TelegramInboundUpdateMapper(Protocol):
    def chat_id_ref_from_update(self, *, update: Mapping[str, Any]) -> str | None: ...

    def command_text_from_update(self, *, update: Mapping[str, Any]) -> str | None: ...

    def command_from_update(
        self,
        *,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        update: Mapping[str, Any],
        received_at: datetime,
    ) -> TelegramInboundCommand | None: ...


class TelegramCommandHandlerFactory(Protocol):
    def __call__(self, organization_id: OrganizationId, /) -> TelegramCommandHandler: ...


@dataclass(frozen=True, slots=True)
class TelegramProviderWorkerResult:
    fetched: int
    handled: int
    duplicates: int
    ignored: int
    cursor: int


class TelegramProviderWorker:
    """Poll one provider instance and advance its cursor only after durable handling."""

    def __init__(
        self,
        *,
        provider_instance_id: UUID,
        organization_id: OrganizationId | None,
        provider_repository: NotificationProviderRepository,
        update_source: TelegramUpdateSource,
        scope_resolver: TelegramRecipientScopeResolver,
        command_handler_factory: TelegramCommandHandlerFactory,
        mapper: TelegramInboundUpdateMapper,
        long_poll_timeout_seconds: int = 30,
    ) -> None:
        if long_poll_timeout_seconds <= 0 or long_poll_timeout_seconds > 50:
            raise ValueError("Telegram long-poll timeout must be in (0, 50]")
        self._provider_instance_id = provider_instance_id
        self._organization_id = organization_id
        self._provider_repository = provider_repository
        self._update_source = update_source
        self._scope_resolver = scope_resolver
        self._command_handler_factory = command_handler_factory
        self._mapper = mapper
        self._long_poll_timeout_seconds = long_poll_timeout_seconds

    def run_once(self, *, now: datetime) -> TelegramProviderWorkerResult:
        cursor = self._provider_repository.get_cursor(
            provider_instance_id=self._provider_instance_id
        )
        if cursor is None:
            raise ValueError("Telegram provider cursor is unavailable")
        updates = self._update_source.fetch_updates(
            offset=cursor.last_update_id + 1,
            long_poll_timeout_seconds=self._long_poll_timeout_seconds,
        )
        handled = 0
        duplicates = 0
        ignored = 0
        current_cursor = cursor.last_update_id
        for raw_update in sorted(updates, key=_update_id):
            update_id = _update_id(raw_update)
            if update_id <= current_cursor:
                duplicates += 1
                continue
            chat_id_ref = self._mapper.chat_id_ref_from_update(update=raw_update)
            organization_id = self._organization_id
            if organization_id is None and chat_id_ref is not None:
                organization_id = self._scope_resolver.resolve_organization(
                    provider_instance_id=self._provider_instance_id,
                    chat_id_ref=chat_id_ref,
                    command_text=self._mapper.command_text_from_update(update=raw_update),
                    now=now,
                )
            command = None
            if organization_id is not None:
                command = self._mapper.command_from_update(
                    organization_id=organization_id,
                    provider_instance_id=self._provider_instance_id,
                    update=raw_update,
                    received_at=now,
                )
            if command is None:
                ignored += 1
            else:
                assert organization_id is not None
                handler: TelegramCommandHandler = self._command_handler_factory(
                    organization_id
                )
                result = handler.handle(command=command)
                handled += 1
                duplicates += int(result.idempotent_replay)
            advanced = self._provider_repository.advance_cursor(
                provider_instance_id=self._provider_instance_id,
                expected_last_update_id=current_cursor,
                next_update_id=update_id,
                updated_at=now,
            )
            current_cursor = advanced.last_update_id
        return TelegramProviderWorkerResult(
            fetched=len(updates),
            handled=handled,
            duplicates=duplicates,
            ignored=ignored,
            cursor=current_cursor,
        )


def _update_id(update: Mapping[str, Any]) -> int:
    value = update.get("update_id")
    if not isinstance(value, int) or value < 0:
        raise ValueError("Telegram update_id is invalid")
    return value
