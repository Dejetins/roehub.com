from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.notifications.application.telegram_commands import (
    TelegramInboundCommand,
)
from trading.platform.secrets import SecretValue
from trading.shared_kernel.primitives import OrganizationId


class TelegramUpdateMapper:
    def chat_id_ref_from_update(self, *, update: Mapping[str, Any]) -> str | None:
        raw_chat_id = _raw_chat_id(update=update)
        return None if raw_chat_id is None else _chat_id_ref(raw_chat_id=raw_chat_id)

    def command_text_from_update(self, *, update: Mapping[str, Any]) -> str | None:
        message = update.get("message")
        if not isinstance(message, Mapping):
            return None
        text = message.get("text")
        return text if isinstance(text, str) else None

    def command_from_update(
        self,
        *,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        update: Mapping[str, Any],
        received_at: datetime,
    ) -> TelegramInboundCommand | None:
        update_id = update.get("update_id")
        message = update.get("message")
        if not isinstance(update_id, int) or not isinstance(message, Mapping):
            return None
        text = message.get("text")
        if not isinstance(text, str) or not text.startswith("/"):
            return None
        chat_id_ref = self.chat_id_ref_from_update(update=update)
        raw_chat_id = _raw_chat_id(update=update)
        if chat_id_ref is None or raw_chat_id is None:
            return None
        return TelegramInboundCommand(
            organization_id=organization_id,
            provider_instance_id=provider_instance_id,
            telegram_update_id=update_id,
            chat_id_ref=chat_id_ref,
            chat_id=SecretValue.from_text(str(raw_chat_id)),
            command_text=text,
            received_at=received_at,
        )


def _chat_id_ref(*, raw_chat_id: int) -> str:
    digest = sha256(str(raw_chat_id).encode()).hexdigest()
    suffix = str(abs(raw_chat_id))[-4:].rjust(4, "0")
    return f"telegram_ref:{digest[:16]}:{suffix}"


def _raw_chat_id(*, update: Mapping[str, Any]) -> int | None:
    message = update.get("message")
    if not isinstance(message, Mapping):
        return None
    chat = message.get("chat")
    if not isinstance(chat, Mapping):
        return None
    raw_chat_id = chat.get("id")
    if not isinstance(raw_chat_id, int) or isinstance(raw_chat_id, bool) or raw_chat_id == 0:
        return None
    return raw_chat_id
