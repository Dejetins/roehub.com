from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, Mapping

from trading.contexts.notifications.application.telegram_commands import (
    TelegramInboundCommand,
)


class TelegramUpdateMapper:
    def command_from_update(
        self, *, update: Mapping[str, Any], received_at: datetime
    ) -> TelegramInboundCommand | None:
        update_id = update.get("update_id")
        message = update.get("message")
        if not isinstance(update_id, int) or not isinstance(message, Mapping):
            return None
        text = message.get("text")
        chat = message.get("chat")
        if not isinstance(text, str) or not text.startswith("/"):
            return None
        if not isinstance(chat, Mapping):
            return None
        raw_chat_id = chat.get("id")
        if not isinstance(raw_chat_id, int) or raw_chat_id == 0:
            return None
        return TelegramInboundCommand(
            telegram_update_id=update_id,
            chat_id_ref=_chat_id_ref(raw_chat_id=raw_chat_id),
            command_text=text,
            received_at=received_at,
        )


def _chat_id_ref(*, raw_chat_id: int) -> str:
    digest = sha256(str(raw_chat_id).encode()).hexdigest()
    suffix = str(abs(raw_chat_id))[-4:].rjust(4, "0")
    return f"telegram_ref:{digest[:16]}:{suffix}"
