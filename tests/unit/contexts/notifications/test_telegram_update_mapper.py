from __future__ import annotations

from datetime import datetime, timezone

from trading.contexts.notifications.adapters.inbound.telegram import TelegramUpdateMapper


def test_telegram_update_mapper_redacts_chat_id_and_ignores_non_commands() -> None:
    mapper = TelegramUpdateMapper()
    received_at = datetime(2026, 6, 29, 15, 0, tzinfo=timezone.utc)

    command = mapper.command_from_update(
        update={
            "update_id": 501,
            "message": {
                "chat": {"id": 123456789},
                "text": "/stats today",
            },
        },
        received_at=received_at,
    )
    ignored = mapper.command_from_update(
        update={"update_id": 502, "message": {"chat": {"id": 123456789}, "text": "hello"}},
        received_at=received_at,
    )

    assert command is not None
    assert command.telegram_update_id == 501
    assert command.command_text == "/stats today"
    assert command.chat_id_ref.startswith("telegram_ref:")
    assert "123456789" not in command.chat_id_ref
    assert command.chat_id_ref.endswith(":6789")
    assert ignored is None
