from __future__ import annotations

from pathlib import Path

from apps.worker.telegram_bot_worker.wiring.modules.telegram_bot_worker import (
    build_telegram_command_handler,
    load_telegram_bot_worker_runtime_config,
    telegram_bot_credential_presence,
)


def test_telegram_bot_worker_config_defaults_disabled_and_reports_presence_only() -> None:
    for config_path in (
        Path("configs/dev/notifications.yaml"),
        Path("configs/test/notifications.yaml"),
        Path("configs/prod/notifications.yaml"),
    ):
        runtime_config = load_telegram_bot_worker_runtime_config(config_path=config_path)
        assert runtime_config.enabled is False
        assert runtime_config.telegram_enabled is False
        assert runtime_config.poll_interval_seconds == 5
        assert runtime_config.long_poll_timeout_seconds == 30

    presence = telegram_bot_credential_presence(
        environ={
            "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN": "set",
            "TELEGRAM_BOT_TOKEN": "",
        }
    )
    assert presence == {
        "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN": True,
        "TELEGRAM_BOT_TOKEN": False,
    }
    assert "set" not in repr(presence)


def test_telegram_bot_worker_builds_command_handler() -> None:
    handler = build_telegram_command_handler()

    assert handler is not None
