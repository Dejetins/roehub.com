from __future__ import annotations

from pathlib import Path
from uuid import UUID

from apps.worker.telegram_bot_worker.wiring.modules.telegram_bot_worker import (
    build_telegram_command_handler,
    load_telegram_bot_worker_runtime_config,
    openbao_service_input_presence,
)
from trading.contexts.notifications.adapters import InMemoryNotificationRepository
from trading.contexts.notifications.application import (
    InMemoryNotificationTelegramBindingStore,
    NotificationTelegramBindingService,
)
from trading.shared_kernel.primitives import OrganizationId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000003")


def test_telegram_bot_worker_config_defaults_disabled_and_reports_presence_only() -> None:
    for config_path in (
        Path("configs/dev/notifications.yaml"),
        Path("configs/test/notifications.yaml"),
        Path("configs/prod/notifications.yaml"),
    ):
        runtime_config = load_telegram_bot_worker_runtime_config(config_path=config_path)
        assert runtime_config.enabled is False
        assert runtime_config.poll_interval_seconds == 5
        assert runtime_config.long_poll_timeout_seconds == 30
        assert runtime_config.telegram_connect_timeout_seconds == 3.0

    presence = openbao_service_input_presence(
        environ={
            "ROEHUB_NOTIFICATIONS_OPENBAO_ADDRESS": "http://openbao:8200",
            "ROEHUB_TELEGRAM_WORKER_OPENBAO_TOKEN_FILE": "/run/secrets/token",
            "ROEHUB_OPENBAO_ROOT": "kv/roehub",
        }
    )
    assert presence == {
        "ROEHUB_NOTIFICATIONS_OPENBAO_ADDRESS": True,
        "ROEHUB_TELEGRAM_WORKER_OPENBAO_TOKEN_FILE": True,
        "ROEHUB_OPENBAO_ROOT": True,
    }
    assert "/run/secrets/token" not in repr(presence)


def test_telegram_bot_worker_builds_command_handler() -> None:
    handler = build_telegram_command_handler(
        repository=InMemoryNotificationRepository(),
        binding_service=NotificationTelegramBindingService(
            store=InMemoryNotificationTelegramBindingStore(),
            organization_id=_ORGANIZATION_ID,
            provider_instance_id=_PROVIDER_INSTANCE_ID,
        ),
    )

    assert handler is not None
