from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4

import requests

from trading.contexts.notifications.adapters import (
    FakeNotificationProvider,
    LogOnlyNotificationProvider,
    TelegramBotApiNotificationProvider,
    TelegramNotificationProviderConfig,
)
from trading.contexts.notifications.domain import NotificationDelivery


def test_log_and_fake_providers_return_redacted_deterministic_success() -> None:
    delivery = _delivery(provider_key="log_only")

    log_result = LogOnlyNotificationProvider().send(delivery=delivery)
    fake_result = FakeNotificationProvider().send(
        delivery=_delivery(provider_key="fake")
    )

    assert log_result.status == "sent"
    assert fake_result.status == "sent"
    assert log_result.provider_message_id == f"log_only:{delivery.delivery_id}"
    assert log_result.redacted_request_hash is not None
    assert len(log_result.redacted_request_hash) == 64
    assert len(log_result.redacted_response_hash or "") == 64


def test_telegram_provider_disabled_suppresses_without_network_call() -> None:
    session = CapturingTelegramSession(response=_response(status_code=200, payload={"ok": True}))
    provider = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(enabled=False, credential=None),
        session=session,
    )

    result = provider.send(delivery=_delivery(provider_key="telegram_bot_api"))

    assert result.status == "suppressed"
    assert result.error_code == "telegram_provider_disabled"
    assert session.calls == []


def test_telegram_timeout_and_5xx_are_unknown_states() -> None:
    credential = "stage03-credential"
    timeout_provider = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(enabled=True, credential=credential),
        session=TimeoutTelegramSession(),
    )
    server_error_provider = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(enabled=True, credential=credential),
        session=CapturingTelegramSession(
            response=_response(status_code=502, payload={"ok": False})
        ),
    )

    timeout_result = timeout_provider.send(delivery=_delivery(provider_key="telegram_bot_api"))
    server_error_result = server_error_provider.send(
        delivery=_delivery(provider_key="telegram_bot_api")
    )

    assert timeout_result.status == "unknown"
    assert timeout_result.error_code == "telegram_timeout"
    assert server_error_result.status == "unknown"
    assert server_error_result.error_code == "telegram_http_502"
    assert credential not in repr(timeout_result)
    assert credential not in repr(server_error_result)


def test_telegram_rate_limit_retries_with_retry_after() -> None:
    provider = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(
            enabled=True, credential="stage03-credential"
        ),
        session=CapturingTelegramSession(
            response=_response(
                status_code=429,
                payload={"ok": False, "parameters": {"retry_after": 7}},
            )
        ),
    )

    result = provider.send(delivery=_delivery(provider_key="telegram_bot_api"))

    assert result.status == "retry"
    assert result.error_code == "telegram_rate_limited"
    assert result.retry_after_seconds == 7


def test_telegram_success_extracts_provider_message_id_without_result_secrets() -> None:
    credential = "stage03-credential"
    session = CapturingTelegramSession(
        response=_response(status_code=200, payload={"ok": True, "result": {"message_id": 42}})
    )
    provider = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(enabled=True, credential=credential),
        session=session,
    )

    result = provider.send(delivery=_delivery(provider_key="telegram_bot_api"))

    assert result.status == "sent"
    assert result.provider_message_id == "42"
    assert result.redacted_request_hash is not None
    assert result.redacted_response_hash is not None
    assert credential not in repr(result)
    assert session.calls[0]["json"] == {
        "chat_id": "telegram_ref:user:stage03",
        "text": "Stage 03 provider smoke",
    }


@dataclass(frozen=True, slots=True)
class TelegramResponse:
    status_code: int
    payload: Mapping[str, Any]

    @property
    def text(self) -> str:
        return repr(dict(self.payload))

    def json(self) -> Mapping[str, Any]:
        return self.payload


@dataclass(slots=True)
class CapturingTelegramSession:
    response: TelegramResponse
    calls: list[Mapping[str, Any]] = field(default_factory=list)

    def post(
        self, *, url: str, json: Mapping[str, str], timeout: float
    ) -> TelegramResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self.response


class TimeoutTelegramSession:
    def post(
        self, *, url: str, json: Mapping[str, str], timeout: float
    ) -> TelegramResponse:
        _ = (url, json, timeout)
        raise requests.exceptions.Timeout


def _response(*, status_code: int, payload: Mapping[str, Any]) -> TelegramResponse:
    return TelegramResponse(status_code=status_code, payload=payload)


def _delivery(*, provider_key: str) -> NotificationDelivery:
    return NotificationDelivery(
        delivery_id=uuid4(),
        event_id=UUID("33333333-3333-4333-8333-333333333333"),
        report_run_id=None,
        command_id=None,
        route_id=uuid4(),
        provider_key=provider_key,  # type: ignore[arg-type]
        channel_key="telegram",
        recipient_address_ref="telegram_ref:user:stage03",
        template_key="strategy_signal",
        rendered_payload_json={"text": "Stage 03 provider smoke"},
        status="pending",
        attempt_count=0,
        created_at=datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc),
    )
