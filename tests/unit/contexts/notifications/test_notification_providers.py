from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4

import httpx
import pytest

from trading.contexts.notifications.adapters import (
    FakeNotificationProvider,
    HttpNotificationProvider,
    HttpNotificationProviderConfig,
    LogOnlyNotificationProvider,
    TelegramBotApiNotificationProvider,
    TelegramNotificationProviderConfig,
)
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationProviderDescriptor,
    NotificationProviderInstance,
)
from trading.shared_kernel.primitives import OrganizationId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
_LOG_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000001")
_FAKE_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000002")
_TELEGRAM_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000003")
_HTTP_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000004")


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
        config=TelegramNotificationProviderConfig(
            instance=_telegram_instance(status="disabled")
        ),
        credential_source=lambda: "unused",
        recipient_resolver=lambda *_args: "unused",
        session=session,
    )

    result = provider.send(delivery=_delivery(provider_key="telegram_bot_api"))

    assert result.status == "suppressed"
    assert result.error_code == "provider_disabled"
    assert session.calls == []


def test_degraded_instances_remain_callable_without_cross_instance_fallback() -> None:
    telegram_session = CapturingTelegramSession(
        response=_response(status_code=200, payload={"ok": True})
    )
    telegram = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(
            instance=_telegram_instance(status="degraded")
        ),
        credential_source=lambda: "stage11-credential",
        recipient_resolver=lambda *_args: "recipient-1",
        session=telegram_session,
    )
    http_session = CapturingHttpProviderSession(
        response=HttpProviderResponseFixture(status_code=202, payload={}, headers={})
    )
    http = HttpNotificationProvider(
        config=HttpNotificationProviderConfig(
            instance=_http_instance(status="degraded"),
            descriptor=_http_descriptor(),
            endpoint_url="https://provider.test/v1/deliveries",
        ),
        session=http_session,
    )

    assert telegram.send(
        delivery=_delivery(provider_key="telegram_bot_api")
    ).status == "sent"
    assert http.send(delivery=_delivery(provider_key="custom_http")).status == "sent"
    assert len(telegram_session.calls) == 1
    assert len(http_session.calls) == 1


def test_custom_http_provider_rejects_cross_origin_health_credentials() -> None:
    with pytest.raises(ValueError, match="endpoint origin"):
        HttpNotificationProviderConfig(
            instance=_http_instance(),
            descriptor=_http_descriptor(),
            endpoint_url="https://provider.test/v1/deliveries",
            health_url="https://other.test/health",
        )


def test_telegram_timeout_and_5xx_are_unknown_states() -> None:
    credential = "stage03-credential"
    timeout_provider = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(instance=_telegram_instance()),
        credential_source=lambda: credential,
        recipient_resolver=lambda *_args: "recipient-1",
        session=TimeoutTelegramSession(),
    )
    server_error_provider = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(instance=_telegram_instance()),
        credential_source=lambda: credential,
        recipient_resolver=lambda *_args: "recipient-1",
        session=CapturingTelegramSession(
            response=_response(status_code=502, payload={"ok": False})
        ),
    )

    timeout_result = timeout_provider.send(delivery=_delivery(provider_key="telegram_bot_api"))
    server_error_result = server_error_provider.send(
        delivery=_delivery(provider_key="telegram_bot_api")
    )

    assert timeout_result.status == "unknown"
    assert timeout_result.error_code == "provider_timeout_after_acceptance_possible"
    assert server_error_result.status == "retry"
    assert server_error_result.error_code == "provider_http_error"
    assert credential not in repr(timeout_result)
    assert credential not in repr(server_error_result)


def test_telegram_rate_limit_retries_with_retry_after() -> None:
    provider = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(instance=_telegram_instance()),
        credential_source=lambda: "stage03-credential",
        recipient_resolver=lambda *_args: "recipient-1",
        session=CapturingTelegramSession(
            response=_response(
                status_code=429,
                payload={"ok": False, "parameters": {"retry_after": 7}},
            )
        ),
    )

    result = provider.send(delivery=_delivery(provider_key="telegram_bot_api"))

    assert result.status == "retry"
    assert result.error_code == "provider_rate_limited"
    assert result.retry_after_seconds == 7


def test_telegram_success_extracts_provider_message_id_without_result_secrets() -> None:
    credential = "stage03-credential"
    session = CapturingTelegramSession(
        response=_response(status_code=200, payload={"ok": True, "result": {"message_id": 42}})
    )
    provider = TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(instance=_telegram_instance()),
        credential_source=lambda: credential,
        recipient_resolver=lambda *_args: "recipient-1",
        session=session,
    )

    result = provider.send(delivery=_delivery(provider_key="telegram_bot_api"))

    assert result.status == "sent"
    assert result.provider_message_id == "42"
    assert result.redacted_request_hash is not None
    assert result.redacted_response_hash is not None
    assert credential not in repr(result)
    assert session.calls[0]["json"] == {
        "chat_id": "recipient-1",
        "text": "Stage 03 provider smoke",
    }


def test_custom_http_provider_sends_versioned_idempotent_payload() -> None:
    credential = "custom-provider-credential"
    session = CapturingHttpProviderSession(
        response=HttpProviderResponseFixture(
            status_code=202,
            payload={"message_id": "accepted-42"},
            headers={},
        )
    )
    provider = HttpNotificationProvider(
        config=HttpNotificationProviderConfig(
            instance=_http_instance(),
            descriptor=_http_descriptor(),
            endpoint_url="https://provider.test/v1/deliveries",
        ),
        credential_source=lambda: credential,
        session=session,
    )
    delivery = _delivery(provider_key="custom_http")

    result = provider.send(delivery=delivery)

    assert result.status == "sent"
    assert result.provider_message_id == "accepted-42"
    call = session.calls[0]
    assert call["headers"]["X-Roehub-Delivery-Id"] == str(delivery.delivery_id)
    assert call["json"]["apiVersion"] == "notifications.roehub.io/v1"
    assert call["json"]["metadata"]["organization_id"] == str(_ORGANIZATION_ID)
    assert credential not in repr(result)


def test_custom_http_provider_classifies_retry_after_and_post_acceptance_timeout() -> None:
    rate_limited = HttpNotificationProvider(
        config=HttpNotificationProviderConfig(
            instance=_http_instance(),
            descriptor=_http_descriptor(),
            endpoint_url="https://provider.test/v1/deliveries",
        ),
        session=CapturingHttpProviderSession(
            response=HttpProviderResponseFixture(
                status_code=429,
                payload={},
                headers={"Retry-After": "11"},
            )
        ),
    ).send(delivery=_delivery(provider_key="custom_http"))
    unknown = HttpNotificationProvider(
        config=HttpNotificationProviderConfig(
            instance=_http_instance(),
            descriptor=_http_descriptor(),
            endpoint_url="https://provider.test/v1/deliveries",
        ),
        session=TimeoutHttpProviderSession(),
    ).send(delivery=_delivery(provider_key="custom_http"))

    assert rate_limited.status == "retry"
    assert rate_limited.retry_after_seconds == 11
    assert unknown.status == "unknown"
    assert unknown.error_code == "provider_timeout_after_acceptance_possible"


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
        self,
        *,
        url: str,
        json: Mapping[str, str],
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> TelegramResponse:
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return self.response

    def get(self, *, url: str, timeout: httpx.Timeout) -> TelegramResponse:
        self.calls.append({"url": url, "timeout": timeout})
        return self.response


class TimeoutTelegramSession:
    def post(
        self,
        *,
        url: str,
        json: Mapping[str, str],
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> TelegramResponse:
        _ = (url, json, headers, timeout)
        raise httpx.ReadTimeout("controlled", request=httpx.Request("POST", url))

    def get(self, *, url: str, timeout: httpx.Timeout) -> TelegramResponse:
        _ = timeout
        raise httpx.ReadTimeout("controlled", request=httpx.Request("GET", url))


@dataclass(frozen=True, slots=True)
class HttpProviderResponseFixture:
    status_code: int
    payload: Mapping[str, Any]
    headers: Mapping[str, str]

    def json(self) -> Mapping[str, Any]:
        return self.payload


@dataclass(slots=True)
class CapturingHttpProviderSession:
    response: HttpProviderResponseFixture
    calls: list[Mapping[str, Any]] = field(default_factory=list)

    def post(
        self,
        *,
        url: str,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> HttpProviderResponseFixture:
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return self.response

    def get(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> HttpProviderResponseFixture:
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        return self.response


class TimeoutHttpProviderSession:
    def post(
        self,
        *,
        url: str,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> HttpProviderResponseFixture:
        _ = (json, headers, timeout)
        raise httpx.ReadTimeout("controlled", request=httpx.Request("POST", url))

    def get(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> HttpProviderResponseFixture:
        _ = (headers, timeout)
        raise httpx.ReadTimeout("controlled", request=httpx.Request("GET", url))


def _response(*, status_code: int, payload: Mapping[str, Any]) -> TelegramResponse:
    return TelegramResponse(status_code=status_code, payload=payload)


def _delivery(*, provider_key: str) -> NotificationDelivery:
    provider_instance_id = {
        "log_only": _LOG_INSTANCE_ID,
        "fake": _FAKE_INSTANCE_ID,
        "telegram_bot_api": _TELEGRAM_INSTANCE_ID,
        "custom_http": _HTTP_INSTANCE_ID,
    }[provider_key]
    return NotificationDelivery(
        delivery_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=provider_instance_id,
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


def _telegram_instance(*, status: str = "active") -> NotificationProviderInstance:
    now = datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)
    return NotificationProviderInstance(
        instance_id=_TELEGRAM_INSTANCE_ID,
        package_id=UUID("00000000-0000-4000-8000-000000000103"),
        provider_key="telegram_bot_api",
        scope="organization",
        organization_id=_ORGANIZATION_ID,
        display_name="Organization Telegram",
        config_json={},
        secret_ref=(
            "openbao://kv/roehub/telegram/providers/"
            f"{_ORGANIZATION_ID}/{_TELEGRAM_INSTANCE_ID}#bot_token"
        ),
        status=status,  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
    )


def _http_instance(*, status: str = "active") -> NotificationProviderInstance:
    now = datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)
    return NotificationProviderInstance(
        instance_id=_HTTP_INSTANCE_ID,
        package_id=UUID("00000000-0000-4000-8000-000000000104"),
        provider_key="custom_http",
        scope="organization",
        organization_id=_ORGANIZATION_ID,
        display_name="Organization custom HTTP",
        config_json={},
        secret_ref=(
            f"openbao://kv/roehub/plugins/{_ORGANIZATION_ID}/"
            f"{_HTTP_INSTANCE_ID}#bearer_token"
        ),
        status=status,  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
    )


def _http_descriptor() -> NotificationProviderDescriptor:
    return NotificationProviderDescriptor(
        provider_key="custom_http",
        display_name="Custom HTTP",
        package_version="1.0.0",
        config_schema={"type": "object"},
        channels=("webhook",),
        templates=("plain_text.v1",),
        error_codes=(
            "provider_disabled",
            "provider_scope_mismatch",
            "provider_secret_unavailable",
            "provider_connect_timeout",
            "provider_transport_error",
            "provider_timeout_after_acceptance_possible",
            "provider_rate_limited",
            "provider_http_error",
        ),
    )
