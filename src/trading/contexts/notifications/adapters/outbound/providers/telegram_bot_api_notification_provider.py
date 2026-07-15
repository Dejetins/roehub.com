from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Callable, Mapping, Protocol, cast
from uuid import UUID

import httpx

from trading.contexts.notifications.application.ports import NotificationProviderResult
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationProviderDescriptor,
    NotificationProviderHealth,
    NotificationProviderInstance,
    telegram_bot_provider_descriptor,
)
from trading.shared_kernel.primitives import OrganizationId


@dataclass(frozen=True, slots=True)
class TelegramNotificationProviderConfig:
    instance: NotificationProviderInstance
    api_base_url: str = "https://api.telegram.org"
    connect_timeout_seconds: float = 3.0
    overall_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.instance.provider_key != "telegram_bot_api":
            raise ValueError("Telegram provider config requires a Telegram instance")
        api_base_url = self.api_base_url.strip().rstrip("/")
        if not api_base_url.startswith(("https://", "http://")):
            raise ValueError("Telegram api_base_url must be HTTP(S)")
        if not 0 < self.connect_timeout_seconds <= 3:
            raise ValueError("Telegram connect timeout must be in (0, 3]")
        if not 0 < self.overall_timeout_seconds <= 10:
            raise ValueError("Telegram overall timeout must be in (0, 10]")
        if self.connect_timeout_seconds > self.overall_timeout_seconds:
            raise ValueError("Telegram connect timeout must not exceed overall timeout")
        object.__setattr__(self, "api_base_url", api_base_url)


class TelegramHttpResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    def json(self) -> Any: ...


class TelegramHttpSession(Protocol):
    def post(
        self,
        *,
        url: str,
        json: Mapping[str, str],
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> TelegramHttpResponse: ...

    def get(self, *, url: str, timeout: httpx.Timeout) -> TelegramHttpResponse: ...


class TelegramBotApiNotificationProvider:
    def __init__(
        self,
        *,
        config: TelegramNotificationProviderConfig,
        credential_source: Callable[[], str],
        recipient_resolver: Callable[[str, OrganizationId, UUID], str],
        session: TelegramHttpSession | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._credential_source = credential_source
        self._recipient_resolver = recipient_resolver
        self._session = (
            session if session is not None else cast(TelegramHttpSession, httpx.Client())
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def provider_instance_id(self) -> UUID:
        return self._config.instance.instance_id

    @property
    def provider_key(self) -> str:
        return self._config.instance.provider_key

    @property
    def organization_id(self) -> OrganizationId | None:
        return self._config.instance.organization_id

    @property
    def descriptor(self) -> NotificationProviderDescriptor:
        return telegram_bot_provider_descriptor()

    def health(self) -> NotificationProviderHealth:
        if self._config.instance.status == "disabled":
            return NotificationProviderHealth(
                instance_id=self.provider_instance_id,
                status="disabled",
                checked_at=self._clock(),
                error_code="provider_disabled",
            )
        try:
            credential = self._credential_source().strip()
            if not credential:
                raise ValueError("credential is empty")
            response = self._session.get(
                url=f"{self._config.api_base_url}/bot{credential}/getMe",
                timeout=self._timeout(),
            )
            payload = _parse_json_object(response=response)
            ready = response.status_code == 200 and payload.get("ok") is True
        except Exception:  # noqa: BLE001
            ready = False
        return NotificationProviderHealth(
            instance_id=self.provider_instance_id,
            status="ready" if ready else "degraded",
            checked_at=self._clock(),
            error_code=None if ready else "provider_transport_error",
        )

    def send(self, *, delivery: NotificationDelivery) -> NotificationProviderResult:
        request_hash = _redacted_request_hash(delivery=delivery)
        if self._config.instance.status == "disabled":
            return NotificationProviderResult(
                status="suppressed",
                error_code="provider_disabled",
                redacted_request_hash=request_hash,
            )
        if (
            delivery.provider_instance_id != self.provider_instance_id
            or delivery.provider_key != self.provider_key
            or not self._config.instance.permits(
                organization_id=delivery.organization_id
            )
        ):
            return NotificationProviderResult(
                status="dead_letter",
                error_code="provider_scope_mismatch",
                redacted_request_hash=request_hash,
            )

        try:
            credential = self._credential_source().strip()
            if not credential:
                raise ValueError("credential is empty")
            recipient = self._recipient_resolver(
                delivery.recipient_address_ref,
                delivery.organization_id,
                delivery.provider_instance_id,
            )
            if not recipient.strip():
                raise ValueError("recipient is empty")
        except Exception:  # noqa: BLE001
            return NotificationProviderResult(
                status="dead_letter",
                error_code="provider_secret_unavailable",
                redacted_request_hash=request_hash,
            )

        try:
            response = self._session.post(
                url=f"{self._config.api_base_url}/bot{credential}/sendMessage",
                json={
                    "chat_id": recipient,
                    "text": str(delivery.rendered_payload_json.get("text", ""))[:4096],
                },
                headers={"X-Roehub-Delivery-Id": str(delivery.delivery_id)},
                timeout=self._timeout(),
            )
        except (httpx.ConnectTimeout, httpx.ConnectError):
            return NotificationProviderResult(
                status="retry",
                error_code="provider_connect_timeout",
                redacted_request_hash=request_hash,
            )
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            return NotificationProviderResult(
                status="unknown",
                error_code="provider_timeout_after_acceptance_possible",
                redacted_request_hash=request_hash,
            )
        except httpx.TransportError:
            return NotificationProviderResult(
                status="unknown",
                error_code="provider_transport_error",
                redacted_request_hash=request_hash,
            )
        except Exception:  # noqa: BLE001
            return NotificationProviderResult(
                status="unknown",
                error_code="provider_transport_error",
                redacted_request_hash=request_hash,
            )

        payload = _parse_json_object(response=response)
        response_hash = _redacted_response_hash(
            status_code=response.status_code,
            ok=payload.get("ok") is True,
        )
        if response.status_code == 200 and payload.get("ok") is True:
            return NotificationProviderResult(
                status="sent",
                provider_message_id=_provider_message_id(payload=payload),
                redacted_request_hash=request_hash,
                redacted_response_hash=response_hash,
            )
        if response.status_code == 429:
            return NotificationProviderResult(
                status="retry",
                error_code="provider_rate_limited",
                retry_after_seconds=_retry_after_seconds(payload=payload),
                redacted_request_hash=request_hash,
                redacted_response_hash=response_hash,
            )
        if response.status_code in {500, 502, 503, 504}:
            return NotificationProviderResult(
                status="retry",
                error_code="provider_http_error",
                redacted_request_hash=request_hash,
                redacted_response_hash=response_hash,
            )
        return NotificationProviderResult(
            status="dead_letter",
            error_code="provider_http_error",
            redacted_request_hash=request_hash,
            redacted_response_hash=response_hash,
        )

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            self._config.overall_timeout_seconds,
            connect=self._config.connect_timeout_seconds,
        )


def _redacted_request_hash(*, delivery: NotificationDelivery) -> str:
    payload = {
        "delivery_id": str(delivery.delivery_id),
        "organization_id": str(delivery.organization_id),
        "provider_instance_id": str(delivery.provider_instance_id),
        "provider_key": delivery.provider_key,
        "template_key": delivery.template_key,
        "recipient": "<redacted>",
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _redacted_response_hash(*, status_code: int, ok: bool) -> str:
    return sha256(f"status={status_code};ok={str(ok).lower()}".encode()).hexdigest()


def _parse_json_object(*, response: TelegramHttpResponse) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _retry_after_seconds(*, payload: Mapping[str, object]) -> int | None:
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        return None
    retry_after = parameters.get("retry_after")
    if isinstance(retry_after, int) and retry_after >= 0:
        return retry_after
    return None


def _provider_message_id(*, payload: Mapping[str, object]) -> str | None:
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    message_id = result.get("message_id")
    return str(message_id) if isinstance(message_id, int | str) else None
