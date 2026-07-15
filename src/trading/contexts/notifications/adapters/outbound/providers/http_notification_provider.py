from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Callable, Mapping, Protocol, cast
from urllib.parse import urlsplit
from uuid import UUID

import httpx

from trading.contexts.notifications.application.ports import NotificationProviderResult
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationProviderDescriptor,
    NotificationProviderHealth,
    NotificationProviderInstance,
)
from trading.shared_kernel.primitives import OrganizationId


@dataclass(frozen=True, slots=True)
class HttpNotificationProviderConfig:
    instance: NotificationProviderInstance
    descriptor: NotificationProviderDescriptor
    endpoint_url: str
    health_url: str | None = None
    connect_timeout_seconds: float = 3.0
    overall_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.descriptor.provider_key != self.instance.provider_key:
            raise ValueError("HTTP provider descriptor does not match its instance")
        endpoint_url = _validated_url(self.endpoint_url, field="endpoint_url")
        health_url = (
            _validated_url(self.health_url, field="health_url")
            if self.health_url is not None
            else f"{endpoint_url.rstrip('/')}/health"
        )
        if _origin(health_url) != _origin(endpoint_url):
            raise ValueError("HTTP provider health_url must use the endpoint origin")
        if not 0 < self.connect_timeout_seconds <= 3:
            raise ValueError("HTTP provider connect timeout must be in (0, 3]")
        if not 0 < self.overall_timeout_seconds <= 10:
            raise ValueError("HTTP provider overall timeout must be in (0, 10]")
        if self.connect_timeout_seconds > self.overall_timeout_seconds:
            raise ValueError("HTTP provider connect timeout must not exceed overall timeout")
        object.__setattr__(self, "endpoint_url", endpoint_url)
        object.__setattr__(self, "health_url", health_url)


class HttpProviderResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    def json(self) -> Any: ...


class HttpProviderSession(Protocol):
    def post(
        self,
        *,
        url: str,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> HttpProviderResponse: ...

    def get(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> HttpProviderResponse: ...


class HttpNotificationProvider:
    """Versioned custom provider client with explicit delivery idempotency."""

    def __init__(
        self,
        *,
        config: HttpNotificationProviderConfig,
        credential_source: Callable[[], str] | None = None,
        session: HttpProviderSession | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._credential_source = credential_source
        self._session = (
            session if session is not None else cast(HttpProviderSession, httpx.Client())
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
        return self._config.descriptor

    def health(self) -> NotificationProviderHealth:
        if self._config.instance.status == "disabled":
            return NotificationProviderHealth(
                instance_id=self.provider_instance_id,
                status="disabled",
                checked_at=self._clock(),
                error_code=self._error_code("provider_disabled"),
            )
        try:
            response = self._session.get(
                url=self._required_health_url(),
                headers=self._headers(delivery_id=None),
                timeout=self._timeout(),
            )
            ready = 200 <= response.status_code < 300
        except Exception:  # noqa: BLE001
            ready = False
        return NotificationProviderHealth(
            instance_id=self.provider_instance_id,
            status="ready" if ready else "degraded",
            checked_at=self._clock(),
            error_code=None if ready else self._error_code("provider_transport_error"),
        )

    def send(self, *, delivery: NotificationDelivery) -> NotificationProviderResult:
        request_hash = _redacted_request_hash(delivery=delivery)
        if self._config.instance.status == "disabled":
            return self._result(
                status="suppressed",
                error_code="provider_disabled",
                request_hash=request_hash,
            )
        if (
            delivery.provider_instance_id != self.provider_instance_id
            or delivery.provider_key != self.provider_key
            or not self._config.instance.permits(organization_id=delivery.organization_id)
        ):
            return self._result(
                status="dead_letter",
                error_code="provider_scope_mismatch",
                request_hash=request_hash,
            )
        try:
            headers = self._headers(delivery_id=delivery.delivery_id)
        except Exception:  # noqa: BLE001
            return self._result(
                status="dead_letter",
                error_code="provider_secret_unavailable",
                request_hash=request_hash,
            )
        try:
            response = self._session.post(
                url=self._config.endpoint_url,
                json=_delivery_payload(delivery=delivery),
                headers=headers,
                timeout=self._timeout(),
            )
        except (httpx.ConnectTimeout, httpx.ConnectError):
            return self._result(
                status="retry",
                error_code="provider_connect_timeout",
                request_hash=request_hash,
            )
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            return self._result(
                status="unknown",
                error_code="provider_timeout_after_acceptance_possible",
                request_hash=request_hash,
            )
        except httpx.TransportError:
            return self._result(
                status="unknown",
                error_code="provider_transport_error",
                request_hash=request_hash,
            )
        except Exception:  # noqa: BLE001
            return self._result(
                status="unknown",
                error_code="provider_transport_error",
                request_hash=request_hash,
            )

        response_hash = _redacted_response_hash(status_code=response.status_code)
        if 200 <= response.status_code < 300:
            return NotificationProviderResult(
                status="sent",
                provider_message_id=_provider_message_id(response=response),
                redacted_request_hash=request_hash,
                redacted_response_hash=response_hash,
            )
        if response.status_code == 429:
            return self._result(
                status="retry",
                error_code="provider_rate_limited",
                request_hash=request_hash,
                response_hash=response_hash,
                retry_after_seconds=_retry_after_seconds(response=response),
            )
        if response.status_code in {500, 502, 503, 504}:
            return self._result(
                status="retry",
                error_code="provider_http_error",
                request_hash=request_hash,
                response_hash=response_hash,
            )
        return self._result(
            status="dead_letter",
            error_code="provider_http_error",
            request_hash=request_hash,
            response_hash=response_hash,
        )

    def _headers(self, *, delivery_id: UUID | None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Roehub-Provider-Contract": self.descriptor.contract_version,
        }
        if delivery_id is not None:
            headers["X-Roehub-Delivery-Id"] = str(delivery_id)
        if self._credential_source is not None:
            credential = self._credential_source().strip()
            if not credential:
                raise ValueError("provider credential is empty")
            headers["Authorization"] = f"Bearer {credential}"
        return headers

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            self._config.overall_timeout_seconds,
            connect=self._config.connect_timeout_seconds,
        )

    def _required_health_url(self) -> str:
        if self._config.health_url is None:
            raise ValueError("HTTP provider health URL is unavailable")
        return self._config.health_url

    def _error_code(self, preferred: str) -> str:
        if preferred in self.descriptor.error_codes:
            return preferred
        return self.descriptor.error_codes[0]

    def _result(
        self,
        *,
        status: str,
        error_code: str,
        request_hash: str,
        response_hash: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> NotificationProviderResult:
        return NotificationProviderResult(
            status=status,  # type: ignore[arg-type]
            error_code=self._error_code(error_code),
            retry_after_seconds=retry_after_seconds,
            redacted_request_hash=request_hash,
            redacted_response_hash=response_hash,
        )


def _delivery_payload(*, delivery: NotificationDelivery) -> dict[str, object]:
    return {
        "apiVersion": "notifications.roehub.io/v1",
        "kind": "NotificationDelivery",
        "metadata": {
            "delivery_id": str(delivery.delivery_id),
            "organization_id": str(delivery.organization_id),
            "provider_instance_id": str(delivery.provider_instance_id),
        },
        "spec": {
            "channel": delivery.channel_key,
            "template": delivery.template_key,
            "recipient_address_ref": delivery.recipient_address_ref,
            "payload": dict(delivery.rendered_payload_json),
        },
    }


def _redacted_request_hash(*, delivery: NotificationDelivery) -> str:
    value = {
        "delivery_id": str(delivery.delivery_id),
        "organization_id": str(delivery.organization_id),
        "provider_instance_id": str(delivery.provider_instance_id),
        "provider_key": delivery.provider_key,
        "template_key": delivery.template_key,
        "recipient": "<redacted>",
        "payload": "<redacted>",
    }
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _redacted_response_hash(*, status_code: int) -> str:
    return sha256(f"status={status_code}".encode()).hexdigest()


def _provider_message_id(*, response: HttpProviderResponse) -> str | None:
    header_value = response.headers.get("X-Provider-Message-Id")
    if header_value:
        return header_value[:256]
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    if isinstance(payload, Mapping):
        value = payload.get("message_id")
        if isinstance(value, (str, int)):
            return str(value)[:256]
    return None


def _retry_after_seconds(*, response: HttpProviderResponse) -> int | None:
    raw_value = response.headers.get("Retry-After")
    if raw_value is None or not raw_value.isdigit():
        return None
    return min(int(raw_value), 86_400)


def _validated_url(raw: str, *, field: str) -> str:
    normalized = raw.strip()
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"HTTP provider {field} must be HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"HTTP provider {field} must not contain credentials")
    return normalized


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.casefold()
    port = parsed.port if parsed.port is not None else {"http": 80, "https": 443}.get(scheme)
    return scheme, (parsed.hostname or "").casefold(), port
