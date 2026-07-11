from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Protocol, cast

import requests

from trading.contexts.notifications.application.ports import NotificationProviderResult
from trading.contexts.notifications.domain import NotificationDelivery


@dataclass(frozen=True, slots=True)
class TelegramNotificationProviderConfig:
    enabled: bool
    credential: str | None = field(repr=False)
    api_base_url: str = "https://api.telegram.org"
    send_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        api_base_url = self.api_base_url.strip().rstrip("/")
        if not api_base_url.startswith(("https://", "http://")):
            raise ValueError(
                "TelegramNotificationProviderConfig.api_base_url must start with http:// or https://"
            )
        if self.send_timeout_s <= 0:
            raise ValueError("TelegramNotificationProviderConfig.send_timeout_s must be > 0")
        credential = self.credential.strip() if self.credential is not None else None
        if self.enabled and not credential:
            raise ValueError(
                "TelegramNotificationProviderConfig.credential must be present when enabled"
            )
        object.__setattr__(self, "api_base_url", api_base_url)
        object.__setattr__(self, "credential", credential)


class TelegramHttpResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    def json(self) -> Any: ...

    @property
    def text(self) -> str: ...


class TelegramHttpSession(Protocol):
    def post(
        self, *, url: str, json: Mapping[str, str], timeout: float
    ) -> TelegramHttpResponse: ...


class TelegramBotApiNotificationProvider:
    provider_key = "telegram_bot_api"

    def __init__(
        self,
        *,
        config: TelegramNotificationProviderConfig,
        session: TelegramHttpSession | None = None,
    ) -> None:
        self._config = config
        self._session = (
            session
            if session is not None
            else cast(TelegramHttpSession, requests.Session())
        )

    def send(self, *, delivery: NotificationDelivery) -> NotificationProviderResult:
        request_hash = _redacted_request_hash(delivery=delivery)
        if not self._config.enabled:
            return NotificationProviderResult(
                status="suppressed",
                error_code="telegram_provider_disabled",
                redacted_request_hash=request_hash,
            )

        try:
            response = self._session.post(
                url=_send_message_url(config=self._config),
                json={
                    "chat_id": delivery.recipient_address_ref,
                    "text": str(delivery.rendered_payload_json.get("text", ""))[:4096],
                },
                timeout=self._config.send_timeout_s,
            )
        except requests.exceptions.Timeout:
            return NotificationProviderResult(
                status="unknown",
                error_code="telegram_timeout",
                redacted_request_hash=request_hash,
            )
        except Exception:  # noqa: BLE001
            return NotificationProviderResult(
                status="unknown",
                error_code="telegram_transport_error",
                redacted_request_hash=request_hash,
            )

        response_hash = _redacted_response_hash(response=response)
        if response.status_code == 200:
            payload = _parse_json_object(response=response)
            if payload.get("ok") is True:
                return NotificationProviderResult(
                    status="sent",
                    provider_message_id=_provider_message_id(payload=payload),
                    redacted_request_hash=request_hash,
                    redacted_response_hash=response_hash,
                )
            return NotificationProviderResult(
                status="dead_letter",
                error_code="telegram_api_error",
                redacted_request_hash=request_hash,
                redacted_response_hash=response_hash,
            )
        if response.status_code == 429:
            retry_after = _retry_after_seconds(response=response)
            return NotificationProviderResult(
                status="retry",
                error_code="telegram_rate_limited",
                retry_after_seconds=retry_after,
                redacted_request_hash=request_hash,
                redacted_response_hash=response_hash,
            )
        if response.status_code in {500, 502, 503, 504}:
            return NotificationProviderResult(
                status="unknown",
                error_code=f"telegram_http_{response.status_code}",
                redacted_request_hash=request_hash,
                redacted_response_hash=response_hash,
            )
        return NotificationProviderResult(
            status="dead_letter",
            error_code=f"telegram_http_{response.status_code}",
            redacted_request_hash=request_hash,
            redacted_response_hash=response_hash,
        )


def _send_message_url(*, config: TelegramNotificationProviderConfig) -> str:
    return f"{config.api_base_url}/bot{config.credential}/sendMessage"


def _redacted_request_hash(*, delivery: NotificationDelivery) -> str:
    payload = {
        "provider_key": "telegram_bot_api",
        "route_id": str(delivery.route_id),
        "template_key": delivery.template_key,
        "recipient_address_ref": "<redacted>",
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _redacted_response_hash(*, response: TelegramHttpResponse) -> str:
    payload = {
        "status_code": response.status_code,
        "body_excerpt": _safe_response_excerpt(response=response),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _safe_response_excerpt(*, response: TelegramHttpResponse) -> str:
    text = response.text
    if not text:
        return ""
    return text[:300]


def _parse_json_object(*, response: TelegramHttpResponse) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _retry_after_seconds(*, response: TelegramHttpResponse) -> int | None:
    payload = _parse_json_object(response=response)
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
    if isinstance(message_id, int | str):
        return str(message_id)
    return None
