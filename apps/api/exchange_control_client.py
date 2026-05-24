from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx

INTERNAL_SERVICE_HEADER = "X-Roehub-Internal-Service"
REQUEST_ID_HEADER = "X-Request-Id"
INTERNAL_SERVICE_NAME = "apps/api"

_BASE_URL_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL"
_TOKEN_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN"
_ROUTES_ENABLED_ENV = "ROEHUB_EXCHANGE_CONNECTIONS_PUBLIC_ROUTES_ENABLED"
_TIMEOUT_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_TIMEOUT_SECONDS"
_DEFAULT_TIMEOUT_SECONDS = 2.0


class ExchangeControlClientError(RuntimeError):
    """Sanitized exchange-control client error safe for API logs."""


@dataclass(frozen=True)
class ExchangeControlCapabilities:
    service: str
    service_identity: str
    contract_version: str
    capabilities: tuple[str, ...]


class ExchangeControlClient(Protocol):
    def get_capabilities(
        self, *, request_id: str | None = None
    ) -> ExchangeControlCapabilities: ...


@dataclass(frozen=True)
class ExchangeControlClientConfig:
    base_url: str | None
    internal_api_token: str | None
    timeout_seconds: float
    public_routes_enabled: bool

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "ExchangeControlClientConfig":
        config = cls(
            base_url=_read_optional_str(environ.get(_BASE_URL_ENV)),
            internal_api_token=_read_optional_str(environ.get(_TOKEN_ENV)),
            timeout_seconds=_read_positive_float(
                value=environ.get(_TIMEOUT_ENV),
                default=_DEFAULT_TIMEOUT_SECONDS,
                name=_TIMEOUT_ENV,
            ),
            public_routes_enabled=_read_bool(environ.get(_ROUTES_ENABLED_ENV)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.public_routes_enabled:
            return
        if not self.base_url:
            raise ValueError(
                "exchange connection routes require "
                "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL"
            )
        if not self.internal_api_token:
            raise ValueError(
                "exchange connection routes require ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN"
            )

    def build_client(self) -> ExchangeControlClient | None:
        if not self.base_url or not self.internal_api_token:
            return None
        return HttpExchangeControlClient(
            base_url=self.base_url,
            internal_api_token=self.internal_api_token,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True)
class HttpExchangeControlClient:
    base_url: str
    internal_api_token: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("exchange-control internal base URL is required")
        if not self.internal_api_token.strip():
            raise ValueError("exchange-control internal API token is required")
        if self.timeout_seconds <= 0:
            raise ValueError("exchange-control internal timeout must be positive")

    def get_capabilities(self, *, request_id: str | None = None) -> ExchangeControlCapabilities:
        effective_request_id = request_id or f"apps-api-{uuid.uuid4()}"
        try:
            with httpx.Client(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.get(
                    "/internal/v1/capabilities",
                    headers={
                        "Authorization": f"Bearer {self.internal_api_token}",
                        INTERNAL_SERVICE_HEADER: INTERNAL_SERVICE_NAME,
                        REQUEST_ID_HEADER: effective_request_id,
                    },
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ExchangeControlClientError(
                f"exchange-control internal request failed with status {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ExchangeControlClientError("exchange-control internal request failed") from exc
        payload = response.json()
        return _capabilities_from_payload(payload)


@dataclass(frozen=True)
class InMemoryExchangeControlClient:
    capabilities: ExchangeControlCapabilities = ExchangeControlCapabilities(
        service="exchange-control",
        service_identity="exchange-control",
        contract_version="internal-v1",
        capabilities=("capabilities.read",),
    )

    def get_capabilities(self, *, request_id: str | None = None) -> ExchangeControlCapabilities:
        _ = request_id
        return self.capabilities


def build_exchange_control_client_from_environ(
    *,
    environ: Mapping[str, str],
) -> ExchangeControlClient | None:
    return ExchangeControlClientConfig.from_environ(environ).build_client()


def _capabilities_from_payload(payload: object) -> ExchangeControlCapabilities:
    if not isinstance(payload, dict):
        raise ExchangeControlClientError("exchange-control internal response is invalid")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(item, str) and item for item in capabilities
    ):
        raise ExchangeControlClientError("exchange-control capabilities response is invalid")
    service = payload.get("service")
    service_identity = payload.get("service_identity")
    contract_version = payload.get("contract_version")
    if not isinstance(service, str) or not isinstance(service_identity, str):
        raise ExchangeControlClientError("exchange-control capabilities response is invalid")
    if not isinstance(contract_version, str):
        raise ExchangeControlClientError("exchange-control capabilities response is invalid")
    return ExchangeControlCapabilities(
        service=service,
        service_identity=service_identity,
        contract_version=contract_version,
        capabilities=tuple(capabilities),
    )


def _read_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _read_bool(value: str | None) -> bool:
    if value is None or value == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_positive_float(*, value: str | None, default: float, name: str) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


__all__ = [
    "INTERNAL_SERVICE_HEADER",
    "INTERNAL_SERVICE_NAME",
    "REQUEST_ID_HEADER",
    "ExchangeControlCapabilities",
    "ExchangeControlClient",
    "ExchangeControlClientConfig",
    "ExchangeControlClientError",
    "HttpExchangeControlClient",
    "InMemoryExchangeControlClient",
    "build_exchange_control_client_from_environ",
]
