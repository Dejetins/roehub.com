"""API-side adapter for the redacted operational-health service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx

from apps.monitoring.operational_health import OperationalSnapshot

_BASE_URL_ENV = "ROEHUB_OPERATIONAL_HEALTH_URL"
_TIMEOUT_ENV = "ROEHUB_OPERATIONAL_HEALTH_TIMEOUT_SECONDS"
_DEFAULT_TIMEOUT_SECONDS = 2.0


class OperationalHealthClientError(RuntimeError):
    """Sanitized operational-health transport or contract failure."""


class OperationalHealthClient(Protocol):
    def snapshot(self) -> OperationalSnapshot: ...


@dataclass(frozen=True)
class HttpOperationalHealthClient:
    base_url: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("operational health URL must use HTTP or HTTPS")
        if self.timeout_seconds <= 0:
            raise ValueError("operational health timeout must be positive")

    def snapshot(self) -> OperationalSnapshot:
        try:
            with httpx.Client(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.get("/api/v1/operational-health")
                response.raise_for_status()
                return OperationalSnapshot.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as error:
            raise OperationalHealthClientError(
                "operational health snapshot is unavailable"
            ) from error


def build_operational_health_client_from_environ(
    *, environ: Mapping[str, str]
) -> OperationalHealthClient | None:
    raw_url = environ.get(_BASE_URL_ENV, "").strip()
    if not raw_url:
        return None
    raw_timeout = environ.get(_TIMEOUT_ENV, "").strip()
    try:
        timeout = float(raw_timeout) if raw_timeout else _DEFAULT_TIMEOUT_SECONDS
    except ValueError as error:
        raise ValueError(f"{_TIMEOUT_ENV} must be a number") from error
    return HttpOperationalHealthClient(
        base_url=raw_url,
        timeout_seconds=timeout,
    )


__all__ = [
    "HttpOperationalHealthClient",
    "OperationalHealthClient",
    "OperationalHealthClientError",
    "build_operational_health_client_from_environ",
]
