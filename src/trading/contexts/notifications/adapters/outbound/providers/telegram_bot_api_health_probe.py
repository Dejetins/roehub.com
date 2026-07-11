from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Protocol, cast

import requests


@dataclass(frozen=True, slots=True)
class TelegramApiHealthProbeConfig:
    enabled: bool
    credential: str | None = field(repr=False)
    api_base_url: str = "https://api.telegram.org"
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        api_base_url = self.api_base_url.strip().rstrip("/")
        credential = self.credential.strip() if self.credential is not None else None
        if not api_base_url.startswith(("https://", "http://")):
            raise ValueError("TelegramApiHealthProbeConfig.api_base_url must be HTTP(S)")
        if self.timeout_s <= 0:
            raise ValueError("TelegramApiHealthProbeConfig.timeout_s must be > 0")
        object.__setattr__(self, "api_base_url", api_base_url)
        object.__setattr__(self, "credential", credential)


@dataclass(frozen=True, slots=True)
class TelegramApiHealthProbeResult:
    up: bool
    latency_seconds: float
    error_code: str | None = None


class TelegramHealthHttpResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    def json(self) -> Any: ...


class TelegramHealthHttpSession(Protocol):
    def get(self, *, url: str, timeout: float) -> TelegramHealthHttpResponse: ...


class TelegramBotApiHealthProbe:
    def __init__(
        self,
        *,
        config: TelegramApiHealthProbeConfig,
        session: TelegramHealthHttpSession | None = None,
    ) -> None:
        self._config = config
        self._session = (
            session
            if session is not None
            else cast(TelegramHealthHttpSession, requests.Session())
        )

    def probe(self) -> TelegramApiHealthProbeResult:
        started_at = monotonic()
        if not self._config.enabled:
            return TelegramApiHealthProbeResult(
                up=False,
                latency_seconds=0.0,
                error_code="telegram_probe_disabled",
            )
        if not self._config.credential:
            return TelegramApiHealthProbeResult(
                up=False,
                latency_seconds=0.0,
                error_code="telegram_probe_credential_missing",
            )
        try:
            response = self._session.get(
                url=(
                    f"{self._config.api_base_url}/bot"
                    f"{self._config.credential}/getMe"
                ),
                timeout=self._config.timeout_s,
            )
        except requests.exceptions.Timeout:
            return self._result(started_at=started_at, error_code="telegram_probe_timeout")
        except Exception:  # noqa: BLE001
            return self._result(
                started_at=started_at,
                error_code="telegram_probe_transport_error",
            )

        if response.status_code != 200:
            return self._result(
                started_at=started_at,
                error_code=f"telegram_probe_http_{response.status_code}",
            )
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001
            payload = None
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            return self._result(
                started_at=started_at,
                error_code="telegram_probe_api_error",
            )
        return TelegramApiHealthProbeResult(
            up=True,
            latency_seconds=max(monotonic() - started_at, 0.0),
        )

    @staticmethod
    def _result(*, started_at: float, error_code: str) -> TelegramApiHealthProbeResult:
        return TelegramApiHealthProbeResult(
            up=False,
            latency_seconds=max(monotonic() - started_at, 0.0),
            error_code=error_code,
        )
