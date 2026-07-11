from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from trading.contexts.notifications.adapters import (
    TelegramApiHealthProbeConfig,
    TelegramBotApiHealthProbe,
)


def test_telegram_api_health_probe_uses_get_me_without_exposing_credential() -> None:
    credential = "test-credential"
    session = _Session(response=_Response(status_code=200, payload={"ok": True}))
    config = TelegramApiHealthProbeConfig(
        enabled=True,
        credential=credential,
        timeout_s=3.0,
    )
    probe = TelegramBotApiHealthProbe(
        config=config,
        session=session,
    )

    result = probe.probe()

    assert result.up is True
    assert result.error_code is None
    assert result.latency_seconds >= 0
    assert session.url is not None
    assert session.url.endswith(f"/bot{credential}/getMe")
    assert session.timeout == 3.0
    assert credential not in repr(result)
    assert credential not in repr(config)


def test_telegram_api_health_probe_normalizes_network_failures() -> None:
    probe = TelegramBotApiHealthProbe(
        config=TelegramApiHealthProbeConfig(
            enabled=True,
            credential="test-credential",
        ),
        session=_TimeoutSession(),
    )

    result = probe.probe()

    assert result.up is False
    assert result.error_code == "telegram_probe_timeout"


def test_telegram_api_health_probe_reports_missing_credential_without_crashing() -> None:
    probe = TelegramBotApiHealthProbe(
        config=TelegramApiHealthProbeConfig(enabled=True, credential=None),
        session=_TimeoutSession(),
    )

    result = probe.probe()

    assert result.up is False
    assert result.error_code == "telegram_probe_credential_missing"


@dataclass(frozen=True, slots=True)
class _Response:
    status_code: int
    payload: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.payload


@dataclass(slots=True)
class _Session:
    response: _Response
    url: str | None = None
    timeout: float | None = None

    def get(self, *, url: str, timeout: float) -> _Response:
        self.url = url
        self.timeout = timeout
        return self.response


class _TimeoutSession:
    def get(self, *, url: str, timeout: float) -> _Response:
        _ = url, timeout
        raise requests.exceptions.Timeout
