from types import SimpleNamespace
from typing import Any

import pytest

from trading.contexts.market_data.adapters.outbound.clients.common_http.http_client import (
    HttpResponse,
)
from trading.contexts.market_data.adapters.outbound.clients.history_probe_http import (
    HistoryProbeHttpClient,
)

ARGS: dict[str, Any] = dict(
    url="https://exchange.invalid/klines",
    params={},
    timeout_s=10,
    retries=5,
    backoff_base_s=1,
    backoff_max_s=5,
    backoff_jitter_s=1,
)


def stub(**values: Any) -> Any:
    return SimpleNamespace(**values)


def test_probe_retries_one_read_without_replaying_commands_or_exposing_provider_details():
    calls = []
    response = HttpResponse(status_code=200, headers={}, body=[])

    def get_json(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("private-provider-detail")
        return response

    probe = HistoryProbeHttpClient(stub(get_json=get_json))
    assert probe.get_json(**ARGS) == response
    assert len(calls) == 2 and all(c["retries"] == 0 and c["timeout_s"] <= 5 for c in calls)


def test_probe_deadline_stops_the_retry_and_redacts_failures(monkeypatch):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("private-provider-detail")

    times = iter([0, 2])
    monkeypatch.setattr(
        "trading.contexts.market_data.adapters.outbound.clients.history_probe_http.monotonic",
        lambda: next(times),
    )
    probe = HistoryProbeHttpClient(stub(get_json=fail), budget_seconds=1, started=0)
    with pytest.raises(RuntimeError, match="timed out") as error:
        probe.get_json(**ARGS)
    assert len(calls) == 1 and "private-provider-detail" not in str(error.value)
