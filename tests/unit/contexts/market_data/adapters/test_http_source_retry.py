from types import SimpleNamespace

import pytest
import requests

from trading.contexts.market_data.adapters.outbound.clients.common_http.http_client import (
    RequestsHttpClient,
    retry_after_seconds,
)
from trading.contexts.market_data.application.services.source_retry import TemporarySourceError


def call():
    return RequestsHttpClient().get_json(url='https://source.invalid/candles', params={},
        timeout_s=1, retries=2, backoff_base_s=0, backoff_max_s=0, backoff_jitter_s=0)


@pytest.mark.parametrize("failure", [requests.ConnectionError,
                                     requests.exceptions.ChunkedEncodingError])
def test_network_loss_is_classified_after_bounded_transport_retries(monkeypatch, failure):
    calls = []
    def get(*args, **kwargs):
        calls.append(1)
        raise failure('sensitive-url')
    monkeypatch.setattr(requests, 'get', get)
    with pytest.raises(TemporarySourceError, match='source_unavailable'):
        call()
    assert len(calls) == 3


@pytest.mark.parametrize('status,body,headers,code,after', [
    (429, {}, {'Retry-After':'120'}, 'source_rate_limited', 120),
    (503, {}, {'Retry-After':'30'}, 'source_unavailable', 30),
    (200, {'retCode':10006}, {'Retry-After':'60'}, 'source_rate_limited', 60),
    (200, {'retCode':10016}, {}, 'source_unavailable', 0),
])
def test_temporary_codes_and_cooldown(monkeypatch, status, body, headers, code, after):
    monkeypatch.setattr(requests, 'get', lambda *a, **kw: SimpleNamespace(
        status_code=status, headers=headers, json=lambda: body))
    with pytest.raises(TemporarySourceError) as caught:
        call()
    assert caught.value.code == code and caught.value.retry_after_s == after


def test_permanent_http_error_is_not_retried_and_payload_is_not_exposed(monkeypatch):
    calls = []
    def get(*a, **kw):
        calls.append(1)
        return SimpleNamespace(status_code=400, headers={}, text='sensitive-payload')
    monkeypatch.setattr(requests, 'get', get)
    with pytest.raises(RuntimeError, match='^source_http_400$'):
        call()
    assert len(calls) == 1
    assert retry_after_seconds('invalid') == 0
    assert retry_after_seconds('nan') == 0
    assert retry_after_seconds('-5') == 0


def test_invalid_json_uses_bounded_transport_retry_without_endless_job_retry(monkeypatch):
    calls = []
    def body():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("private-body")
        return []
    monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(
        status_code=200, headers={}, json=body))
    assert call().body == [] and len(calls) == 3
    def invalid():
        raise ValueError("private-body")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(
        status_code=200, headers={}, json=invalid))
    with pytest.raises(RuntimeError, match="^source_invalid_json$"):
        call()
