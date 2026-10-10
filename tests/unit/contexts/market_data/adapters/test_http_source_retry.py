from types import SimpleNamespace

import pytest

from trading.contexts.market_data.adapters.outbound.clients.common_http.http_client import (
    RequestsHttpClient,
    retry_after_seconds,
)
from trading.contexts.market_data.application.services.source_retry import TemporarySourceError


@pytest.fixture
def harness(monkeypatch):
    sent, reserved, completed = [], [], []
    responses = []
    permit = SimpleNamespace(check=lambda: None, complete=lambda **kw: completed.append(kw))
    def reserve(policy):
        reserved.append(policy)
        return permit
    def get(**kwargs):
        sent.append(1)
        return responses.pop(0)
    http = RequestsHttpClient(admission=SimpleNamespace(reserve=reserve),
                              policy=SimpleNamespace(request=lambda *_: SimpleNamespace(
                                  provider='binance')))
    monkeypatch.setattr(http, '_transport', lambda *_: SimpleNamespace(get=get))
    monkeypatch.setattr(http, '_return_transport', lambda _: None)
    def call():
        return http.get_json(url='https://source.invalid/candles', params={}, timeout_s=1,
            retries=2, backoff_base_s=0, backoff_max_s=0, backoff_jitter_s=0)
    yield SimpleNamespace(call=call, responses=responses, sent=sent, http=http,
                          reserved=reserved, completed=completed)
    http.close()


def test_network_loss_retries_reacquire_and_never_refund_sent_quota(harness):
    harness.responses.extend([('temporary',)] * 3)
    with pytest.raises(TemporarySourceError, match='source_unavailable'):
        harness.call()
    assert len(harness.sent) == len(harness.reserved) == len(harness.completed) == 3


@pytest.mark.parametrize('status,body,headers,code,after', [
    (429, {}, {'Retry-After': '120'}, 'source_rate_limited', 120),
    (503, {}, {'Retry-After': '30'}, 'source_unavailable', 30),
    (200, {'retCode': 10006}, {'Retry-After': '60'}, 'source_rate_limited', 60),
    (200, {'retCode': 10016}, {}, 'source_unavailable', 0),
])
def test_temporary_codes_and_shared_cooldown(harness, status, body, headers, code, after):
    harness.responses.extend([('ok', status, headers, body)] * 3)
    with pytest.raises(TemporarySourceError) as caught:
        harness.call()
    assert caught.value.code == code and caught.value.retry_after_s == after
    assert harness.completed[-1]['cooldown_s'] == after


def test_permanent_http_error_is_not_retried_and_payload_is_not_exposed(harness):
    harness.responses.append(('ok', 400, {}, 'sensitive-payload'))
    with pytest.raises(RuntimeError, match='^source_http_400$'):
        harness.call()
    assert len(harness.sent) == 1
    assert all(retry_after_seconds(v) == 0 for v in ('invalid', 'nan', '-5'))


def test_invalid_json_has_bounded_retries_without_endless_job_retry(harness):
    harness.responses.extend([('ok', 200, {}, None)]*2 + [('ok', 200, {}, [])])
    assert harness.call().body == [] and len(harness.sent) == 3
    harness.responses.extend([('ok', 200, {}, None)]*3)
    with pytest.raises(RuntimeError, match='^source_invalid_json$'):
        harness.call()


def test_unwired_http_client_fails_closed():
    http = RequestsHttpClient()
    try:
        with pytest.raises(RuntimeError, match='shared_provider_admission_required'):
            http.get_json(url='https://source.invalid', params={}, timeout_s=1, retries=0,
                          backoff_base_s=0, backoff_max_s=0, backoff_jitter_s=0)
    finally:
        http.close()


def test_bybit_ban_preserves_longer_retry_after(harness):
    harness.http.policy = SimpleNamespace(request=lambda *_: SimpleNamespace(provider='bybit'))
    harness.responses.append(('ok', 403, {'Retry-After': '1200'}, {}))
    with pytest.raises(TemporarySourceError) as caught:
        harness.call()
    assert caught.value.retry_after_s == 1200
    assert harness.completed[-1]['cooldown_s'] == 1200 and len(harness.sent) == 1
