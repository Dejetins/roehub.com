from pathlib import Path

import pytest

from trading.contexts.market_data.adapters.outbound.clients.common_http.provider_policy import (
    ProviderPolicy,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)


def test_provider_policy_charges_documented_weights_and_aggregates_markets():
    policy = ProviderPolicy(load_market_data_runtime_config(Path('configs/dev/market_data.yaml')))
    spot = policy.request('https://api.binance.com/api/v3/klines', {'limit': 1000})
    futures = policy.request('https://fapi.binance.com/fapi/v1/klines', {'limit': 1000})
    assert spot.windows[0].cost == 2 and futures.windows[0].cost == 5
    assert (spot.provider, spot.scope, spot.concurrency) == (
        futures.provider, futures.scope, futures.concurrency)
    for count, expected in ((99,1),(100,2),(499,2),(500,5),(1000,5),(1001,10)):
        assert policy.request('https://fapi.binance.com/fapi/v1/klines',
                              {'limit': count}).windows[0].cost == expected
    first = policy.request('https://api.bybit.com/v5/market/kline', {'category': 'spot'})
    second = policy.request('https://api.bybit.com/v5/market/kline', {'category': 'linear'})
    assert first == second and len(first.windows) == 2
    funding = policy.request('https://fapi.binance.com/fapi/v1/fundingRate', {})
    assert any(w.dimension == 'funding' and w.seconds == 300 for w in funding.windows)


def test_unrecognized_endpoint_fails_closed():
    policy = ProviderPolicy(load_market_data_runtime_config(Path('configs/dev/market_data.yaml')))
    with pytest.raises(ValueError, match='provider_endpoint_not_supported'):
        policy.request('https://api.binance.com/api/v3/order', {})
    with pytest.raises(ValueError, match='provider_endpoint_not_configured'):
        policy.request('https://unknown.invalid/api/v3/klines', {})

    with pytest.raises(ValueError, match='provider_endpoint_not_configured'):
        policy.request('http://api.binance.com/api/v3/klines', {})
