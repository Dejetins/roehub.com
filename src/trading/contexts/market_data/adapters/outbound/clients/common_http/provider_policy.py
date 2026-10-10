"""Public market-data policies; unknown endpoints fail closed.

Conservative installation ceilings, not exchange capacity claims. All supported
Bybit public markets share the same IP quota; Binance product dimensions share
one concurrency/cooldown scope. See the runbook for primary documentation.
"""

from typing import Any, Mapping
from urllib.parse import urlsplit

from trading.contexts.market_data.application.ports.provider_admission import (
    ProviderRequest,
    QuotaWindow,
)


class ProviderPolicy:
    def __init__(self, config):
        self.markets = config.markets
        for provider in {m.exchange for m in self.markets}:
            scopes = {m.rest.limiter.egress_scope for m in self.markets if m.exchange == provider}
            if len(scopes) != 1 or not next(iter(scopes)):
                raise ValueError("one_explicit_egress_scope_per_provider_required")

    def request(self, url: str, params: Mapping[str, Any]) -> ProviderRequest:
        target = urlsplit(url)
        markets = [
            m for m in self.markets
            if (urlsplit(m.rest.base_url).scheme, urlsplit(m.rest.base_url).netloc)
            == (target.scheme, target.netloc)
        ]
        if not markets:
            raise ValueError("provider_endpoint_not_configured")
        market = markets[0]
        limiter = market.rest.limiter
        if not limiter.egress_scope:
            raise ValueError("provider_egress_scope_required")
        factor = min(m.rest.limiter.safety_factor for m in markets)
        concurrency = min(
            m.rest.limiter.max_concurrency for m in self.markets if m.exchange == market.exchange
        )
        windows = []
        path = target.path
        if market.exchange == "binance":
            if path == "/api/v3/klines":
                weight, product = 2, "spot"
            elif path == "/api/v3/exchangeInfo":
                weight, product = 20, "spot"
            elif path == "/fapi/v1/klines":
                limit = int(params.get("limit", 500))
                weight, product = (
                    (1 if limit < 100 else 2 if limit < 500 else 5 if limit <= 1000 else 10),
                    "futures",
                )
            elif path == "/fapi/v1/exchangeInfo":
                weight, product = 1, "futures"
            elif path in {"/fapi/v1/fundingRate", "/fapi/v1/fundingInfo"}:
                weight, product = 1, "futures"
                windows.append(QuotaWindow("funding", 300, int(500 * factor), 1))
            else:
                raise ValueError("provider_endpoint_not_supported")
            # 1200/min is a conservative ceiling for both product families.
            windows += [
                QuotaWindow(product + "_weight", 60, int(1200 * factor), weight),
                QuotaWindow(product + "_raw", 300, int(6000 * factor), 1),
            ]
        elif path in {
            "/v5/market/kline",
            "/v5/market/instruments-info",
            "/v5/market/funding/history",
        }:
            # One-second ceiling also protects against small bursts near bucket edges.
            windows += [
                QuotaWindow("requests", 5, int(300 * factor), 1),
                QuotaWindow("requests", 1, max(1, int(50 * factor)), 1),
            ]
        else:
            raise ValueError("provider_endpoint_not_supported")
        return ProviderRequest(market.exchange, limiter.egress_scope, concurrency, tuple(windows))
