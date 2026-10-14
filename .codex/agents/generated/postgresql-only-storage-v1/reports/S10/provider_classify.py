"""Sanitized HTTP denial classification; never persist response bodies."""

import json

import requests

urls = {
    "binance_spot": "https://api.binance.com/api/v3/exchangeInfo",
    "binance_futures": "https://fapi.binance.com/fapi/v1/exchangeInfo",
    "bybit_spot": "https://api.bybit.com/v5/market/instruments-info?category=spot",
    "bybit_futures": "https://api.bybit.com/v5/market/instruments-info?category=linear",
}
for provider, url in urls.items():
    try:
        response = requests.get(url, timeout=10)
        body = response.text.lower()
        category = "unspecified_access_denial"
        if "restricted location" in body or "country" in body or "geographic" in body:
            category = "response_indicates_location_policy"
        elif "proxy" in body or "network policy" in body:
            category = "response_indicates_network_policy"
        elif "rate limit" in body or response.status_code == 429:
            category = "response_indicates_rate_limit"
        elif "api key" in body or "authentication" in body:
            category = "response_indicates_authentication"
        print(
            json.dumps(
                {
                    "provider": provider,
                    "http_status": response.status_code,
                    "classification": category,
                    "transport": "http_response_received",
                    "raw_response_saved": False,
                }
            ),
            flush=True,
        )
    except requests.RequestException as error:
        print(json.dumps({"provider": provider, "exception": type(error).__name__}), flush=True)
