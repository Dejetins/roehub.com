from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from trading.contexts.exchange_control.adapters.outbound import exchange_account_state
from trading.contexts.exchange_control.adapters.outbound.exchange_account_state import (
    HttpExchangeAccountStateReader,
)
from trading.contexts.exchange_control.application.account_state import (
    ExchangeAccountStateReadRequest,
)
from trading.contexts.exchange_control.application.validation import (
    ExchangeCredentialPlaintext,
)


def test_binance_futures_reader_uses_read_only_account_config_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_get_json_any(
        *, url: str, headers: dict[str, str], timeout_seconds: float
    ) -> Any:
        _ = timeout_seconds
        calls.append(url)
        if "/fapi/v2/account" in url:
            assert headers["X-MBX-APIKEY"] == "api-key"
            assert "signature=" in url
            return {
                "assets": [
                    {
                        "asset": "USDT",
                        "walletBalance": "100",
                        "availableBalance": "80",
                    }
                ]
            }
        if "/fapi/v2/positionRisk" in url:
            return [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0",
                    "entryPrice": "0",
                    "leverage": "1",
                    "marginType": "isolated",
                    "positionSide": "BOTH",
                }
            ]
        if "/fapi/v1/openOrders" in url:
            return []
        if "/fapi/v1/exchangeInfo" in url:
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                            {
                                "filterType": "LOT_SIZE",
                                "stepSize": "0.001",
                                "minQty": "0.001",
                            },
                            {"filterType": "MIN_NOTIONAL", "notional": "50"},
                        ],
                    }
                ]
            }
        raise AssertionError(f"unexpected Binance URL: {url}")

    monkeypatch.setattr(exchange_account_state, "_get_json_any", fake_get_json_any)

    result = HttpExchangeAccountStateReader().read_account_state(
        request=ExchangeAccountStateReadRequest(
            exchange_name="binance",
            market_type="futures",
            environment="testnet",
            credential=ExchangeCredentialPlaintext(
                api_key="api-key",
                api_secret="api-secret",
            ),
            instrument_keys=("binance:futures:BTCUSDT",),
        ),
        now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC),
    )

    assert result.exchange_name == "binance"
    assert result.market_type == "futures"
    assert result.environment == "testnet"
    assert result.balances[0].asset == "USDT"
    assert result.balances[0].free == Decimal("80")
    assert result.positions[0].instrument_key == "binance:futures:BTCUSDT"
    assert result.positions[0].margin_mode == "isolated"
    assert result.positions[0].leverage == Decimal("1")
    assert result.instrument_filters[0].min_notional == Decimal("50")
    assert all("/leverage" not in call and "/marginType" not in call for call in calls)


def test_bybit_futures_reader_normalizes_config_guard_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_json(
        *, url: str, headers: dict[str, str], timeout_seconds: float
    ) -> dict[str, Any]:
        _ = headers, timeout_seconds
        if "/v5/account/wallet-balance" in url:
            return {
                "retCode": 0,
                "result": {
                    "list": [
                        {
                            "coin": [
                                {
                                    "coin": "USDT",
                                    "walletBalance": "100",
                                    "availableToWithdraw": "95",
                                    "locked": "0",
                                }
                            ]
                        }
                    ]
                },
            }
        if "/v5/order/realtime" in url:
            return {"retCode": 0, "result": {"list": []}}
        if "/v5/position/list" in url:
            return {
                "retCode": 0,
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "side": "Buy",
                            "size": "0",
                            "avgPrice": "0",
                            "leverage": "1",
                            "tradeMode": 1,
                            "positionIdx": 0,
                        }
                    ]
                },
            }
        if "/v5/market/instruments-info" in url:
            return {
                "retCode": 0,
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "priceFilter": {"tickSize": "0.1"},
                            "lotSizeFilter": {
                                "basePrecision": "0.001",
                                "minOrderQty": "0.001",
                                "minOrderAmt": "5",
                            },
                            "leverageFilter": {"maxLeverage": "100"},
                        }
                    ]
                },
            }
        raise AssertionError(f"unexpected Bybit URL: {url}")

    monkeypatch.setattr(exchange_account_state, "_get_json", fake_get_json)

    result = HttpExchangeAccountStateReader().read_account_state(
        request=ExchangeAccountStateReadRequest(
            exchange_name="bybit",
            market_type="futures",
            environment="testnet",
            credential=ExchangeCredentialPlaintext(
                api_key="api-key",
                api_secret="api-secret",
            ),
            instrument_keys=("bybit:futures:BTCUSDT",),
        ),
        now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC),
    )

    assert result.exchange_name == "bybit"
    assert result.positions[0].margin_mode == "isolated"
    assert result.positions[0].position_mode == "one_way"
    assert result.positions[0].leverage == Decimal("1")
    assert result.instrument_filters[0].step_size == Decimal("0.001")
