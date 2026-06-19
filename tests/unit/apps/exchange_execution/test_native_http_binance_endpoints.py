from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from apps.exchange_execution.adapters import native_http
from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeExecutionCredential,
    ExchangeOrderCommand,
)
from trading.shared_kernel.primitives import UserId


def test_binance_testnet_adapter_routes_spot_and_futures_to_demo_endpoints() -> None:
    spot = _command(market_type="spot")
    futures = _command(market_type="futures")

    assert native_http._binance_base_and_order_path(spot) == (
        "https://demo-api.binance.com",
        "/api/v3/order",
    )
    assert native_http._binance_base_and_order_path(futures) == (
        "https://demo-fapi.binance.com",
        "/fapi/v1/order",
    )
    assert native_http._binance_base_and_listen_key_path(market_type="futures") == (
        "https://demo-fapi.binance.com",
        "/fapi/v1/listenKey",
    )


def test_binance_spot_private_stream_degrades_without_deprecated_listen_key() -> None:
    adapter = native_http.BinanceTestnetOrderAdapter()
    session = adapter.ensure_private_stream_session(
        connection=ExchangeExecutionConnection(
            connection_id=uuid4(),
            owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000013001"),
            exchange_name="binance",
            market_type="spot",
            environment="testnet",
            connection_readiness="ready_for_trading",
            effective_capability="trading",
            credential=ExchangeExecutionCredential(
                api_key="api-key",
                api_secret="api-secret",
            ),
        )
    )

    assert session.status == "degraded"
    assert session.status_reason == "binance_spot_rest_user_stream_deprecated"
    assert session.keepalive_at is None
    assert session.expires_at is None


def test_binance_order_params_use_plain_decimal_strings() -> None:
    futures = _command(market_type="futures")

    params = native_http._binance_order_params(futures)

    assert params["quantity"] == "0.001"
    assert params["price"] == "10000"
    assert "E" not in str(params["quantity"])
    assert "E" not in str(params["price"])


def test_bybit_order_params_use_plain_decimal_strings() -> None:
    futures = _command(market_type="futures")

    params = native_http._bybit_order_params(futures)

    assert params["qty"] == "0.001"
    assert params["price"] == "10000"
    assert "E" not in str(params["qty"])
    assert "E" not in str(params["price"])


def _command(*, market_type: str) -> ExchangeOrderCommand:
    return ExchangeOrderCommand(
        intent_id=uuid4(),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000013001"),
        exchange_connection_id=uuid4(),
        exchange_name="binance",
        environment="testnet",
        market_type=market_type,
        instrument_key=f"binance:{market_type}:BTCUSDT",
        side="buy",
        order_type="limit",
        quantity=Decimal("0.001"),
        quote_notional=None,
        limit_price=Decimal("10000"),
        client_order_id="test-client-order",
    )
