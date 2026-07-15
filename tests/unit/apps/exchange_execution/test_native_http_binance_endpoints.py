from __future__ import annotations

import hashlib
import io
from decimal import Decimal
from email.message import Message
from pathlib import Path
from uuid import uuid4

import pytest

from apps.exchange_execution.adapters import emulator, native_http
from trading.contexts.live_execution.application.ports import ExchangeOrderAdapterError
from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeExecutionCredential,
    ExchangeOrderCommand,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId.from_string("00000000-0000-4000-8000-000000013000")


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
            organization_id=_ORGANIZATION_ID,
            owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000013001"),
            exchange_name="binance",
            market_type="spot",
            environment="testnet",
            connection_readiness="ready_for_trading",
            effective_capability="trading",
            secret_reference_hash="4" * 64,
            account_revision_hash="3" * 64,
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


@pytest.mark.parametrize(
    "failure", ("os_error", "http_2013", "http_408", "http_503", "invalid_json")
)
def test_submit_transport_ambiguity_is_always_unknown_state(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"not-json"

    def _urlopen(*_args: object, **_kwargs: object):
        if failure == "os_error":
            raise OSError("synthetic transport failure")
        if failure in {"http_2013", "http_408", "http_503"}:
            raise native_http.urllib.error.HTTPError(
                "https://example.invalid",
                400 if failure == "http_2013" else 408 if failure == "http_408" else 503,
                "unavailable",
                Message(),
                io.BytesIO(b'{"code":-2013}' if failure == "http_2013" else b"{}"),
            )
        return _Response()

    monkeypatch.setattr(native_http.urllib.request, "urlopen", _urlopen)
    with pytest.raises(ExchangeOrderAdapterError) as error_info:
        native_http.BinanceTestnetOrderAdapter().submit_order(
            command=_command(market_type="spot"),
            credential=ExchangeExecutionCredential(
                api_key="<redacted>",
                api_secret="<redacted>",
            ),
        )
    assert error_info.value.unknown_state is True


def test_bybit_submit_application_error_is_unknown_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_http.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _JsonResponse(
            b'{"retCode":10006,"retMsg":"synthetic rate limit","result":{}}'
        ),
    )

    with pytest.raises(ExchangeOrderAdapterError) as error_info:
        native_http.BybitTestnetOrderAdapter().submit_order(
            command=_command(market_type="spot"),
            credential=ExchangeExecutionCredential(
                api_key="<redacted>",
                api_secret="<redacted>",
            ),
        )

    assert error_info.value.reason == "exchange_ret_code_10006"
    assert error_info.value.unknown_state is True


def test_binance_order_not_found_is_terminal_only_for_client_order_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _not_found(*_args: object, **_kwargs: object):
        raise native_http.urllib.error.HTTPError(
            "https://example.invalid",
            400,
            "not found",
            Message(),
            io.BytesIO(b'{"code":-2013}'),
        )

    monkeypatch.setattr(native_http.urllib.request, "urlopen", _not_found)
    result = (
        native_http.BinanceTestnetOrderAdapter().get_order_status_by_client_order_id(
            command=_command(market_type="spot"),
            client_order_id="missing-order",
            credential=ExchangeExecutionCredential(
                api_key="<redacted>",
                api_secret="<redacted>",
            ),
        )
    )

    assert result.lookup_outcome == "confirmed_absent"


def test_native_adapter_revision_hashes_bind_loaded_module_bytes() -> None:
    module_bytes = Path(native_http.__file__).read_bytes()
    expected_binance = hashlib.sha256(
        b"core:binance-testnet\0" + module_bytes
    ).hexdigest()
    expected_bybit = hashlib.sha256(b"core:bybit-testnet\0" + module_bytes).hexdigest()

    assert native_http.BinanceTestnetOrderAdapter().revision_hash == expected_binance
    assert native_http.BybitTestnetOrderAdapter().revision_hash == expected_bybit
    emulator_bytes = Path(emulator.__file__).read_bytes()
    expected_emulator = hashlib.sha256(
        b"core:exchange-emulator\0" + emulator_bytes
    ).hexdigest()
    assert emulator.ExchangeExecutionEmulatorAdapter(
        exchange_name="binance"
    ).revision_hash == expected_emulator


def test_bybit_empty_client_order_lookup_is_typed_confirmed_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_http,
        "_bybit_signed_json",
        lambda **_kwargs: {"retCode": 0, "result": {"list": []}},
    )
    result = native_http.BybitTestnetOrderAdapter().get_order_status_by_client_order_id(
        command=_command(market_type="spot"),
        client_order_id="missing-order",
        credential=ExchangeExecutionCredential(
            api_key="<redacted>",
            api_secret="<redacted>",
        ),
    )
    assert result.lookup_outcome == "confirmed_absent"
    assert result.exchange_order_id == ""


def _command(*, market_type: str) -> ExchangeOrderCommand:
    return ExchangeOrderCommand(
        intent_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
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


class _JsonResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body
