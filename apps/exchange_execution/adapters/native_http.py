from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import perf_counter
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.live_execution.application.ports import ExchangeOrderAdapterError
from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeExecutionCredential,
    ExchangeOrderCancelResult,
    ExchangeOrderCommand,
    ExchangeOrderStatusResult,
    ExchangeOrderSubmitResult,
    ExchangePrivateStreamSession,
    ExecutionFillFact,
)

_RECV_WINDOW = "5000"
_BINANCE_SPOT_TESTNET_URL = "https://testnet.binance.vision"
_BINANCE_FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"
_BYBIT_TESTNET_URL = "https://api-testnet.bybit.com"


@dataclass(frozen=True, slots=True)
class BinanceTestnetOrderAdapter:
    timeout_seconds: float = 5.0
    exchange_name: str = "binance"

    def server_time_ms(self) -> int:
        payload = _get_json(
            url=f"{_BINANCE_SPOT_TESTNET_URL}/api/v3/time",
            headers={},
            timeout_seconds=self.timeout_seconds,
        )
        return int(payload["serverTime"])

    def submit_order(
        self, *, command: ExchangeOrderCommand, credential: object
    ) -> ExchangeOrderSubmitResult:
        secret = _credential(credential)
        base_url, path = _binance_base_and_order_path(command)
        params = _binance_order_params(command)
        started = perf_counter()
        payload = _binance_signed_json(
            method="POST",
            base_url=base_url,
            path=path,
            params=params,
            credential=secret,
            timeout_seconds=self.timeout_seconds,
        )
        observed_at = datetime.now(tz=UTC)
        return ExchangeOrderSubmitResult(
            exchange_order_id=str(payload.get("orderId") or payload.get("clientOrderId")),
            exchange_status=str(payload.get("status") or "submitted").lower(),
            submitted_at=observed_at,
            latency_ms=_elapsed_ms(started),
            metadata={"provider": "binance", "http_method": "POST"},
        )

    def get_order_status(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderStatusResult:
        secret = _credential(credential)
        base_url, path = _binance_base_and_order_path(command)
        params: dict[str, object] = {"symbol": _symbol(command), "orderId": exchange_order_id}
        started = perf_counter()
        payload = _binance_signed_json(
            method="GET",
            base_url=base_url,
            path=path,
            params=params,
            credential=secret,
            timeout_seconds=self.timeout_seconds,
        )
        observed_at = datetime.now(tz=UTC)
        return ExchangeOrderStatusResult(
            exchange_order_id=str(payload.get("orderId") or exchange_order_id),
            exchange_status=str(payload.get("status") or "status_checked").lower(),
            checked_at=observed_at,
            latency_ms=_elapsed_ms(started),
            metadata={"provider": "binance", "http_method": "GET"},
            fills=_binance_order_fills(
                command=command,
                exchange_order_id=exchange_order_id,
                credential=secret,
                timeout_seconds=self.timeout_seconds,
            ),
        )

    def cancel_order(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderCancelResult:
        secret = _credential(credential)
        base_url, path = _binance_base_and_order_path(command)
        params: dict[str, object] = {"symbol": _symbol(command), "orderId": exchange_order_id}
        started = perf_counter()
        payload = _binance_signed_json(
            method="DELETE",
            base_url=base_url,
            path=path,
            params=params,
            credential=secret,
            timeout_seconds=self.timeout_seconds,
        )
        observed_at = datetime.now(tz=UTC)
        return ExchangeOrderCancelResult(
            exchange_order_id=str(payload.get("orderId") or exchange_order_id),
            exchange_status=str(payload.get("status") or "cancelled").lower(),
            cancelled_at=observed_at,
            latency_ms=_elapsed_ms(started),
            metadata={"provider": "binance", "http_method": "DELETE"},
        )

    def ensure_private_stream_session(
        self,
        *,
        connection: ExchangeExecutionConnection,
    ) -> ExchangePrivateStreamSession:
        base_url, path = _binance_base_and_listen_key_path(
            market_type=connection.market_type
        )
        started = perf_counter()
        payload = _request_json(
            method="POST",
            url=f"{base_url}{path}",
            headers={"X-MBX-APIKEY": connection.credential.api_key},
            timeout_seconds=self.timeout_seconds,
        )
        listen_key = str(payload.get("listenKey") or "")
        if not listen_key:
            raise ExchangeOrderAdapterError(reason="private_stream_listen_key_missing")
        keepalive_url = f"{base_url}{path}?{urllib.parse.urlencode({'listenKey': listen_key})}"
        _request_json(
            method="PUT",
            url=keepalive_url,
            headers={"X-MBX-APIKEY": connection.credential.api_key},
            timeout_seconds=self.timeout_seconds,
            allow_empty=True,
        )
        now = datetime.now(tz=UTC)
        return ExchangePrivateStreamSession(
            session_id=_session_uuid("binance", listen_key),
            exchange_name="binance",
            environment=connection.environment,
            market_type=connection.market_type,
            status="ready",
            status_reason="listen_key_keepalive_ok",
            opened_at=now,
            keepalive_at=now,
            expires_at=now + timedelta(minutes=60),
            metadata={"provider": "binance", "latency_ms": round(_elapsed_ms(started), 3)},
        )


@dataclass(frozen=True, slots=True)
class BybitTestnetOrderAdapter:
    timeout_seconds: float = 5.0
    exchange_name: str = "bybit"

    def server_time_ms(self) -> int:
        payload = _get_json(
            url=f"{_BYBIT_TESTNET_URL}/v5/market/time",
            headers={},
            timeout_seconds=self.timeout_seconds,
        )
        result = payload.get("result")
        if isinstance(result, Mapping):
            return int(result.get("timeNano", "0")) // 1_000_000
        return int(payload.get("time", int(time.time() * 1000)))

    def submit_order(
        self, *, command: ExchangeOrderCommand, credential: object
    ) -> ExchangeOrderSubmitResult:
        secret = _credential(credential)
        params = _bybit_order_params(command)
        started = perf_counter()
        payload = _bybit_signed_json(
            method="POST",
            path="/v5/order/create",
            params=params,
            credential=secret,
            timeout_seconds=self.timeout_seconds,
        )
        result = _bybit_result(payload)
        observed_at = datetime.now(tz=UTC)
        return ExchangeOrderSubmitResult(
            exchange_order_id=str(result.get("orderId") or result.get("orderLinkId")),
            exchange_status="submitted",
            submitted_at=observed_at,
            latency_ms=_elapsed_ms(started),
            metadata={"provider": "bybit", "http_method": "POST"},
        )

    def get_order_status(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderStatusResult:
        secret = _credential(credential)
        params: dict[str, object] = {
            "category": _bybit_category(command),
            "symbol": _symbol(command),
            "orderId": exchange_order_id,
        }
        started = perf_counter()
        payload = _bybit_signed_json(
            method="GET",
            path="/v5/order/realtime",
            params=params,
            credential=secret,
            timeout_seconds=self.timeout_seconds,
        )
        result = _bybit_result(payload)
        rows = result.get("list") if isinstance(result, Mapping) else None
        row = rows[0] if isinstance(rows, list) and rows else {}
        observed_at = datetime.now(tz=UTC)
        return ExchangeOrderStatusResult(
            exchange_order_id=str(row.get("orderId") or exchange_order_id),
            exchange_status=str(row.get("orderStatus") or "status_checked").lower(),
            checked_at=observed_at,
            latency_ms=_elapsed_ms(started),
            metadata={"provider": "bybit", "http_method": "GET"},
            fills=_bybit_order_fills(
                command=command,
                exchange_order_id=exchange_order_id,
                credential=secret,
                timeout_seconds=self.timeout_seconds,
            ),
        )

    def cancel_order(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderCancelResult:
        secret = _credential(credential)
        params: dict[str, object] = {
            "category": _bybit_category(command),
            "symbol": _symbol(command),
            "orderId": exchange_order_id,
        }
        started = perf_counter()
        payload = _bybit_signed_json(
            method="POST",
            path="/v5/order/cancel",
            params=params,
            credential=secret,
            timeout_seconds=self.timeout_seconds,
        )
        result = _bybit_result(payload)
        observed_at = datetime.now(tz=UTC)
        return ExchangeOrderCancelResult(
            exchange_order_id=str(result.get("orderId") or exchange_order_id),
            exchange_status="cancel_requested",
            cancelled_at=observed_at,
            latency_ms=_elapsed_ms(started),
            metadata={"provider": "bybit", "http_method": "POST"},
        )

    def ensure_private_stream_session(
        self,
        *,
        connection: ExchangeExecutionConnection,
    ) -> ExchangePrivateStreamSession:
        server_time = self.server_time_ms()
        now = datetime.now(tz=UTC)
        return ExchangePrivateStreamSession(
            session_id=_session_uuid("bybit", f"{connection.connection_id}:{server_time}"),
            exchange_name="bybit",
            environment=connection.environment,
            market_type=connection.market_type,
            status="ready",
            status_reason="private_ws_auth_probe_ready",
            opened_at=now,
            keepalive_at=now,
            expires_at=None,
            metadata={"provider": "bybit", "server_time_ms": server_time},
        )


def _binance_signed_json(
    *,
    method: str,
    base_url: str,
    path: str,
    params: Mapping[str, object],
    credential: ExchangeExecutionCredential,
    timeout_seconds: float,
) -> dict[str, Any]:
    signed = {**params, "recvWindow": _RECV_WINDOW, "timestamp": str(int(time.time() * 1000))}
    query = urllib.parse.urlencode(signed)
    signature = hmac.new(
        credential.api_secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    url = f"{base_url}{path}?{query}&signature={signature}"
    return _request_json(
        method=method,
        url=url,
        headers={"X-MBX-APIKEY": credential.api_key},
        timeout_seconds=timeout_seconds,
    )


def _bybit_signed_json(
    *,
    method: str,
    path: str,
    params: Mapping[str, object],
    credential: ExchangeExecutionCredential,
    timeout_seconds: float,
) -> dict[str, Any]:
    timestamp = str(int(time.time() * 1000))
    if method == "GET":
        body = urllib.parse.urlencode(params)
        url = f"{_BYBIT_TESTNET_URL}{path}?{body}"
        data = None
    else:
        body = json.dumps(params, separators=(",", ":"), sort_keys=True)
        url = f"{_BYBIT_TESTNET_URL}{path}"
        data = body.encode("utf-8")
    signing_payload = f"{timestamp}{credential.api_key}{_RECV_WINDOW}{body}"
    signature = hmac.new(
        credential.api_secret.encode("utf-8"),
        signing_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return _request_json(
        method=method,
        url=url,
        headers={
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": credential.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": _RECV_WINDOW,
            "X-BAPI-SIGN": signature,
        },
        data=data,
        timeout_seconds=timeout_seconds,
    )


def _binance_signed_payload(
    *,
    method: str,
    base_url: str,
    path: str,
    params: Mapping[str, object],
    credential: ExchangeExecutionCredential,
    timeout_seconds: float,
) -> Any:
    signed = {**params, "recvWindow": _RECV_WINDOW, "timestamp": str(int(time.time() * 1000))}
    query = urllib.parse.urlencode(signed)
    signature = hmac.new(
        credential.api_secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    url = f"{base_url}{path}?{query}&signature={signature}"
    return _request_json_payload(
        method=method,
        url=url,
        headers={"X-MBX-APIKEY": credential.api_key},
        timeout_seconds=timeout_seconds,
    )


def _request_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    timeout_seconds: float,
    data: bytes | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ExchangeOrderAdapterError(reason=f"exchange_http_{exc.code}") from exc
    except TimeoutError as exc:
        raise ExchangeOrderAdapterError(
            reason="exchange_request_timeout",
            unknown_state=True,
        ) from exc
    except OSError as exc:
        raise ExchangeOrderAdapterError(reason="exchange_request_failed") from exc
    if allow_empty and not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExchangeOrderAdapterError(reason="exchange_response_invalid") from exc
    if not isinstance(payload, dict):
        raise ExchangeOrderAdapterError(reason="exchange_response_invalid")
    if "retCode" in payload and int(payload.get("retCode", 0)) != 0:
        raise ExchangeOrderAdapterError(reason=f"exchange_ret_code_{payload.get('retCode')}")
    return payload


def _request_json_payload(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    timeout_seconds: float,
) -> Any:
    request = urllib.request.Request(url=url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ExchangeOrderAdapterError(reason=f"exchange_http_{exc.code}") from exc
    except TimeoutError as exc:
        raise ExchangeOrderAdapterError(
            reason="exchange_request_timeout",
            unknown_state=True,
        ) from exc
    except OSError as exc:
        raise ExchangeOrderAdapterError(reason="exchange_request_failed") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExchangeOrderAdapterError(reason="exchange_response_invalid") from exc


def _get_json(*, url: str, headers: dict[str, str], timeout_seconds: float) -> dict[str, Any]:
    return _request_json(
        method="GET",
        url=url,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )


def _credential(value: object) -> ExchangeExecutionCredential:
    if not isinstance(value, ExchangeExecutionCredential):
        raise ExchangeOrderAdapterError(reason="exchange_credential_invalid")
    return value


def _symbol(command: ExchangeOrderCommand) -> str:
    parts = command.instrument_key.split(":")
    return parts[-1].upper()


def _binance_base_and_order_path(command: ExchangeOrderCommand) -> tuple[str, str]:
    if command.environment != "testnet":
        raise ExchangeOrderAdapterError(reason="mainnet_hard_block")
    if command.market_type == "spot":
        return _BINANCE_SPOT_TESTNET_URL, "/api/v3/order"
    if command.market_type == "futures":
        return _BINANCE_FUTURES_TESTNET_URL, "/fapi/v1/order"
    raise ExchangeOrderAdapterError(reason="unsupported_market_type")


def _binance_base_and_listen_key_path(*, market_type: str) -> tuple[str, str]:
    if market_type == "spot":
        return _BINANCE_SPOT_TESTNET_URL, "/api/v3/userDataStream"
    if market_type == "futures":
        return _BINANCE_FUTURES_TESTNET_URL, "/fapi/v1/listenKey"
    raise ExchangeOrderAdapterError(reason="unsupported_market_type")


def _binance_order_params(command: ExchangeOrderCommand) -> dict[str, object]:
    params: dict[str, object] = {
        "symbol": _symbol(command),
        "side": command.side.upper(),
        "type": command.order_type.upper(),
        "newClientOrderId": command.client_order_id,
    }
    if command.quantity is not None:
        params["quantity"] = str(command.quantity.normalize())
    elif command.quote_notional is not None and command.market_type == "spot":
        params["quoteOrderQty"] = str(command.quote_notional.normalize())
    else:
        raise ExchangeOrderAdapterError(reason="unsupported_order_sizing")
    if command.order_type == "limit":
        if command.limit_price is None:
            raise ExchangeOrderAdapterError(reason="limit_price_required")
        params["price"] = str(command.limit_price.normalize())
        params["timeInForce"] = "GTC"
    return params


def _bybit_order_params(command: ExchangeOrderCommand) -> dict[str, object]:
    params: dict[str, object] = {
        "category": _bybit_category(command),
        "symbol": _symbol(command),
        "side": command.side.capitalize(),
        "orderType": command.order_type.capitalize(),
        "orderLinkId": command.client_order_id,
    }
    if command.quantity is not None:
        params["qty"] = str(command.quantity.normalize())
    elif command.quote_notional is not None and command.market_type == "spot":
        params["qty"] = str(command.quote_notional.normalize())
        params["marketUnit"] = "quoteCoin"
    else:
        raise ExchangeOrderAdapterError(reason="unsupported_order_sizing")
    if command.order_type == "limit":
        if command.limit_price is None:
            raise ExchangeOrderAdapterError(reason="limit_price_required")
        params["price"] = str(command.limit_price.normalize())
        params["timeInForce"] = "GTC"
    return params


def _bybit_category(command: ExchangeOrderCommand) -> str:
    if command.environment != "testnet":
        raise ExchangeOrderAdapterError(reason="mainnet_hard_block")
    if command.market_type == "spot":
        return "spot"
    if command.market_type == "futures":
        return "linear"
    raise ExchangeOrderAdapterError(reason="unsupported_market_type")


def _bybit_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ExchangeOrderAdapterError(reason="exchange_response_invalid")
    return result


def _binance_order_fills(
    *,
    command: ExchangeOrderCommand,
    exchange_order_id: str,
    credential: ExchangeExecutionCredential,
    timeout_seconds: float,
) -> tuple[ExecutionFillFact, ...]:
    base_url, _path = _binance_base_and_order_path(command)
    trades_path = "/api/v3/myTrades" if command.market_type == "spot" else "/fapi/v1/userTrades"
    try:
        payload = _binance_signed_payload(
            method="GET",
            base_url=base_url,
            path=trades_path,
            params={"symbol": _symbol(command), "orderId": exchange_order_id},
            credential=credential,
            timeout_seconds=timeout_seconds,
        )
    except ExchangeOrderAdapterError:
        return ()
    if not isinstance(payload, list):
        return ()
    fills: list[ExecutionFillFact] = []
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        trade_id = str(row.get("id") or row.get("tradeId") or "")
        price = _decimal_from(row.get("price"))
        quantity = _decimal_from(row.get("qty"))
        fee = _decimal_from(row.get("commission"), default="0")
        fee_asset = str(row.get("commissionAsset") or row.get("commissionAsset".lower()) or "")
        timestamp = _datetime_from_millis(row.get("time"))
        if not trade_id or price is None or quantity is None or not fee_asset or timestamp is None:
            continue
        fills.append(
            ExecutionFillFact(
                provider_trade_id=trade_id,
                price=price,
                quantity=quantity,
                fee_amount=fee or Decimal("0"),
                fee_asset=fee_asset,
                filled_at=timestamp,
                liquidity=_maker_taker(row.get("isMaker")),
                metadata={"provider": "binance"},
            )
        )
    return tuple(fills)


def _bybit_order_fills(
    *,
    command: ExchangeOrderCommand,
    exchange_order_id: str,
    credential: ExchangeExecutionCredential,
    timeout_seconds: float,
) -> tuple[ExecutionFillFact, ...]:
    try:
        payload = _bybit_signed_json(
            method="GET",
            path="/v5/execution/list",
            params={
                "category": _bybit_category(command),
                "symbol": _symbol(command),
                "orderId": exchange_order_id,
            },
            credential=credential,
            timeout_seconds=timeout_seconds,
        )
    except ExchangeOrderAdapterError:
        return ()
    result = _bybit_result(payload)
    rows = result.get("list")
    if not isinstance(rows, list):
        return ()
    fills: list[ExecutionFillFact] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        trade_id = str(row.get("execId") or "")
        price = _decimal_from(row.get("execPrice"))
        quantity = _decimal_from(row.get("execQty"))
        fee = _decimal_from(row.get("execFee"), default="0")
        fee_asset = str(row.get("feeCurrency") or row.get("feeRate") or "UNKNOWN")
        timestamp = _datetime_from_millis(row.get("execTime"))
        if not trade_id or price is None or quantity is None or timestamp is None:
            continue
        fills.append(
            ExecutionFillFact(
                provider_trade_id=trade_id,
                price=price,
                quantity=quantity,
                fee_amount=fee or Decimal("0"),
                fee_asset=fee_asset,
                filled_at=timestamp,
                liquidity=str(row.get("execType") or "") or None,
                metadata={"provider": "bybit"},
            )
        )
    return tuple(fills)


def _decimal_from(value: object, *, default: str | None = None) -> Decimal | None:
    if value is None:
        return Decimal(default) if default is not None else None
    text = str(value).strip()
    if not text:
        return Decimal(default) if default is not None else None
    return Decimal(text)


def _datetime_from_millis(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        millis = int(str(value))
    except ValueError:
        return None
    return datetime.fromtimestamp(millis / 1000, tz=UTC)


def _maker_taker(value: object) -> str | None:
    if value is True:
        return "maker"
    if value is False:
        return "taker"
    return None


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


def _session_uuid(exchange: str, value: str) -> UUID:
    digest = hashlib.sha256(f"{exchange}:{value}".encode("utf-8")).hexdigest()
    return UUID(digest[:32])
