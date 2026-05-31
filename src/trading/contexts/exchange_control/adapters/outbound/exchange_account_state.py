from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from trading.contexts.exchange_control.application.account_state import (
    ExchangeAccountStateReader,
    ExchangeAccountStateReadRequest,
    ExchangeAccountStateReadResult,
    ExchangeBalanceState,
    ExchangeInstrumentFilterState,
    ExchangeOpenOrderState,
    ExchangePositionState,
)

_BYBIT_MAINNET_URL = "https://api.bybit.com"
_BYBIT_TESTNET_URL = "https://api-testnet.bybit.com"
_RECV_WINDOW = "5000"


@dataclass(frozen=True)
class SkippedExchangeAccountStateReader(ExchangeAccountStateReader):
    requires_plaintext: bool = False

    def read_account_state(
        self,
        *,
        request: ExchangeAccountStateReadRequest,
        now: datetime,
    ) -> ExchangeAccountStateReadResult:
        return ExchangeAccountStateReadResult(
            exchange_name=request.exchange_name,
            market_type=request.market_type,
            environment=request.environment,
            account_mode="unknown",
            balances=(),
            positions=(),
            open_orders=(),
            instrument_filters=(),
            observed_at=now,
            source_hash=_source_hash({"status": "degraded", "reason": "sync_disabled"}),
            sync_status="degraded",
            sync_reason="account_state_sync_disabled",
        )


@dataclass(frozen=True)
class HttpExchangeAccountStateReader(ExchangeAccountStateReader):
    timeout_seconds: float = 3.0
    requires_plaintext: bool = True

    def read_account_state(
        self,
        *,
        request: ExchangeAccountStateReadRequest,
        now: datetime,
    ) -> ExchangeAccountStateReadResult:
        if request.exchange_name == "bybit":
            return _read_bybit_account_state(
                request=request,
                now=now,
                timeout_seconds=self.timeout_seconds,
            )
        return ExchangeAccountStateReadResult(
            exchange_name=request.exchange_name,
            market_type=request.market_type,
            environment=request.environment,
            account_mode="unknown",
            balances=(),
            positions=(),
            open_orders=(),
            instrument_filters=(),
            observed_at=now,
            source_hash=_source_hash(
                {
                    "exchange": request.exchange_name,
                    "market_type": request.market_type,
                    "status": "degraded",
                }
            ),
            sync_status="degraded",
            sync_reason="account_state_reader_unsupported_exchange",
        )


def _read_bybit_account_state(
    *,
    request: ExchangeAccountStateReadRequest,
    now: datetime,
    timeout_seconds: float,
) -> ExchangeAccountStateReadResult:
    base_url = _BYBIT_TESTNET_URL if request.environment == "testnet" else _BYBIT_MAINNET_URL
    category = "spot" if request.market_type == "spot" else "linear"
    symbols = tuple(_symbol_from_instrument_key(item) for item in request.instrument_keys)

    wallet_payload = _bybit_get_signed_json(
        base_url=base_url,
        path="/v5/account/wallet-balance",
        params={"accountType": "UNIFIED"},
        api_key=request.credential.api_key,
        api_secret=request.credential.api_secret,
        timeout_seconds=timeout_seconds,
    )
    balances = _bybit_balances(payload=wallet_payload)

    orders_payload = _bybit_get_signed_json(
        base_url=base_url,
        path="/v5/order/realtime",
        params={"category": category, "openOnly": "0"},
        api_key=request.credential.api_key,
        api_secret=request.credential.api_secret,
        timeout_seconds=timeout_seconds,
    )
    open_orders = _bybit_open_orders(
        payload=orders_payload,
        exchange=request.exchange_name,
        market_type=request.market_type,
    )

    positions: tuple[ExchangePositionState, ...] = ()
    if category != "spot":
        positions_payload = _bybit_get_signed_json(
            base_url=base_url,
            path="/v5/position/list",
            params={"category": category},
            api_key=request.credential.api_key,
            api_secret=request.credential.api_secret,
            timeout_seconds=timeout_seconds,
        )
        positions = _bybit_positions(
            payload=positions_payload,
            exchange=request.exchange_name,
            market_type=request.market_type,
        )

    filters: list[ExchangeInstrumentFilterState] = []
    for symbol in symbols:
        if not symbol:
            continue
        instrument_payload = _get_json(
            url=f"{base_url}/v5/market/instruments-info?category={category}&symbol={symbol}",
            headers={},
            timeout_seconds=timeout_seconds,
        )
        filters.extend(
            _bybit_instrument_filters(
                payload=instrument_payload,
                exchange=request.exchange_name,
                market_type=request.market_type,
            )
        )

    normalized = {
        "exchange_name": request.exchange_name,
        "market_type": request.market_type,
        "environment": request.environment,
        "account_mode": "unified",
        "balances": [_balance_payload(item) for item in balances],
        "positions": [_position_payload(item) for item in positions],
        "open_orders": [_order_payload(item) for item in open_orders],
        "instrument_filters": [_filter_payload(item) for item in filters],
        "observed_at": now.isoformat(),
    }
    return ExchangeAccountStateReadResult(
        exchange_name=request.exchange_name,
        market_type=request.market_type,
        environment=request.environment,
        account_mode="unified",
        balances=balances,
        positions=positions,
        open_orders=open_orders,
        instrument_filters=tuple(filters),
        observed_at=now,
        source_hash=_source_hash(normalized),
    )


def _bybit_get_signed_json(
    *,
    base_url: str,
    path: str,
    params: dict[str, str],
    api_key: str,
    api_secret: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    timestamp = str(int(time.time() * 1000))
    signing_payload = f"{timestamp}{api_key}{_RECV_WINDOW}{query}"
    signature = hmac.new(
        api_secret.encode("utf-8"),
        signing_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return _get_json(
        url=f"{base_url}{path}?{query}" if query else f"{base_url}{path}",
        headers={
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": _RECV_WINDOW,
            "X-BAPI-SIGN": signature,
        },
        timeout_seconds=timeout_seconds,
    )


def _get_json(
    *,
    url: str,
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(url=url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exchange account state response is invalid")
    if int(payload.get("retCode", 0)) != 0:
        raise ValueError("exchange account state request rejected")
    return payload


def _bybit_balances(*, payload: dict[str, Any]) -> tuple[ExchangeBalanceState, ...]:
    result = payload.get("result")
    accounts = result.get("list") if isinstance(result, dict) else None
    if not isinstance(accounts, list):
        return ()
    balances: list[ExchangeBalanceState] = []
    for account in accounts:
        coins = account.get("coin") if isinstance(account, dict) else None
        if not isinstance(coins, list):
            continue
        for coin in coins:
            if not isinstance(coin, dict):
                continue
            asset = str(coin.get("coin") or "").strip().upper()
            if not asset:
                continue
            total = _decimal_or_none(coin.get("walletBalance"))
            locked = _decimal_or_zero(coin.get("locked"))
            available = _decimal_or_none(coin.get("availableToWithdraw"))
            free = (
                available
                if available is not None
                else max(Decimal("0"), (total or Decimal("0")) - locked)
            )
            balances.append(
                ExchangeBalanceState(
                    asset=asset,
                    free=free,
                    locked=locked,
                    total=total,
                )
            )
    return tuple(sorted(balances, key=lambda item: item.asset))


def _bybit_open_orders(
    *,
    payload: dict[str, Any],
    exchange: str,
    market_type: str,
) -> tuple[ExchangeOpenOrderState, ...]:
    result = payload.get("result")
    rows = result.get("list") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        return ()
    orders: list[ExchangeOpenOrderState] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        order_id = str(row.get("orderId") or "").strip()
        if not symbol or not order_id:
            continue
        orders.append(
            ExchangeOpenOrderState(
                instrument_key=f"{exchange}:{market_type}:{symbol}",
                exchange_order_ref=order_id,
                side=str(row.get("side") or "").lower(),
                order_type=str(row.get("orderType") or "").lower(),
                quantity=_decimal_or_zero(row.get("qty")),
                price=_decimal_or_none(row.get("price")),
                status=str(row.get("orderStatus") or "").lower(),
            )
        )
    return tuple(orders)


def _bybit_positions(
    *,
    payload: dict[str, Any],
    exchange: str,
    market_type: str,
) -> tuple[ExchangePositionState, ...]:
    result = payload.get("result")
    rows = result.get("list") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        return ()
    positions: list[ExchangePositionState] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        positions.append(
            ExchangePositionState(
                instrument_key=f"{exchange}:{market_type}:{symbol}",
                side=_position_side(row.get("side")),
                quantity=_decimal_or_zero(row.get("size")),
                entry_price=_decimal_or_none(row.get("avgPrice")),
                leverage=_decimal_or_none(row.get("leverage")),
                margin_mode=str(row.get("tradeMode") or "") or None,
                position_mode=str(row.get("positionIdx") or "") or None,
            )
        )
    return tuple(positions)


def _bybit_instrument_filters(
    *,
    payload: dict[str, Any],
    exchange: str,
    market_type: str,
) -> tuple[ExchangeInstrumentFilterState, ...]:
    result = payload.get("result")
    rows = result.get("list") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        return ()
    filters: list[ExchangeInstrumentFilterState] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        price_filter = row.get("priceFilter")
        lot_filter = row.get("lotSizeFilter")
        leverage_filter = row.get("leverageFilter")
        filters.append(
            ExchangeInstrumentFilterState(
                instrument_key=f"{exchange}:{market_type}:{symbol}",
                tick_size=_decimal_or_none(
                    price_filter.get("tickSize") if isinstance(price_filter, dict) else None
                ),
                step_size=_decimal_or_none(
                    lot_filter.get("basePrecision") if isinstance(lot_filter, dict) else None
                ),
                min_qty=_decimal_or_none(
                    lot_filter.get("minOrderQty") if isinstance(lot_filter, dict) else None
                ),
                min_notional=_decimal_or_none(
                    lot_filter.get("minOrderAmt") if isinstance(lot_filter, dict) else None
                ),
                max_leverage=_decimal_or_none(
                    leverage_filter.get("maxLeverage")
                    if isinstance(leverage_filter, dict)
                    else None
                ),
            )
        )
    return tuple(filters)


def _symbol_from_instrument_key(value: str) -> str:
    parts = value.split(":")
    return (parts[-1] if parts else value).strip().upper()


def _position_side(value: object) -> str:
    side = str(value or "").strip().lower()
    if side == "buy":
        return "long"
    if side == "sell":
        return "short"
    return "net"


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _decimal_or_zero(value: object) -> Decimal:
    return _decimal_or_none(value) or Decimal("0")


def _source_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _balance_payload(value: ExchangeBalanceState) -> dict[str, str | None]:
    return {
        "asset": value.asset,
        "free": str(value.free),
        "locked": str(value.locked),
        "total": str(value.total) if value.total is not None else None,
    }


def _position_payload(value: ExchangePositionState) -> dict[str, str | None]:
    return {
        "instrument_key": value.instrument_key,
        "side": value.side,
        "quantity": str(value.quantity),
        "entry_price": str(value.entry_price) if value.entry_price is not None else None,
        "leverage": str(value.leverage) if value.leverage is not None else None,
        "margin_mode": value.margin_mode,
        "position_mode": value.position_mode,
    }


def _order_payload(value: ExchangeOpenOrderState) -> dict[str, str | None]:
    return {
        "instrument_key": value.instrument_key,
        "exchange_order_ref": value.exchange_order_ref,
        "side": value.side,
        "order_type": value.order_type,
        "quantity": str(value.quantity),
        "price": str(value.price) if value.price is not None else None,
        "status": value.status,
    }


def _filter_payload(value: ExchangeInstrumentFilterState) -> dict[str, str | None]:
    return {
        "instrument_key": value.instrument_key,
        "tick_size": str(value.tick_size) if value.tick_size is not None else None,
        "step_size": str(value.step_size) if value.step_size is not None else None,
        "min_qty": str(value.min_qty) if value.min_qty is not None else None,
        "min_notional": str(value.min_notional) if value.min_notional is not None else None,
        "max_leverage": str(value.max_leverage) if value.max_leverage is not None else None,
    }


__all__ = [
    "HttpExchangeAccountStateReader",
    "SkippedExchangeAccountStateReader",
]
