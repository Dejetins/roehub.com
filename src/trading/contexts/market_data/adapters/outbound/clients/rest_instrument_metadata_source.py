from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.clients.common_http import HttpClient
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    MarketConfig,
    MarketDataRuntimeConfig,
)
from trading.contexts.market_data.application.dto import ExchangeInstrumentMetadata
from trading.contexts.market_data.application.ports.sources.instrument_metadata_source import (
    InstrumentMetadataSource,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol


@dataclass(frozen=True, slots=True)
class RestInstrumentMetadataSource(InstrumentMetadataSource):
    """
    REST metadata source for instrument enrichment over Binance/Bybit spot/futures.

    Parameters:
    - cfg: runtime market configuration.
    - http: HTTP client with retry/timeout support.

    Assumptions/Invariants:
    - `market_id` routing is defined in runtime config.
    - Returned metadata rows are normalized to domain DTOs.
    """

    cfg: MarketDataRuntimeConfig
    http: HttpClient

    def __post_init__(self) -> None:
        """
        Validate required source dependencies.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Runtime config and HTTP client are non-null.

        Errors/Exceptions:
        - Raises `ValueError` when dependencies are missing.

        Side effects:
        - None.
        """
        if self.cfg is None:  # type: ignore[truthy-bool]
            raise ValueError("RestInstrumentMetadataSource requires cfg")
        if self.http is None:  # type: ignore[truthy-bool]
            raise ValueError("RestInstrumentMetadataSource requires http")

    def list_for_market(self, market_id: MarketId) -> Sequence[ExchangeInstrumentMetadata]:
        """
        Fetch exchange metadata rows for one market id.

        Parameters:
        - market_id: market identity from runtime config.

        Returns:
        - Sequence of normalized metadata rows.

        Assumptions/Invariants:
        - Supported exchanges are `binance` and `bybit`.

        Errors/Exceptions:
        - Raises `ValueError` for unsupported exchange or malformed payloads.
        - Propagates HTTP client errors.

        Side effects:
        - Performs one or more REST requests.
        """
        market = self.cfg.market_by_id(market_id)
        if market.exchange == "binance":
            return self._list_binance_metadata(market)
        if market.exchange == "bybit":
            return self._list_bybit_metadata(market)
        raise ValueError(f"unsupported exchange for metadata source: {market.exchange!r}")

    def _list_binance_metadata(self, market: MarketConfig) -> list[ExchangeInstrumentMetadata]:
        """
        Fetch and normalize Binance exchange-info payload for one market.

        Parameters:
        - market: runtime market config entry.

        Returns:
        - List of metadata rows.

        Assumptions/Invariants:
        - Spot endpoint is `/api/v3/exchangeInfo`, futures endpoint is `/fapi/v1/exchangeInfo`.

        Errors/Exceptions:
        - Raises `RuntimeError` when payload shape is invalid.

        Side effects:
        - Performs one HTTP GET request.
        """
        path = (
            "/fapi/v1/exchangeInfo"
            if market.market_type == "futures"
            else "/api/v3/exchangeInfo"
        )
        response = self.http.get_json(
            url=market.rest.base_url.rstrip("/") + path,
            params={},
            timeout_s=market.rest.timeout_s,
            retries=market.rest.retries,
            backoff_base_s=market.rest.backoff.base_s,
            backoff_max_s=market.rest.backoff.max_s,
            backoff_jitter_s=market.rest.backoff.jitter_s,
        )
        body = response.body
        if not isinstance(body, dict):
            raise RuntimeError(
                f"Unexpected Binance exchangeInfo payload type: {type(body).__name__}"
            )

        rows = body.get("symbols")
        if not isinstance(rows, list):
            raise RuntimeError("Unexpected Binance exchangeInfo payload: missing symbols list")

        out: list[ExchangeInstrumentMetadata] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol_text = _as_non_empty_string(item.get("symbol"))
            if symbol_text is None:
                continue

            filters = item.get("filters")
            price_step = _as_positive_float(
                _binance_filter_value(filters, "PRICE_FILTER", "tickSize")
            )
            qty_step = _as_positive_float(_binance_filter_value(filters, "LOT_SIZE", "stepSize"))
            min_notional = _as_positive_float(
                _binance_filter_value(filters, "MIN_NOTIONAL", "minNotional")
                or _binance_filter_value(filters, "NOTIONAL", "minNotional")
            )

            out.append(
                ExchangeInstrumentMetadata(
                    instrument_id=InstrumentId(market.market_id, Symbol(symbol_text)),
                    base_asset=_as_non_empty_string(item.get("baseAsset")),
                    quote_asset=_as_non_empty_string(item.get("quoteAsset")),
                    price_step=price_step,
                    qty_step=qty_step,
                    min_notional=min_notional,
                )
            )
        return out

    def _list_bybit_metadata(self, market: MarketConfig) -> list[ExchangeInstrumentMetadata]:
        """
        Fetch and normalize Bybit V5 instrument-info payload (with pagination).

        Parameters:
        - market: runtime market config entry.

        Returns:
        - List of metadata rows.

        Assumptions/Invariants:
        - Category is `spot` for spot market type and `linear` for futures.

        Errors/Exceptions:
        - Raises `RuntimeError` on malformed payload or non-zero retCode.

        Side effects:
        - Performs one or more HTTP GET requests until pagination ends.
        """
        category = "spot" if market.market_type == "spot" else "linear"
        path = "/v5/market/instruments-info"

        cursor = ""
        out: list[ExchangeInstrumentMetadata] = []
        while True:
            params: dict[str, Any] = {"category": category, "limit": 1000}
            if cursor:
                params["cursor"] = cursor

            response = self.http.get_json(
                url=market.rest.base_url.rstrip("/") + path,
                params=params,
                timeout_s=market.rest.timeout_s,
                retries=market.rest.retries,
                backoff_base_s=market.rest.backoff.base_s,
                backoff_max_s=market.rest.backoff.max_s,
                backoff_jitter_s=market.rest.backoff.jitter_s,
            )
            body = response.body
            if not isinstance(body, dict):
                raise RuntimeError(f"Unexpected Bybit payload type: {type(body).__name__}")

            ret_code = body.get("retCode")
            if ret_code not in (0, "0", None):
                raise RuntimeError(f"Bybit retCode={ret_code!r} for instruments-info")

            result = body.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Unexpected Bybit payload: missing result mapping")

            rows = result.get("list")
            if not isinstance(rows, list):
                raise RuntimeError("Unexpected Bybit payload: result.list must be a list")

            for item in rows:
                if not isinstance(item, dict):
                    continue
                symbol_text = _as_non_empty_string(item.get("symbol"))
                if symbol_text is None:
                    continue

                raw_price_filter = item.get("priceFilter")
                raw_lot_filter = item.get("lotSizeFilter")
                price_filter = raw_price_filter if isinstance(raw_price_filter, dict) else {}
                lot_filter = raw_lot_filter if isinstance(raw_lot_filter, dict) else {}

                min_notional = _as_positive_float(
                    lot_filter.get("minNotionalValue")
                    or lot_filter.get("minOrderAmt")
                    or lot_filter.get("minOrderQty")
                )

                out.append(
                    ExchangeInstrumentMetadata(
                        instrument_id=InstrumentId(market.market_id, Symbol(symbol_text)),
                        base_asset=_as_non_empty_string(item.get("baseCoin")),
                        quote_asset=_as_non_empty_string(item.get("quoteCoin")),
                        price_step=_as_positive_float(price_filter.get("tickSize")),
                        qty_step=_as_positive_float(lot_filter.get("qtyStep")),
                        min_notional=min_notional,
                    )
                )

            next_cursor = _as_non_empty_string(result.get("nextPageCursor"))
            if next_cursor is None or next_cursor == cursor:
                break
            cursor = next_cursor

        return out


def _binance_filter_value(
    filters: Any,
    filter_type: str,
    key: str,
) -> Any:
    """
    Extract one value from Binance `filters[]` by filter type and key.

    Parameters:
    - filters: Binance filter list payload.
    - filter_type: target filter type string.
    - key: field key inside filter object.

    Returns:
    - Raw filter value or `None` when not found.

    Assumptions/Invariants:
    - Filter entries are dictionaries with `filterType` key.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if not isinstance(filters, list):
        return None
    for item in filters:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("filterType", "")).upper() != filter_type.upper():
            continue
        return item.get(key)
    return None


def _as_non_empty_string(value: Any) -> str | None:
    """
    Normalize raw value into non-empty stripped string.

    Parameters:
    - value: raw payload value.

    Returns:
    - Stripped string or `None` when empty/invalid.

    Assumptions/Invariants:
    - Non-string values are cast via `str(value)` when possible.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _as_positive_float(value: Any) -> float | None:
    """
    Convert raw payload value into positive float.

    Parameters:
    - value: raw value from exchange payload.

    Returns:
    - Positive float or `None` when value is missing/non-positive/non-numeric.

    Assumptions/Invariants:
    - Numeric strings are accepted.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed
