from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Mapping

from trading.contexts.market_data.adapters.outbound.clients.common_http import HttpClient
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    MarketConfig,
    MarketDataRuntimeConfig,
)
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.instrument_history_start_source import (
    InstrumentHistoryStartSource,
)
from trading.contexts.market_data.application.services.minute_utils import floor_to_minute_utc
from trading.shared_kernel.primitives import InstrumentId, UtcTimestamp

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RestInstrumentHistoryStartSource(InstrumentHistoryStartSource):
    """
    Resolve symbol-specific history starts from exchange REST metadata endpoints.

    Strategy:
    - Binance futures: `/fapi/v1/exchangeInfo` -> `symbols[].onboardDate`
    - Bybit futures: `/v5/market/instruments-info` -> `list[].launchTime`
    - Binance/Bybit spot: earliest non-empty 1m kline window via binary search

    Notes:
    - Returns `None` when the exchange does not expose or confirm a symbol-specific start.
    - Callers must fall back to market-wide `earliest_available_ts_utc` on `None`.
    """

    cfg: MarketDataRuntimeConfig
    http: HttpClient
    clock: Clock
    _instrument_cache: dict[tuple[int, str], UtcTimestamp | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _binance_futures_market_cache: dict[int, dict[str, UtcTimestamp | None]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cfg is None:  # type: ignore[truthy-bool]
            raise ValueError("RestInstrumentHistoryStartSource requires cfg")
        if self.http is None:  # type: ignore[truthy-bool]
            raise ValueError("RestInstrumentHistoryStartSource requires http")
        if self.clock is None:  # type: ignore[truthy-bool]
            raise ValueError("RestInstrumentHistoryStartSource requires clock")

    def get_history_start(self, instrument_id: InstrumentId) -> UtcTimestamp | None:
        """
        Resolve one instrument's earliest confirmed exchange history minute.

        Parameters:
        - instrument_id: target instrument identity.

        Returns:
        - Minute-level UTC timestamp or `None` when no symbol-specific bound is known.

        Assumptions/Invariants:
        - Network or payload failures degrade to `None` instead of breaking scheduler planning.
        """
        cache_key = _cache_key(instrument_id)
        with self._lock:
            if cache_key in self._instrument_cache:
                return self._instrument_cache[cache_key]

        market = self.cfg.market_by_id(instrument_id.market_id)
        try:
            resolved = self._resolve_history_start(market=market, instrument_id=instrument_id)
            should_cache = True
        except Exception:  # noqa: BLE001
            log.warning(
                "history-start resolve failed for market=%s symbol=%s; "
                "falling back to market earliest",
                market.market_code,
                instrument_id.symbol,
                exc_info=True,
            )
            resolved = None
            should_cache = False

        if should_cache:
            with self._lock:
                self._instrument_cache[cache_key] = resolved
        return resolved

    def _resolve_history_start(
        self,
        *,
        market: MarketConfig,
        instrument_id: InstrumentId,
    ) -> UtcTimestamp | None:
        if market.exchange == "binance" and market.market_type == "futures":
            return self._resolve_binance_futures_history_start(market, instrument_id)
        if market.exchange == "bybit" and market.market_type == "futures":
            return self._resolve_bybit_futures_history_start(market, instrument_id)
        if market.exchange == "binance" and market.market_type == "spot":
            return self._probe_spot_history_start(
                market=market,
                instrument_id=instrument_id,
                request_factory=self._build_binance_spot_probe_request,
                response_has_rows=_binance_kline_response_has_rows,
            )
        if market.exchange == "bybit" and market.market_type == "spot":
            return self._probe_spot_history_start(
                market=market,
                instrument_id=instrument_id,
                request_factory=self._build_bybit_spot_probe_request,
                response_has_rows=_bybit_kline_response_has_rows,
            )
        raise ValueError(
            f"unsupported exchange/market_type for history start source: {market.exchange}/{market.market_type}"  # noqa: E501
        )

    def _resolve_binance_futures_history_start(
        self,
        market: MarketConfig,
        instrument_id: InstrumentId,
    ) -> UtcTimestamp | None:
        market_id_int = int(market.market_id.value)
        with self._lock:
            cached = self._binance_futures_market_cache.get(market_id_int)
        if cached is None:
            cached = self._load_binance_futures_market_cache(market)

        symbol_key = str(instrument_id.symbol).upper()
        resolved = cached.get(symbol_key)
        with self._lock:
            self._instrument_cache[_cache_key(instrument_id)] = resolved
        return resolved

    def _load_binance_futures_market_cache(
        self,
        market: MarketConfig,
    ) -> dict[str, UtcTimestamp | None]:
        response = self.http.get_json(
            url=market.rest.base_url.rstrip("/") + "/fapi/v1/exchangeInfo",
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
                f"Unexpected Binance futures exchangeInfo payload type: {type(body).__name__}"
            )
        rows = body.get("symbols")
        if not isinstance(rows, list):
            raise RuntimeError("Unexpected Binance futures exchangeInfo payload: missing symbols")

        out: dict[str, UtcTimestamp | None] = {}
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            symbol_text = _as_non_empty_string(item.get("symbol"))
            if symbol_text is None:
                continue
            out[symbol_text] = _utc_timestamp_from_ms(item.get("onboardDate"))

        market_id_int = int(market.market_id.value)
        with self._lock:
            existing = self._binance_futures_market_cache.get(market_id_int)
            if existing is not None:
                return existing
            self._binance_futures_market_cache[market_id_int] = out
            for symbol_key, history_start in out.items():
                self._instrument_cache[(market_id_int, symbol_key)] = history_start
        return out

    def _resolve_bybit_futures_history_start(
        self,
        market: MarketConfig,
        instrument_id: InstrumentId,
    ) -> UtcTimestamp | None:
        response = self.http.get_json(
            url=market.rest.base_url.rstrip("/") + "/v5/market/instruments-info",
            params={
                "category": "linear",
                "symbol": str(instrument_id.symbol),
            },
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

        symbol_key = str(instrument_id.symbol).upper()
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            if _as_non_empty_string(item.get("symbol")) != symbol_key:
                continue
            return _utc_timestamp_from_ms(item.get("launchTime"))
        return None

    def _probe_spot_history_start(
        self,
        *,
        market: MarketConfig,
        instrument_id: InstrumentId,
        request_factory: Callable[
            [MarketConfig, InstrumentId, int, int],
            tuple[str, dict[str, Any]],
        ],
        response_has_rows: Callable[[Any], bool],
    ) -> UtcTimestamp | None:
        earliest_start_minute = _minute_index(
            floor_to_minute_utc(market.rest.earliest_available_ts_utc.value)
        )
        now_floor_minute = _minute_index(floor_to_minute_utc(self.clock.now().value))
        if earliest_start_minute >= now_floor_minute:
            return None

        upper_probe_boundary = now_floor_minute + 1
        if not self._has_history_before(
            market=market,
            instrument_id=instrument_id,
            probe_end_minute=upper_probe_boundary,
            request_factory=request_factory,
            response_has_rows=response_has_rows,
        ):
            return None

        low = earliest_start_minute + 1
        high = upper_probe_boundary
        while low < high:
            mid = (low + high) // 2
            if self._has_history_before(
                market=market,
                instrument_id=instrument_id,
                probe_end_minute=mid,
                request_factory=request_factory,
                response_has_rows=response_has_rows,
            ):
                high = mid
            else:
                low = mid + 1

        return _utc_timestamp_from_minute_index(low - 1)

    def _has_history_before(
        self,
        *,
        market: MarketConfig,
        instrument_id: InstrumentId,
        probe_end_minute: int,
        request_factory: Callable[
            [MarketConfig, InstrumentId, int, int],
            tuple[str, dict[str, Any]],
        ],
        response_has_rows: Callable[[Any], bool],
    ) -> bool:
        start_ms = (
            _minute_index(floor_to_minute_utc(market.rest.earliest_available_ts_utc.value))
            * 60_000
        )
        end_ms = (probe_end_minute * 60_000) - 1
        if end_ms < start_ms:
            return False

        url, params = request_factory(market, instrument_id, start_ms, end_ms)
        response = self.http.get_json(
            url=url,
            params=params,
            timeout_s=market.rest.timeout_s,
            retries=market.rest.retries,
            backoff_base_s=market.rest.backoff.base_s,
            backoff_max_s=market.rest.backoff.max_s,
            backoff_jitter_s=market.rest.backoff.jitter_s,
        )
        return response_has_rows(response.body)

    @staticmethod
    def _build_binance_spot_probe_request(
        market: MarketConfig,
        instrument_id: InstrumentId,
        start_ms: int,
        end_ms: int,
    ) -> tuple[str, dict[str, Any]]:
        return (
            market.rest.base_url.rstrip("/") + "/api/v3/klines",
            {
                "symbol": str(instrument_id.symbol),
                "interval": "1m",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1,
            },
        )

    @staticmethod
    def _build_bybit_spot_probe_request(
        market: MarketConfig,
        instrument_id: InstrumentId,
        start_ms: int,
        end_ms: int,
    ) -> tuple[str, dict[str, Any]]:
        return (
            market.rest.base_url.rstrip("/") + "/v5/market/kline",
            {
                "category": "spot",
                "symbol": str(instrument_id.symbol),
                "interval": "1",
                "start": start_ms,
                "end": end_ms,
                "limit": 1,
            },
        )


def _cache_key(instrument_id: InstrumentId) -> tuple[int, str]:
    return (int(instrument_id.market_id.value), str(instrument_id.symbol).upper())


def _minute_index(dt: datetime) -> int:
    return int(dt.timestamp()) // 60


def _utc_timestamp_from_minute_index(minute_index: int) -> UtcTimestamp:
    return UtcTimestamp(datetime.fromtimestamp(minute_index * 60, tz=timezone.utc))


def _utc_timestamp_from_ms(value: Any) -> UtcTimestamp | None:
    if value is None:
        return None
    try:
        parsed_ms = int(value)
    except (TypeError, ValueError):
        return None
    if parsed_ms <= 0:
        return None
    dt = datetime.fromtimestamp(parsed_ms / 1000.0, tz=timezone.utc)
    return UtcTimestamp(floor_to_minute_utc(dt))


def _as_non_empty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _binance_kline_response_has_rows(body: Any) -> bool:
    if not isinstance(body, list):
        raise RuntimeError(f"Unexpected Binance kline payload type: {type(body).__name__}")
    return bool(body)


def _bybit_kline_response_has_rows(body: Any) -> bool:
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected Bybit kline payload type: {type(body).__name__}")

    ret_code = body.get("retCode")
    if ret_code not in (0, "0", None):
        raise RuntimeError(f"Bybit retCode={ret_code!r} for kline")

    result = body.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Unexpected Bybit kline payload: missing result mapping")

    rows = result.get("list")
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected Bybit kline payload: result.list must be a list")
    return bool(rows)
