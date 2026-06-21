from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from uuid import UUID

from trading.contexts.market_data.adapters.outbound.clients.common_http import HttpClient
from trading.contexts.market_data.adapters.outbound.config.instrument_key import (
    build_instrument_key,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    MarketConfig,
    MarketDataRuntimeConfig,
)
from trading.contexts.market_data.application.dto import FundingInstrument, FundingRateRecord
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.funding_instrument_universe_source import (  # noqa: E501
    FundingInstrumentUniverseSource,
)
from trading.contexts.market_data.application.ports.sources.funding_rate_history_source import (
    FundingRateHistorySource,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

BINANCE_STANDARD_8H_NO_ADJUSTMENT_ROW = "binance_standard_8h_no_adjustment_row"
BINANCE_STANDARD_8H_EMERGENCY_FALLBACK = "binance_standard_8h_emergency_fallback"
BYBIT_FUTURES_CATEGORY = "linear"


@dataclass(frozen=True, slots=True)
class RestFundingRateHistorySource(FundingRateHistorySource, FundingInstrumentUniverseSource):
    cfg: MarketDataRuntimeConfig
    http: HttpClient
    clock: Clock
    ingest_id: UUID
    binance_standard_interval_hours: int = 8
    allow_binance_funding_info_failure_fallback: bool = False

    def list_funding_instruments(self, market_id: MarketId) -> Sequence[FundingInstrument]:
        market = self.cfg.market_by_id(market_id)
        if market.market_type != "futures":
            return ()
        if market.exchange == "binance":
            return self._list_binance_funding_universe(market)
        if market.exchange == "bybit":
            return self._list_bybit_funding_universe(market)
        raise ValueError(f"unsupported funding universe exchange={market.exchange!r}")

    def list_funding_rates(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
        funding_interval_minutes: int,
        funding_interval_source: str,
    ) -> Sequence[FundingRateRecord]:
        market = self.cfg.market_by_id(instrument_id.market_id)
        if market.market_type != "futures":
            raise ValueError("funding history is only supported for futures markets")
        if market.exchange == "binance":
            return self._list_binance_funding_rates(
                market=market,
                instrument_id=instrument_id,
                time_range=time_range,
                funding_interval_minutes=funding_interval_minutes,
                funding_interval_source=funding_interval_source,
            )
        if market.exchange == "bybit":
            return self._list_bybit_funding_rates(
                market=market,
                instrument_id=instrument_id,
                time_range=time_range,
                funding_interval_minutes=funding_interval_minutes,
                funding_interval_source=funding_interval_source,
            )
        raise ValueError(f"unsupported funding history exchange={market.exchange!r}")

    def _list_binance_funding_universe(self, market: MarketConfig) -> list[FundingInstrument]:
        symbols = self._fetch_binance_tradable_perpetuals(market)
        adjusted_by_symbol = self._fetch_binance_funding_info(market, symbols)
        now = self.clock.now()
        out: list[FundingInstrument] = []
        for item in symbols:
            symbol = str(item["symbol"])
            adjusted = adjusted_by_symbol.get(symbol.upper())
            if adjusted is None:
                interval_minutes = self.binance_standard_interval_hours * 60
                interval_source = BINANCE_STANDARD_8H_NO_ADJUSTMENT_ROW
                cap = floor = None
            else:
                interval_minutes = int(adjusted["fundingIntervalHours"]) * 60
                interval_source = str(
                    adjusted.get("funding_interval_source") or "binance_fundingInfo"
                )
                cap = _optional_float(adjusted.get("adjustedFundingRateCap"))
                floor = _optional_float(adjusted.get("adjustedFundingRateFloor"))

            instrument_id = InstrumentId(market.market_id, Symbol(symbol))
            out.append(
                FundingInstrument(
                    instrument_id=instrument_id,
                    instrument_key=build_instrument_key(cfg=self.cfg, instrument_id=instrument_id),
                    exchange=market.exchange,
                    market_type=market.market_type,
                    status=str(item.get("status") or "TRADING"),
                    is_tradable=1,
                    base_asset=_optional_str(item.get("baseAsset")),
                    quote_asset=_optional_str(item.get("quoteAsset")),
                    funding_interval_minutes=interval_minutes,
                    funding_interval_source=interval_source,
                    funding_cap=cap,
                    funding_floor=floor,
                    updated_at=now,
                )
            )
        return out

    def _fetch_binance_tradable_perpetuals(self, market: MarketConfig) -> list[Mapping[str, Any]]:
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
            raise RuntimeError("Unexpected Binance exchangeInfo payload")
        rows = body.get("symbols")
        if not isinstance(rows, list):
            raise RuntimeError("Unexpected Binance exchangeInfo payload: missing symbols")
        out: list[Mapping[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = _optional_str(item.get("symbol"))
            if symbol is None:
                continue
            if str(item.get("status") or "").upper() != "TRADING":
                continue
            if str(item.get("contractType") or "").upper() != "PERPETUAL":
                continue
            out.append(item)
        return out

    def _fetch_binance_funding_info(
        self,
        market: MarketConfig,
        symbols: Sequence[Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        try:
            response = self.http.get_json(
                url=market.rest.base_url.rstrip("/") + "/fapi/v1/fundingInfo",
                params={},
                timeout_s=market.rest.timeout_s,
                retries=market.rest.retries,
                backoff_base_s=market.rest.backoff.base_s,
                backoff_max_s=market.rest.backoff.max_s,
                backoff_jitter_s=market.rest.backoff.jitter_s,
            )
        except Exception:
            if not self.allow_binance_funding_info_failure_fallback:
                raise
            fallback: dict[str, Mapping[str, Any]] = {}
            for item in symbols:
                symbol = _optional_str(item.get("symbol"))
                if symbol is None:
                    continue
                fallback[symbol.upper()] = {
                    "symbol": symbol,
                    "fundingIntervalHours": self.binance_standard_interval_hours,
                    "funding_interval_source": BINANCE_STANDARD_8H_EMERGENCY_FALLBACK,
                }
            return fallback

        body = response.body
        if not isinstance(body, list):
            raise RuntimeError("Unexpected Binance fundingInfo payload")
        out: dict[str, Mapping[str, Any]] = {}
        for item in body:
            if not isinstance(item, dict):
                continue
            symbol = _optional_str(item.get("symbol"))
            interval = item.get("fundingIntervalHours")
            if symbol is None or interval is None:
                continue
            out[symbol.upper()] = item
        return out

    def _list_bybit_funding_universe(self, market: MarketConfig) -> list[FundingInstrument]:
        category = BYBIT_FUTURES_CATEGORY
        cursor = ""
        out: list[FundingInstrument] = []
        now = self.clock.now()
        while True:
            params: dict[str, Any] = {"category": category, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            body = self._bybit_get(market=market, path="/v5/market/instruments-info", params=params)
            result = body.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Unexpected Bybit instruments-info payload")
            rows = result.get("list")
            if not isinstance(rows, list):
                raise RuntimeError("Unexpected Bybit instruments-info list")
            for item in rows:
                if not isinstance(item, dict):
                    continue
                symbol = _optional_str(item.get("symbol"))
                if symbol is None:
                    continue
                status = str(item.get("status") or "Trading")
                if status.upper() != "TRADING":
                    continue
                raw_interval = _optional_int(item.get("fundingInterval"))
                interval = raw_interval if raw_interval is not None and raw_interval > 0 else None
                interval_source = "bybit_instruments_info" if interval is not None else None
                instrument_id = InstrumentId(market.market_id, Symbol(symbol))
                out.append(
                    FundingInstrument(
                        instrument_id=instrument_id,
                        instrument_key=build_instrument_key(
                            cfg=self.cfg,
                            instrument_id=instrument_id,
                        ),
                        exchange=market.exchange,
                        market_type=market.market_type,
                        status=status,
                        is_tradable=1,
                        base_asset=_optional_str(item.get("baseCoin")),
                        quote_asset=_optional_str(item.get("quoteCoin")),
                        funding_interval_minutes=interval,
                        funding_interval_source=interval_source,
                        funding_cap=_optional_float(item.get("upperFundingRate")),
                        funding_floor=_optional_float(item.get("lowerFundingRate")),
                        updated_at=now,
                    )
                )
            next_cursor = _optional_str(result.get("nextPageCursor"))
            if next_cursor is None or next_cursor == cursor:
                break
            cursor = next_cursor
        return out

    def _list_binance_funding_rates(
        self,
        *,
        market: MarketConfig,
        instrument_id: InstrumentId,
        time_range: TimeRange,
        funding_interval_minutes: int,
        funding_interval_source: str,
    ) -> list[FundingRateRecord]:
        url = market.rest.base_url.rstrip("/") + "/fapi/v1/fundingRate"
        start = time_range.start.value
        end = time_range.end.value
        step = timedelta(minutes=funding_interval_minutes * 1000)
        cursor = start
        out: list[FundingRateRecord] = []
        while cursor < end:
            window_start = cursor
            window_end = min(end, cursor + step)
            response = self.http.get_json(
                url=url,
                params={
                    "symbol": str(instrument_id.symbol),
                    "startTime": _dt_to_epoch_ms(cursor),
                    "endTime": _dt_to_epoch_ms(window_end) - 1,
                    "limit": 1000,
                },
                timeout_s=market.rest.timeout_s,
                retries=market.rest.retries,
                backoff_base_s=market.rest.backoff.base_s,
                backoff_max_s=market.rest.backoff.max_s,
                backoff_jitter_s=market.rest.backoff.jitter_s,
            )
            body = response.body
            if not isinstance(body, list):
                raise RuntimeError("Unexpected Binance fundingRate payload")
            last_seen = None
            for item in body:
                if not isinstance(item, dict):
                    continue
                funding_time = _dt_from_ms(int(item["fundingTime"]))
                if not (window_start <= funding_time < window_end):
                    continue
                last_seen = funding_time
                out.append(
                    FundingRateRecord(
                        instrument_id=instrument_id,
                        instrument_key=build_instrument_key(
                            cfg=self.cfg,
                            instrument_id=instrument_id,
                        ),
                        funding_time=UtcTimestamp(funding_time),
                        funding_rate=float(item["fundingRate"]),
                        funding_interval_minutes=funding_interval_minutes,
                        funding_interval_source=funding_interval_source,
                        source="binance_fundingRate",
                        ingested_at=self.clock.now(),
                        ingest_id=str(self.ingest_id),
                        mark_price=_optional_float(item.get("markPrice")),
                    )
                )
            cursor = window_end if last_seen is None else last_seen + timedelta(milliseconds=1)
        return out

    def _list_bybit_funding_rates(
        self,
        *,
        market: MarketConfig,
        instrument_id: InstrumentId,
        time_range: TimeRange,
        funding_interval_minutes: int,
        funding_interval_source: str,
    ) -> list[FundingRateRecord]:
        start = time_range.start.value
        end = time_range.end.value
        step = timedelta(minutes=funding_interval_minutes * 200)
        cursor = start
        out: list[FundingRateRecord] = []
        while cursor < end:
            window_start = cursor
            window_end = min(end, cursor + step)
            body = self._bybit_get(
                market=market,
                path="/v5/market/funding/history",
                params={
                    "category": BYBIT_FUTURES_CATEGORY,
                    "symbol": str(instrument_id.symbol),
                    "startTime": _dt_to_epoch_ms(cursor),
                    "endTime": _dt_to_epoch_ms(window_end) - 1,
                    "limit": 200,
                },
            )
            result = body.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Unexpected Bybit funding history payload")
            rows = result.get("list")
            if not isinstance(rows, list):
                raise RuntimeError("Unexpected Bybit funding history list")
            sorted_rows = sorted(
                [item for item in rows if isinstance(item, dict)],
                key=lambda item: int(item["fundingRateTimestamp"]),
            )
            last_seen = None
            for item in sorted_rows:
                funding_time = _dt_from_ms(int(item["fundingRateTimestamp"]))
                if not (window_start <= funding_time < window_end):
                    continue
                last_seen = funding_time
                out.append(
                    FundingRateRecord(
                        instrument_id=instrument_id,
                        instrument_key=build_instrument_key(
                            cfg=self.cfg,
                            instrument_id=instrument_id,
                        ),
                        funding_time=UtcTimestamp(funding_time),
                        funding_rate=float(item["fundingRate"]),
                        funding_interval_minutes=funding_interval_minutes,
                        funding_interval_source=funding_interval_source,
                        source="bybit_funding_history",
                        ingested_at=self.clock.now(),
                        ingest_id=str(self.ingest_id),
                        bybit_category=BYBIT_FUTURES_CATEGORY,
                    )
                )
            cursor = window_end if last_seen is None else last_seen + timedelta(milliseconds=1)
        return out

    def _bybit_get(
        self,
        *,
        market: MarketConfig,
        path: str,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
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
            raise RuntimeError("Unexpected Bybit payload type")
        ret_code = body.get("retCode")
        if ret_code not in (0, "0", None):
            raise RuntimeError(f"Bybit retCode={ret_code!r}")
        return body


def _dt_to_epoch_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _dt_from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
