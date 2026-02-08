from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from typing import Any, Iterator
from uuid import UUID

from trading.contexts.market_data.adapters.outbound.clients.common_http import HttpClient
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    MarketDataRuntimeConfig,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.candle_ingest_source import (
    CandleIngestSource,
)
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    TimeRange,
    UtcTimestamp,
)


@dataclass(frozen=True, slots=True)
class RestCandleIngestSource(CandleIngestSource):
    """
    REST CandleIngestSource для 4 рынков (binance/bybit × spot/futures).

    Важно:
    - отдаёт только CLOSED 1m свечи по семантике TimeRange: [start, end)
    - CandleMeta.source = "rest"
    - ingested_at = clock.now() (UTC, ms)
    - ingest_id = фиксированный UUID на весь запуск/сессию
    - instrument_key = "{exchange}:{market_type}:{symbol}" (из runtime config)
    """

    cfg: MarketDataRuntimeConfig
    clock: Clock
    http: HttpClient
    ingest_id: UUID

    def stream_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        market = self.cfg.market_by_id(instrument_id.market_id)
        exch = market.exchange
        mtype = market.market_type

        symbol = str(instrument_id.symbol)
        instrument_key = f"{market.exchange}:{market.market_type}:{symbol}"

        if exch == "binance":
            yield from self._stream_binance_klines_1m(
                market_base=market.rest.base_url,
                market_type=mtype,
                instrument_id=instrument_id,
                instrument_key=instrument_key,
                time_range=time_range,
            )
            return

        if exch == "bybit":
            category = "spot" if mtype == "spot" else "linear"
            yield from self._stream_bybit_kline_1m(
                market_base=market.rest.base_url,
                category=category,
                instrument_id=instrument_id,
                instrument_key=instrument_key,
                time_range=time_range,
            )
            return

        raise ValueError(f"Unsupported exchange={exch!r}")

    # ---------------- BINANCE ----------------

    def _stream_binance_klines_1m(
        self,
        *,
        market_base: str,
        market_type: str,
        instrument_id: InstrumentId,
        instrument_key: str,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        # spot:   GET /api/v3/klines
        # futures GET /fapi/v1/klines (USD-M)
        path = "/fapi/v1/klines" if market_type == "futures" else "/api/v3/klines"

        start = _ensure_tz_utc(time_range.start.value)
        end = _ensure_tz_utc(time_range.end.value)

        market = self.cfg.market_by_id(instrument_id.market_id)
        timeout_s = market.rest.timeout_s
        retries = market.rest.retries
        backoff = market.rest.backoff

        cursor = start
        max_limit = 1000  # общий безопасный лимит

        while cursor < end:
            # Binance endTime трактуем как inclusive => end_ms-1 сохраняет [start,end)
            window_end = min(end, cursor + timedelta(minutes=max_limit))
            start_ms = int(cursor.timestamp() * 1000)
            end_ms = int(window_end.timestamp() * 1000) - 1

            url = market_base.rstrip("/") + path
            resp = self.http.get_json(
                url=url,
                params={
                    "symbol": str(instrument_id.symbol),
                    "interval": "1m",
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": max_limit,
                },
                timeout_s=timeout_s,
                retries=retries,
                backoff_base_s=backoff.base_s,
                backoff_max_s=backoff.max_s,
                backoff_jitter_s=backoff.jitter_s,
            )

            body = resp.body
            if not isinstance(body, list):
                raise RuntimeError(
                    f"Unexpected Binance klines payload type: {type(body).__name__}"
                )

            if not body:
                cursor = window_end
                continue

            last_open_dt = None
            for item in body:
                row = self._map_binance_kline_item(
                    instrument_id=instrument_id,
                    instrument_key=instrument_key,
                    item=item,
                )

                # фильтр по [start,end) и внутри текущего окна
                if row.candle.ts_open.value < start or row.candle.ts_open.value >= end:
                    continue
                if row.candle.ts_open.value >= window_end:
                    continue

                last_open_dt = row.candle.ts_open.value
                yield row

            cursor = window_end if last_open_dt is None else last_open_dt + timedelta(minutes=1)

    def _map_binance_kline_item(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        item: Any,
    ) -> CandleWithMeta:
        # Binance kline item:
        # [
        #   open_time, open, high, low, close, volume, close_time,
        #   quote_asset_volume, number_of_trades,
        #   taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ...
        # ]
        if not isinstance(item, list) or len(item) < 11:
            raise RuntimeError(f"Invalid Binance kline item: {item!r}")

        open_time_ms = int(item[0])
        open_dt = _dt_from_ms(open_time_ms)
        ts_open = UtcTimestamp(open_dt)
        ts_close = UtcTimestamp(open_dt + timedelta(minutes=1))

        candle = Candle(
            instrument_id=instrument_id,
            ts_open=ts_open,
            ts_close=ts_close,
            open=float(item[1]),
            high=float(item[2]),
            low=float(item[3]),
            close=float(item[4]),
            volume_base=float(item[5]),
            volume_quote=float(item[7]),
        )

        meta = CandleMeta(
            source="rest",
            ingested_at=self.clock.now(),
            ingest_id=self.ingest_id,
            instrument_key=instrument_key,
            trades_count=int(item[8]),
            taker_buy_volume_base=float(item[9]),
            taker_buy_volume_quote=float(item[10]),
        )

        return CandleWithMeta(candle=candle, meta=meta)

    # ---------------- BYBIT ----------------

    def _stream_bybit_kline_1m(
        self,
        *,
        market_base: str,
        category: str,
        instrument_id: InstrumentId,
        instrument_key: str,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        # GET /v5/market/kline
        path = "/v5/market/kline"

        start = _ensure_tz_utc(time_range.start.value)
        end = _ensure_tz_utc(time_range.end.value)

        market = self.cfg.market_by_id(instrument_id.market_id)
        timeout_s = market.rest.timeout_s
        retries = market.rest.retries
        backoff = market.rest.backoff

        cursor = start
        max_limit = 1000

        while cursor < end:
            window_end = min(end, cursor + timedelta(minutes=max_limit))
            start_ms = int(cursor.timestamp() * 1000)
            end_ms = int(window_end.timestamp() * 1000) - 1

            url = market_base.rstrip("/") + path
            resp = self.http.get_json(
                url=url,
                params={
                    "category": category,
                    "symbol": str(instrument_id.symbol),
                    "interval": "1",
                    "start": start_ms,
                    "end": end_ms,
                    "limit": max_limit,
                },
                timeout_s=timeout_s,
                retries=retries,
                backoff_base_s=backoff.base_s,
                backoff_max_s=backoff.max_s,
                backoff_jitter_s=backoff.jitter_s,
            )

            body = resp.body
            if not isinstance(body, dict):
                raise RuntimeError(f"Unexpected Bybit response type: {type(body).__name__}")

            ret_code = body.get("retCode")
            if ret_code not in (0, "0", None):
                raise RuntimeError(f"Bybit retCode={ret_code!r}, body={body!r}")

            result = body.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"Unexpected Bybit result: {result!r}")

            lst = result.get("list")
            if not isinstance(lst, list):
                raise RuntimeError(f"Unexpected Bybit result.list: {lst!r}")

            if not lst:
                cursor = window_end
                continue

            # Bybit часто отдаёт list по убыванию startTime -> переворачиваем
            lst_sorted = list(reversed(lst))

            last_open_dt = None
            for item in lst_sorted:
                row = self._map_bybit_kline_item(
                    instrument_id=instrument_id,
                    instrument_key=instrument_key,
                    item=item,
                )

                if row.candle.ts_open.value < start or row.candle.ts_open.value >= end:
                    continue
                if row.candle.ts_open.value >= window_end:
                    continue

                last_open_dt = row.candle.ts_open.value
                yield row

            cursor = window_end if last_open_dt is None else last_open_dt + timedelta(minutes=1)

    def _map_bybit_kline_item(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        item: Any,
    ) -> CandleWithMeta:
        # Bybit kline item (V5):
        # ["startTime","open","high","low","close","volume","turnover"]
        if not isinstance(item, list) or len(item) < 7:
            raise RuntimeError(f"Invalid Bybit kline item: {item!r}")

        start_ms = int(item[0])
        open_dt = _dt_from_ms(start_ms)
        ts_open = UtcTimestamp(open_dt)
        ts_close = UtcTimestamp(open_dt + timedelta(minutes=1))

        candle = Candle(
            instrument_id=instrument_id,
            ts_open=ts_open,
            ts_close=ts_close,
            open=float(item[1]),
            high=float(item[2]),
            low=float(item[3]),
            close=float(item[4]),
            volume_base=float(item[5]),
            volume_quote=float(item[6]),  # turnover
        )

        meta = CandleMeta(
            source="rest",
            ingested_at=self.clock.now(),
            ingest_id=self.ingest_id,
            instrument_key=instrument_key,
            trades_count=None,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        )

        return CandleWithMeta(candle=candle, meta=meta)


def _dt_from_ms(ms: int):
    return _ensure_tz_utc(_datetime_from_ts(ms / 1000.0))


def _datetime_from_ts(ts: float):
    # отдельная функция, чтобы не тянуть datetime в type hints наружу
    from datetime import datetime

    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _ensure_tz_utc(dt):
    if getattr(dt, "tzinfo", None) is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
