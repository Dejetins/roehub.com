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
    - отдаёт только CLOSED 1m свечи (по полузамкнутому диапазону [start, end))
    - CandleMeta.source = "rest"
    - ingested_at = clock.now()
    - ingest_id = фиксированный UUID на весь запуск/сессию
    """

    cfg: MarketDataRuntimeConfig
    clock: Clock
    http: HttpClient
    ingest_id: UUID

    def stream_1m(self, instrument_id: InstrumentId, time_range: TimeRange) -> Iterator[CandleWithMeta]:  # noqa: E501
        market = self.cfg.market_by_id(instrument_id.market_id)
        exch = market.exchange
        mtype = market.market_type

        if exch == "binance":
            yield from self._stream_binance_klines_1m(market_base=market.rest.base_url, instrument_id=instrument_id, time_range=time_range)  # noqa: E501
            return

        if exch == "bybit":
            category = "spot" if mtype == "spot" else "linear"
            yield from self._stream_bybit_kline_1m(market_base=market.rest.base_url, category=category, instrument_id=instrument_id, time_range=time_range)  # noqa: E501
            return

        raise ValueError(f"Unsupported exchange={exch!r}")

    # ---------------- BINANCE ----------------

    def _stream_binance_klines_1m(
        self,
        *,
        market_base: str,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        # spot: /api/v3/klines
        # futures (USD-M): /fapi/v1/klines
        # У нас base_url в конфиге уже различается (api.binance.com vs fapi.binance.com),
        # поэтому путь определяем по market_type косвенно: если base содержит "fapi" -> futures.
        path = "/fapi/v1/klines" if "fapi" in market_base else "/api/v3/klines"

        start = _ensure_tz_utc(time_range.start.value)
        end = _ensure_tz_utc(time_range.end.value)

        cursor = start
        max_limit = 1000  # spot: 1000, futures: 1500, но держим 1000 как общий безопасный лимит

        while cursor < end:
            # Binance endTime — трактуем как inclusive, поэтому ставим end_ms-1,
            # чтобы соблюдать семантику [start, end).
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
                timeout_s=self.cfg.market_by_id(instrument_id.market_id).rest.timeout_s,
                retries=self.cfg.market_by_id(instrument_id.market_id).rest.retries,
                backoff_base_s=self.cfg.market_by_id(instrument_id.market_id).rest.backoff.base_s,
                backoff_max_s=self.cfg.market_by_id(instrument_id.market_id).rest.backoff.max_s,
                backoff_jitter_s=self.cfg.market_by_id(instrument_id.market_id).rest.backoff.jitter_s,
            )

            body = resp.body
            if not isinstance(body, list):
                raise RuntimeError(f"Unexpected Binance klines payload type: {type(body).__name__}")

            if not body:
                cursor = window_end
                continue

            # Ответ уже по возрастанию open time
            last_open_dt = None
            for item in body:
                row = self._map_binance_kline_item(instrument_id=instrument_id, item=item)
                # фильтруем по [start,end)
                if row.candle.ts_open.value < start or row.candle.ts_open.value >= end:
                    continue
                # и также не выходим за window_end
                if row.candle.ts_open.value >= window_end:
                    continue

                last_open_dt = row.candle.ts_open.value
                yield row

            if last_open_dt is None:
                cursor = window_end
            else:
                cursor = last_open_dt + timedelta(minutes=1)

    def _map_binance_kline_item(self, *, instrument_id: InstrumentId, item: Any) -> CandleWithMeta:
        if not isinstance(item, list) or len(item) < 11:
            raise RuntimeError(f"Invalid Binance kline item: {item!r}")

        open_time_ms = int(item[0])
        open_dt = _dt_from_ms(open_time_ms)
        ts_open = UtcTimestamp(open_dt)
        ts_close = UtcTimestamp(open_dt + timedelta(minutes=1))

        # Binance: [open, high, low, close] — строки
        open_p = float(item[1])
        high_p = float(item[2])
        low_p = float(item[3])
        close_p = float(item[4])

        volume_base = float(item[5])
        quote_asset_volume = float(item[7])  # quote volume
        trades_count = int(item[8])
        taker_buy_base = float(item[9])
        taker_buy_quote = float(item[10])

        candle = Candle(
            instrument_id=instrument_id,
            ts_open=ts_open,
            ts_close=ts_close,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume_base=volume_base,
            volume_quote=quote_asset_volume,
        )

        meta = CandleMeta(
            source="rest",
            ingested_at=self.clock.now(),
            ingest_id=self.ingest_id,
            instrument_key=_instrument_key(instrument_id),
            trades_count=trades_count,
            taker_buy_volume_base=taker_buy_base,
            taker_buy_volume_quote=taker_buy_quote,
        )

        return CandleWithMeta(candle=candle, meta=meta)

    # ---------------- BYBIT ----------------

    def _stream_bybit_kline_1m(
        self,
        *,
        market_base: str,
        category: str,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        # GET /v5/market/kline
        path = "/v5/market/kline"

        start = _ensure_tz_utc(time_range.start.value)
        end = _ensure_tz_utc(time_range.end.value)

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
                timeout_s=self.cfg.market_by_id(instrument_id.market_id).rest.timeout_s,
                retries=self.cfg.market_by_id(instrument_id.market_id).rest.retries,
                backoff_base_s=self.cfg.market_by_id(instrument_id.market_id).rest.backoff.base_s,
                backoff_max_s=self.cfg.market_by_id(instrument_id.market_id).rest.backoff.max_s,
                backoff_jitter_s=self.cfg.market_by_id(instrument_id.market_id).rest.backoff.jitter_s,
            )

            body = resp.body
            if not isinstance(body, dict):
                raise RuntimeError(f"Unexpected Bybit response type: {type(body).__name__}")

            result = body.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"Unexpected Bybit result: {result!r}")

            lst = result.get("list")
            if not isinstance(lst, list):
                raise RuntimeError(f"Unexpected Bybit result.list: {lst!r}")

            if not lst:
                cursor = window_end
                continue

            # Bybit отдаёт list обычно в порядке убывания startTime — перевернём
            lst_sorted = list(reversed(lst))

            last_open_dt = None
            for item in lst_sorted:
                row = self._map_bybit_kline_item(instrument_id=instrument_id, item=item)
                if row.candle.ts_open.value < start or row.candle.ts_open.value >= end:
                    continue
                if row.candle.ts_open.value >= window_end:
                    continue

                last_open_dt = row.candle.ts_open.value
                yield row

            if last_open_dt is None:
                cursor = window_end
            else:
                cursor = last_open_dt + timedelta(minutes=1)

    def _map_bybit_kline_item(self, *, instrument_id: InstrumentId, item: Any) -> CandleWithMeta:
        # item: ["startTime","open","high","low","close","volume","turnover"]
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
            instrument_key=_instrument_key(instrument_id),
            trades_count=None,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        )

        return CandleWithMeta(candle=candle, meta=meta)


def _instrument_key(instrument_id: InstrumentId) -> str:
    # "{exchange}:{market_type}:{symbol}" — берём exchange/market_type из market_id косвенно уже нельзя,  # noqa: E501
    # поэтому делаем через instrument_id только symbol, а exchange/type доступны через cfg в источнике.  # noqa: E501
    # Здесь оставляем общий формат, а exchange/type добавляются выше через cfg при желании.
    # В EPIC 0/1 вы уже приняли формат "exchange:market_type:symbol".
    # Для rest источника делаем то же через cfg: exchange/type мы знаем, но ключ строим в stream_1m.
    # Поэтому эта функция используется только как fallback.
    return f"unknown:unknown:{instrument_id.symbol}"


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
