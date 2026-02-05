from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any, Iterator, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.stores.canonical_candle_reader import (
    CanonicalCandleReader,
)
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


class ClickHouseCanonicalCandleReader(CanonicalCandleReader):
    """
    Читает market_data.canonical_candles_1m.

    Dedup rule:
    - дедуп только на хвосте последних 24 часов относительно clock.now()
    - без FINAL
    - например: ORDER BY ingested_at DESC LIMIT 1 BY (market_id, symbol, ts_open)
    """

    def __init__(self, gateway: ClickHouseGateway, clock: Clock, database: str = "market_data") -> None:  # noqa: E501
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseCanonicalCandleReader requires gateway")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseCanonicalCandleReader requires clock")
        if not database.strip():
            raise ValueError("ClickHouseCanonicalCandleReader requires non-empty database")

        self._gw = gateway
        self._clock = clock
        self._db = database.strip()

    def read_1m(self, instrument_id: InstrumentId, time_range: TimeRange) -> Iterator[CandleWithMeta]:  # noqa: E501
        cutoff = self._cutoff_utc()
        start = time_range.start.value
        end = time_range.end.value

        old_end = min(end, cutoff)
        tail_start = max(start, cutoff)

        # 1) "Старая" часть — без дедупа
        if start < old_end:
            for row in self._select_no_dedup(instrument_id, start, old_end):
                yield row

        # 2) Хвост 24h — с дедупом
        if tail_start < end:
            for row in self._select_tail_dedup(instrument_id, tail_start, end):
                yield row

    def _cutoff_utc(self):
        now = self._clock.now().value
        now_utc = _ensure_tz_utc(now)
        return (now_utc - timedelta(hours=24))

    def _canonical_table(self) -> str:
        return f"{self._db}.canonical_candles_1m"

    def _select_no_dedup(self, instrument_id: InstrumentId, start_dt, end_dt) -> Sequence[CandleWithMeta]: # noqa: E501
        q = f"""
        SELECT
            market_id, symbol, instrument_key,
            ts_open, ts_close,
            open, high, low, close,
            volume_base, volume_quote,
            trades_count, taker_buy_volume_base, taker_buy_volume_quote,
            source, ingested_at, ingest_id
        FROM {self._canonical_table()}
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
          AND ts_open >= %(start)s
          AND ts_open < %(end)s
        ORDER BY ts_open
        """
        rows = self._gw.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "start": _ensure_tz_utc(start_dt),
                "end": _ensure_tz_utc(end_dt),
            },
        )
        return [self._map_row(r) for r in rows]

    def _select_tail_dedup(self, instrument_id: InstrumentId, start_dt, end_dt) -> Sequence[CandleWithMeta]: # noqa: E501
        # Дедуп: берём последнюю версию по ingested_at на ключ (market_id, symbol, ts_open)
        q = f"""
        SELECT
            market_id, symbol, instrument_key,
            ts_open, ts_close,
            open, high, low, close,
            volume_base, volume_quote,
            trades_count, taker_buy_volume_base, taker_buy_volume_quote,
            source, ingested_at, ingest_id
        FROM
        (
            SELECT *
            FROM {self._canonical_table()}
            WHERE market_id = %(market_id)s
              AND symbol = %(symbol)s
              AND ts_open >= %(start)s
              AND ts_open < %(end)s
            ORDER BY ingested_at DESC
            LIMIT 1 BY market_id, symbol, ts_open
        )
        ORDER BY ts_open
        """
        rows = self._gw.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "start": _ensure_tz_utc(start_dt),
                "end": _ensure_tz_utc(end_dt),
            },
        )
        return [self._map_row(r) for r in rows]

    def _map_row(self, r: Mapping[str, Any]) -> CandleWithMeta:
        instrument = InstrumentId(MarketId(int(r["market_id"])), Symbol(str(r["symbol"])))

        ts_open = UtcTimestamp(_ensure_tz_utc(r["ts_open"]))
        ts_close = UtcTimestamp(_ensure_tz_utc(r["ts_close"]))

        candle = Candle(
            instrument_id=instrument,
            ts_open=ts_open,
            ts_close=ts_close,
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume_base=float(r["volume_base"]),
            volume_quote=(float(r["volume_quote"]) if r["volume_quote"] is not None else None),
        )

        meta = CandleMeta(
            source=str(r["source"]),
            ingested_at=UtcTimestamp(_ensure_tz_utc(r["ingested_at"])),
            ingest_id=r.get("ingest_id"),
            instrument_key=str(r["instrument_key"]),
            trades_count=(int(r["trades_count"]) if r["trades_count"] is not None else None),
            taker_buy_volume_base=(
                float(r["taker_buy_volume_base"]) if r["taker_buy_volume_base"] is not None else None # noqa: E501
            ),
            taker_buy_volume_quote=(
                float(r["taker_buy_volume_quote"]) if r["taker_buy_volume_quote"] is not None else None # noqa: E501
            ),
        )

        return CandleWithMeta(candle=candle, meta=meta)


def _ensure_tz_utc(dt) -> Any:
    if getattr(dt, "tzinfo", None) is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
