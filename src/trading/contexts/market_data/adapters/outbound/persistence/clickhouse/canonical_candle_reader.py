from __future__ import annotations

from datetime import timezone
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    CanonicalCandleBatch1m,
)
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

    Read contract:
    - чтение идёт через `FINAL` по всему запрошенному диапазону;
    - это сохраняет strict monotonic source contract для artifact precompute even when
      `canonical_candles_1m` contains historical duplicates;
    - hot-path runtime does not use this reader, so the heavier `FINAL` path stays confined to
      offline/backfill/precompute workloads.
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

    def read_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        """
        Read canonical 1m candles in one deterministic `FINAL` query.

        Args:
            instrument_id: Explicit market/symbol identity to read.
            time_range: Half-open UTC range `[start, end)` for requested candles.
        Returns:
            Iterator[CandleWithMeta]: Canonical candles ordered by strictly increasing `ts_open`.
        Assumptions:
            `canonical_candles_1m FINAL` is the source of truth for offline precompute and may be
            heavier than non-`FINAL` reads.
        Raises:
            Exception: Propagates gateway/storage failures.
        Side Effects:
            Executes one ClickHouse `SELECT ... FINAL`.
        """
        for row in self._select_final(
            instrument_id=instrument_id,
            start_dt=time_range.start.value,
            end_dt=time_range.end.value,
        ):
            yield row

    def read_1m_arrays(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> CanonicalCandleBatch1m:
        """
        Read canonical `1m` candles as a columnar batch for offline/precompute workloads.

        Args:
            instrument_id: Explicit market/symbol identity to read.
            time_range: Half-open UTC range `[start, end)` for requested candles.
        Returns:
            CanonicalCandleBatch1m: Strict columnar payload ordered by increasing `ts_open`.
        Assumptions:
            Precompute workloads prefer numeric arrays over per-row DTO construction.
        Raises:
            Exception: Propagates gateway/storage failures.
        Side Effects:
            Executes one ClickHouse `SELECT ... FINAL`.
        """
        return self._select_final_arrays(
            instrument_id=instrument_id,
            start_dt=time_range.start.value,
            end_dt=time_range.end.value,
        )

    def _canonical_table(self) -> str:
        return f"{self._db}.canonical_candles_1m"

    def _select_final(
        self,
        instrument_id: InstrumentId,
        start_dt,
        end_dt,
    ) -> Sequence[CandleWithMeta]:
        """
        Execute one `FINAL` read against canonical 1m storage for the requested UTC range.

        Args:
            instrument_id: Explicit market/symbol identity to read.
            start_dt: Inclusive UTC range start.
            end_dt: Exclusive UTC range end.
        Returns:
            Sequence[CandleWithMeta]: Materialized canonical rows ordered by `ts_open`.
        Assumptions:
            Historical duplicates may still exist in storage, so artifact precompute must not rely
            on tail-only dedup.
        Raises:
            Exception: Propagates gateway failures.
        Side Effects:
            Executes one ClickHouse `SELECT ... FINAL`.
        """
        q = f"""
        SELECT
            market_id, symbol, instrument_key,
            ts_open, ts_close,
            open, high, low, close,
            volume_base, volume_quote,
            trades_count, taker_buy_volume_base, taker_buy_volume_quote,
            source, ingested_at, ingest_id
        FROM {self._canonical_table()} FINAL
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

    def _select_final_arrays(
        self,
        instrument_id: InstrumentId,
        start_dt,
        end_dt,
    ) -> CanonicalCandleBatch1m:
        """
        Execute one numeric `FINAL` read optimized for precompute columnar array materialization.

        Args:
            instrument_id: Explicit market/symbol identity to read.
            start_dt: Inclusive UTC range start.
            end_dt: Exclusive UTC range end.
        Returns:
            CanonicalCandleBatch1m: Strict columnar arrays for the requested range.
        Assumptions:
            ClickHouse computes UTC epoch milliseconds more efficiently than Python datetime
            object materialization for full-history artifact bootstrap runs.
        Raises:
            Exception: Propagates gateway failures.
        Side Effects:
            Executes one ClickHouse `SELECT ... FINAL`.
        """
        q = f"""
        SELECT
            toUnixTimestamp64Milli(ts_open) AS open_time_ms,
            toUnixTimestamp64Milli(ts_close) AS close_time_ms,
            toFloat32(open) AS open_f32,
            toFloat32(high) AS high_f32,
            toFloat32(low) AS low_f32,
            toFloat32(close) AS close_f32,
            toFloat32(volume_base) AS volume_base_f32
        FROM {self._canonical_table()} FINAL
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
        row_count = len(rows)
        if row_count == 0:
            return CanonicalCandleBatch1m(
                open_time_ms=np.empty(0, dtype=np.int64),
                close_time_ms=np.empty(0, dtype=np.int64),
                ohlcv_f32=np.empty((0, 5), dtype=np.float32),
            )
        open_time_ms = np.fromiter(
            (int(row["open_time_ms"]) for row in rows),
            dtype=np.int64,
            count=row_count,
        )
        close_time_ms = np.fromiter(
            (int(row["close_time_ms"]) for row in rows),
            dtype=np.int64,
            count=row_count,
        )
        ohlcv_f32 = np.empty((row_count, 5), dtype=np.float32)
        ohlcv_f32[:, 0] = np.fromiter(
            (float(row["open_f32"]) for row in rows),
            dtype=np.float32,
            count=row_count,
        )
        ohlcv_f32[:, 1] = np.fromiter(
            (float(row["high_f32"]) for row in rows),
            dtype=np.float32,
            count=row_count,
        )
        ohlcv_f32[:, 2] = np.fromiter(
            (float(row["low_f32"]) for row in rows),
            dtype=np.float32,
            count=row_count,
        )
        ohlcv_f32[:, 3] = np.fromiter(
            (float(row["close_f32"]) for row in rows),
            dtype=np.float32,
            count=row_count,
        )
        ohlcv_f32[:, 4] = np.fromiter(
            (float(row["volume_base_f32"]) for row in rows),
            dtype=np.float32,
            count=row_count,
        )
        return CanonicalCandleBatch1m(
            open_time_ms=np.ascontiguousarray(open_time_ms, dtype=np.int64),
            close_time_ms=np.ascontiguousarray(close_time_ms, dtype=np.int64),
            ohlcv_f32=np.ascontiguousarray(ohlcv_f32, dtype=np.float32),
        )

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
