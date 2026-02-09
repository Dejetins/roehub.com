from __future__ import annotations

from datetime import timezone
from typing import Any, Iterable, Mapping

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter


class ClickHouseRawKlineWriter(RawKlineWriter):
    """
    RawKlineWriter реализация для ClickHouse.

    Пишем только в raw_* таблицы:
    - market_id in {1,2} -> raw_binance_klines_1m
    - market_id in {3,4} -> raw_bybit_klines_1m

    instrument_key НЕ требуем от parquet: он должен быть уже в CandleMeta (генерится в источнике).
    """

    def __init__(self, gateway: ClickHouseGateway, database: str = "market_data") -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseRawKlineWriter requires gateway")
        if not database.strip():
            raise ValueError("ClickHouseRawKlineWriter requires non-empty database")
        self._gw = gateway
        self._db = database.strip()

    def write_1m(self, rows: Iterable[CandleWithMeta]) -> None:
        """
        Write 1m candles into proper raw tables grouped by market route.

        Parameters:
        - rows: iterable of normalized candle-with-meta rows.

        Returns:
        - None.

        Assumptions/Invariants:
        - Rows can belong to multiple markets within one call.
        - Routing is determined per row by `market_id`.

        Errors/Exceptions:
        - Raises `ValueError` for unsupported market ids.
        - Propagates gateway insert errors.

        Side effects:
        - Executes one or more ClickHouse inserts into `raw_*_klines_1m`.
        """
        materialized = list(rows)
        if not materialized:
            return

        payload_by_table = self._payload_by_table(materialized)
        for table, payload in payload_by_table.items():
            fq_table = f"{self._db}.{table}"
            self._gw.insert_rows(fq_table, payload)

    def _raw_table_for_market_id(self, market_id: int) -> str:
        """
        Map market id to target raw table name.

        Parameters:
        - market_id: integer market id.

        Returns:
        - Raw table name without database prefix.

        Assumptions/Invariants:
        - Market ids `1,2` are Binance; `3,4` are Bybit.

        Errors/Exceptions:
        - Raises `ValueError` for unsupported market ids.

        Side effects:
        - None.
        """
        if market_id in (1, 2):
            return "raw_binance_klines_1m"
        if market_id in (3, 4):
            return "raw_bybit_klines_1m"
        raise ValueError(f"Unsupported market_id={market_id} for raw writer routing")

    def _payload_by_table(
        self,
        rows: list[CandleWithMeta],
    ) -> dict[str, list[Mapping[str, Any]]]:
        """
        Build per-table payload groups for one mixed market batch.

        Parameters:
        - rows: materialized rows from caller.

        Returns:
        - Mapping `table_name -> payload rows`.

        Assumptions/Invariants:
        - Every row has a supported `market_id`.

        Errors/Exceptions:
        - Raises `ValueError` for unsupported market ids.

        Side effects:
        - None.
        """
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            market_id = int(row.candle.instrument_id.market_id.value)
            table = self._raw_table_for_market_id(market_id)
            grouped.setdefault(table, []).append(self._row_to_raw_payload(row))
        return grouped

    def _row_to_raw_payload(self, row: CandleWithMeta) -> Mapping[str, Any]:
        """
        Convert domain row into raw-table-specific payload mapping.

        Parameters:
        - row: domain row to serialize.

        Returns:
        - Payload dictionary matching target raw table schema.

        Assumptions/Invariants:
        - Market id determines payload schema family (Binance or Bybit).

        Errors/Exceptions:
        - Raises `ValueError` for unsupported market ids.

        Side effects:
        - None.
        """
        m_id = int(row.candle.instrument_id.market_id.value)
        if m_id in (1, 2):
            return self._to_binance_raw(row)
        if m_id in (3, 4):
            return self._to_bybit_raw(row)
        raise ValueError(f"Unsupported market_id={m_id} for raw payload mapping")

    def _to_binance_raw(self, row: CandleWithMeta) -> Mapping[str, Any]:
        """
        Map one candle row to Binance raw table schema.

        Parameters:
        - row: candle row for Binance market.

        Returns:
        - Payload dictionary for `raw_binance_klines_1m`.

        Assumptions/Invariants:
        - Non-nullable numeric fields are normalized to 0/0.0 when source value is absent.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        c = row.candle
        m = row.meta

        symbol = str(c.instrument_id.symbol)
        open_time = _ensure_tz_utc(c.ts_open.value)
        close_time = _ensure_tz_utc(c.ts_close.value)

        # Binance raw DDL: non-nullable fields -> normalize None -> 0/0.0
        quote_asset_volume = float(c.volume_quote) if c.volume_quote is not None else 0.0
        number_of_trades = int(m.trades_count) if m.trades_count is not None else 0
        taker_buy_base = float(m.taker_buy_volume_base) if m.taker_buy_volume_base is not None else 0.0  # noqa: E501
        taker_buy_quote = float(m.taker_buy_volume_quote) if m.taker_buy_volume_quote is not None else 0.0  # noqa: E501

        return {
            "market_id": int(c.instrument_id.market_id.value),
            "symbol": symbol,
            "instrument_key": m.instrument_key,
            "open_time": open_time,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume_base),
            "close_time": close_time,
            "quote_asset_volume": quote_asset_volume,
            "number_of_trades": number_of_trades,
            "taker_buy_base_asset_volume": taker_buy_base,
            "taker_buy_quote_asset_volume": taker_buy_quote,
            "source": m.source,
            "ingested_at": _ensure_tz_utc(m.ingested_at.value),
            "ingest_id": m.ingest_id,
        }

    def _to_bybit_raw(self, row: CandleWithMeta) -> Mapping[str, Any]:
        """
        Map one candle row to Bybit raw table schema.

        Parameters:
        - row: candle row for Bybit market.

        Returns:
        - Payload dictionary for `raw_bybit_klines_1m`.

        Assumptions/Invariants:
        - `interval_min` is always integer `1`.
        - Non-nullable turnover is normalized to `0.0` when absent.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        c = row.candle
        m = row.meta

        symbol = str(c.instrument_id.symbol)
        ts_open = _ensure_tz_utc(c.ts_open.value)

        # Bybit raw DDL: turnover non-nullable -> normalize None -> 0.0
        turnover = float(c.volume_quote) if c.volume_quote is not None else 0.0

        start_time_ms = int(ts_open.timestamp() * 1000)

        return {
            "market_id": int(c.instrument_id.market_id.value),
            "symbol": symbol,
            "instrument_key": m.instrument_key,
            "interval_min": 1,
            "start_time_ms": start_time_ms,
            "start_time_utc": ts_open,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume_base),
            "turnover": turnover,
            "source": m.source,
            "ingested_at": _ensure_tz_utc(m.ingested_at.value),
            "ingest_id": m.ingest_id,
        }


def _ensure_tz_utc(dt) -> Any:
    # Для драйверов CH/parquet встречаются naive datetime. В адаптерах делаем их tz-aware UTC.
    if getattr(dt, "tzinfo", None) is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
