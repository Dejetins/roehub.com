from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from importlib import import_module
from typing import Any, Iterator, Mapping, Protocol, Sequence, cast

from trading.contexts.market_data.adapters.outbound.config.instrument_key import (
    build_instrument_key,
)
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
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


class ParquetScanner(Protocol):
    """
    Внутренний контракт для сканирования parquet (implementation detail).

    Нужен для unit-тестов без реального parquet/pyarrow.
    """

    def scan_filtered(
        self,
        *,
        market_id: int,
        symbol: str,
        start_ts_open,
        end_ts_open,
        columns: Sequence[str],
        batch_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        ...


@dataclass(frozen=True, slots=True)
class ParquetColumnMap:
    """
    Column map для parquet ingestion.

    V1 defaults ожидают канонические имена:
    - market_id, symbol, ts_open, ts_close
    - open/high/low/close, volume_base, volume_quote
    - trades_count, taker_buy_volume_base, taker_buy_volume_quote

    Можно переопределить, если в файле другие имена.
    """

    market_id: str = "market_id"
    symbol: str = "symbol"
    ts_open: str = "ts_open"
    ts_close: str = "ts_close"

    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"

    volume_base: str = "volume_base"
    volume_quote: str = "volume_quote"

    trades_count: str = "trades_count"
    taker_buy_volume_base: str = "taker_buy_volume_base"
    taker_buy_volume_quote: str = "taker_buy_volume_quote"


class ParquetCandleIngestSource(CandleIngestSource):
    """
    CandleIngestSource из .parquet.

    Ключевые решения:
    - parquet ОБЯЗАН содержать market_id и symbol
    - instrument_key генерируется здесь через runtime config:
      "{exchange}:{market_type}:{symbol}" (parquet его не обязан иметь)
    - meta.source = "file"
    - meta.ingested_at = clock.now() (один на весь stream_1m вызов)
    """

    def __init__(
        self,
        scanner: ParquetScanner,
        cfg: MarketDataRuntimeConfig,
        clock: Clock,
        column_map: ParquetColumnMap | None = None,
        batch_size: int = 50_000,
    ) -> None:
        """
        Initialize parquet ingestion source and its dependencies.

        Parameters:
        - scanner: parquet scanner implementation yielding filtered rows.
        - cfg: runtime config for market_id -> exchange/market_type resolution.
        - clock: clock used to stamp `meta.ingested_at`.
        - column_map: optional parquet column aliases.
        - batch_size: scanner read batch size.

        Returns:
        - None.

        Assumptions/Invariants:
        - scanner/cfg/clock are provided and valid.
        - batch_size is positive.

        Errors/Exceptions:
        - Raises `ValueError` if a required dependency is missing or batch_size is invalid.

        Side effects:
        - None.
        """
        if scanner is None:  # type: ignore[truthy-bool]
            raise ValueError("ParquetCandleIngestSource requires scanner")
        if cfg is None:  # type: ignore[truthy-bool]
            raise ValueError("ParquetCandleIngestSource requires cfg")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("ParquetCandleIngestSource requires clock")
        if batch_size <= 0:
            raise ValueError("ParquetCandleIngestSource requires batch_size > 0")

        self._scanner = scanner
        self._cfg = cfg
        self._clock = clock
        self._cols = column_map or ParquetColumnMap()
        self._batch_size = batch_size

    def stream_1m(self, instrument_id: InstrumentId, time_range: TimeRange) -> Iterator[CandleWithMeta]:  # noqa: E501
        """
        Stream parquet candles for a single instrument and UTC half-open range.

        Parameters:
        - instrument_id: instrument identity used for scanner filtering.
        - time_range: requested half-open interval `[start, end)`.

        Returns:
        - Iterator of mapped `CandleWithMeta` rows.

        Assumptions/Invariants:
        - scanner filter semantics preserve `[start, end)` boundaries.
        - instrument_key is canonical and stable for the requested instrument.

        Errors/Exceptions:
        - Propagates mapping/validation errors from `_map_row(...)`.

        Side effects:
        - Reads parquet rows through the injected scanner.
        """
        ingested_at = self._clock.now()
        instrument_key = build_instrument_key(cfg=self._cfg, instrument_id=instrument_id)

        market_id = int(instrument_id.market_id.value)
        symbol = str(instrument_id.symbol)

        required_cols = self._required_columns()
        for raw in self._scanner.scan_filtered(
            market_id=market_id,
            symbol=symbol,
            start_ts_open=time_range.start.value,
            end_ts_open=time_range.end.value,
            columns=required_cols,
            batch_size=self._batch_size,
        ):
            yield self._map_row(raw=raw, ingested_at=ingested_at, instrument_key=instrument_key)

    def _required_columns(self) -> Sequence[str]:
        # В V1 требуем market_id+symbol обязательно, остальное как в доке.
        c = self._cols
        return [
            c.market_id,
            c.symbol,
            c.ts_open,
            c.ts_close,
            c.open,
            c.high,
            c.low,
            c.close,
            c.volume_base,
            c.volume_quote,
            c.trades_count,
            c.taker_buy_volume_base,
            c.taker_buy_volume_quote,
        ]

    def _map_row(
        self,
        *,
        raw: Mapping[str, Any],
        ingested_at: UtcTimestamp,
        instrument_key: str,
    ) -> CandleWithMeta:
        """
        Map one parquet row to `CandleWithMeta`.

        Parameters:
        - raw: parquet row values by column name.
        - ingested_at: ingestion timestamp used in metadata.
        - instrument_key: canonical key `exchange:market_type:symbol`.

        Returns:
        - Domain row with normalized candle fields and ingestion metadata.

        Assumptions/Invariants:
        - `raw` includes required identifier columns (`market_id`, `symbol`).
        - numeric fields are convertible to expected primitive types.

        Errors/Exceptions:
        - Raises `ValueError` if identifier columns are missing.
        - Propagates primitive validation and type conversion errors.

        Side effects:
        - None.
        """
        c = self._cols

        if c.market_id not in raw or c.symbol not in raw:
            raise ValueError("Parquet row must contain market_id and symbol (required)")

        m_id = MarketId(int(raw[c.market_id]))
        sym = Symbol(str(raw[c.symbol]))
        instrument = InstrumentId(m_id, sym)

        ts_open = UtcTimestamp(_ensure_tz_utc(raw[c.ts_open]))
        ts_close_val = raw.get(c.ts_close)
        if ts_close_val is None:
            ts_close = UtcTimestamp(ts_open.value + timedelta(minutes=1))
        else:
            ts_close = UtcTimestamp(_ensure_tz_utc(ts_close_val))

        candle = Candle(
            instrument_id=instrument,
            ts_open=ts_open,
            ts_close=ts_close,
            open=float(raw[c.open]),
            high=float(raw[c.high]),
            low=float(raw[c.low]),
            close=float(raw[c.close]),
            volume_base=float(raw[c.volume_base]),
            volume_quote=(float(raw[c.volume_quote]) if raw.get(c.volume_quote) is not None else None),  # noqa: E501
        )

        meta = CandleMeta(
            source="file",
            ingested_at=ingested_at,
            ingest_id=None,
            instrument_key=instrument_key,
            trades_count=(int(raw[c.trades_count]) if raw.get(c.trades_count) is not None else None),  # noqa: E501
            taker_buy_volume_base=(
                float(raw[c.taker_buy_volume_base]) if raw.get(c.taker_buy_volume_base) is not None else None  # noqa: E501
            ),
            taker_buy_volume_quote=(
                float(raw[c.taker_buy_volume_quote]) if raw.get(c.taker_buy_volume_quote) is not None else None  # noqa: E501
            ),
        )

        return CandleWithMeta(candle=candle, meta=meta)


class PyArrowParquetScanner:
    """
    Реальный scanner на pyarrow.dataset.

    Примечание:
    - этот класс НЕ является port'ом, это implementation detail parquet адаптера.
    """

    def __init__(self, paths: Sequence[str]) -> None:
        if not paths:
            raise ValueError("PyArrowParquetScanner requires at least one path")
        self._paths = list(paths)

    def scan_filtered(
        self,
        *,
        market_id: int,
        symbol: str,
        start_ts_open,
        end_ts_open,
        columns: Sequence[str],
        batch_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        try:
            ds = cast(Any, import_module("pyarrow.dataset"))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("pyarrow is required for PyArrowParquetScanner") from e

        dataset = ds.dataset(self._paths, format="parquet")

        # фильтры pushdown
        f = (ds.field("market_id") == market_id) & (ds.field("symbol") == symbol) & (
            (ds.field("ts_open") >= start_ts_open) & (ds.field("ts_open") < end_ts_open)
        )

        scanner = dataset.scanner(columns=list(dict.fromkeys(columns)), filter=f, batch_size=batch_size)  # noqa: E501

        for batch in scanner.to_batches():
            table = batch.to_pydict()
            # row-wise yield
            keys = list(table.keys())
            n = len(table[keys[0]]) if keys else 0
            for i in range(n):
                yield {k: table[k][i] for k in keys}


def _ensure_tz_utc(dt) -> Any:
    if getattr(dt, "tzinfo", None) is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
