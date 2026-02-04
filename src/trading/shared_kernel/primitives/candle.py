from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .instrument_id import InstrumentId
from .utc_timestamp import UtcTimestamp


@dataclass(frozen=True, slots=True)
class Candle:
    """
    Candle — каноническая рыночная свеча (без ingestion-метаданных).
    Ingestion-мета живёт отдельно в CandleMeta.

    Поля соответствуют canonical_candles_1m (market fields):
    - (market_id, symbol) => instrument_id
    - ts_open/ts_close
    - OHLC
    - volume_base, volume_quote
    """

    instrument_id: InstrumentId

    ts_open: UtcTimestamp
    ts_close: UtcTimestamp

    open: float
    high: float
    low: float
    close: float

    volume_base: float
    volume_quote: Optional[float]

    def __post_init__(self) -> None:
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("Candle requires instrument_id")

        # Время
        if self.ts_open.value >= self.ts_close.value:
            raise ValueError(f"Candle requires ts_open < ts_close, got {self.ts_open} .. {self.ts_close}")  # noqa: E501

        # OHLC инварианты
        if self.high < max(self.open, self.close):
            raise ValueError("Candle requires high >= max(open, close)")

        if self.low > min(self.open, self.close):
            raise ValueError("Candle requires low <= min(open, close)")

        # Объёмы: неотрицательные
        if self.volume_base < 0:
            raise ValueError("Candle requires volume_base >= 0")

        if self.volume_quote is not None and self.volume_quote < 0:
            raise ValueError("Candle requires volume_quote >= 0 when provided")

    def as_dict(self) -> dict:
        """
        Сериализация свечи как объекта.
        (instrument_id сериализуем как dict, время как str(UtcTimestamp), числа как float)
        """
        return {
            "instrument_id": self.instrument_id.as_dict(),
            "ts_open": str(self.ts_open),
            "ts_close": str(self.ts_close),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume_base": self.volume_base,
            "volume_quote": self.volume_quote,
        }
