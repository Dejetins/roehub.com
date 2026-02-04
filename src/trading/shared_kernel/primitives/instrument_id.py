from __future__ import annotations

from dataclasses import dataclass

from .market_id import MarketId
from .symbol import Symbol


@dataclass(frozen=True, slots=True)
class InstrumentId:
    """
    InstrumentId — доменная идентичность инструмента: (market_id, symbol).

    Важно:
    - instrument_key (строка "{exchange}:{market_type}:{symbol}") — НЕ доменный ID.
      Он живёт в CandleMeta как trace/debug.
    """

    market_id: MarketId
    symbol: Symbol

    def __post_init__(self) -> None:
        # Здесь достаточно проверить "присутствие".
        # Инварианты самих полей проверяются в их классах (MarketId/Symbol).
        if self.market_id is None:  # type: ignore[truthy-bool]
            raise ValueError("InstrumentId requires market_id")
        if self.symbol is None:  # type: ignore[truthy-bool]
            raise ValueError("InstrumentId requires symbol")

    def as_dict(self) -> dict:
        # Каноничная сериализация, зафиксированная в документе.
        return {"market_id": self.market_id.value, "symbol": str(self.symbol)}

    def __str__(self) -> str:
        # Удобное строковое представление для логов (не каноничная сериализация).
        return f"{self.market_id.value}:{self.symbol}"
