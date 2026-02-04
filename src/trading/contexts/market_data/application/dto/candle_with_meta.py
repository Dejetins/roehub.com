from __future__ import annotations

from dataclasses import dataclass

from trading.shared_kernel.primitives import Candle, CandleMeta


@dataclass(frozen=True, slots=True)
class CandleWithMeta:
    """
    CandleWithMeta — упаковка канонической свечи Candle и метаданных ingestion CandleMeta.

    Purpose (см. docs/architecture/market_data/market-data-application-ports.md):
    - единая модель данных для ingestion pipeline и для чтения canonical при необходимости.

    Invariants:
    - candle и meta должны быть предоставлены (не None)
    - детальные инварианты проверяются внутри Candle и CandleMeta
    """

    candle: Candle
    meta: CandleMeta

    def __post_init__(self) -> None:
        if self.candle is None:  # type: ignore[truthy-bool]
            raise ValueError("CandleWithMeta requires candle")
        if self.meta is None:  # type: ignore[truthy-bool]
            raise ValueError("CandleWithMeta requires meta")
