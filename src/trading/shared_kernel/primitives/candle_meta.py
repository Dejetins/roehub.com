from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from .utc_timestamp import UtcTimestamp

_ALLOWED_SOURCES = frozenset({"ws", "rest", "file"})


@dataclass(frozen=True, slots=True)
class CandleMeta:
    """
    CandleMeta — метаданные происхождения и записи свечи (ingestion metadata).

    Важно:
    - Это отдельный объект от Candle (рыночные поля не смешиваем с ingestion).
    - Иммутабельный: после создания не меняется.
    """

    source: str
    ingested_at: UtcTimestamp
    ingest_id: Optional[UUID]
    instrument_key: str

    trades_count: Optional[int]
    taker_buy_volume_base: Optional[float]
    taker_buy_volume_quote: Optional[float]

    def __post_init__(self) -> None:
        # Нормализуем "source" в канонический вид
        normalized_source = self.source.strip().lower()
        object.__setattr__(self, "source", normalized_source)

        if normalized_source not in _ALLOWED_SOURCES:
            raise ValueError(
                f"Invalid source={self.source!r}. Allowed: {sorted(_ALLOWED_SOURCES)}"
            )

        # ingested_at: предполагаем, что UtcTimestamp сам гарантирует UTC + ms.
        # Здесь фиксируем только то, что значение присутствует.
        if self.ingested_at is None:  # type: ignore[truthy-bool]
            raise ValueError("ingested_at must be provided")

        key = self.instrument_key.strip()
        object.__setattr__(self, "instrument_key", key)
        if not key:
            raise ValueError("instrument_key must be non-empty")

        # Nullable(UInt32) в DDL → int | None в домене, но без отрицательных
        self._require_non_negative_int(self.trades_count, "trades_count")

        # Nullable(Float64) → float | None, тоже без отрицательных
        self._require_non_negative_float(self.taker_buy_volume_base, "taker_buy_volume_base")
        self._require_non_negative_float(self.taker_buy_volume_quote, "taker_buy_volume_quote")

    def as_dict(self) -> dict:
        """
        Сериализация в словарь (удобно для DTO/логов).

        Мы не лезем внутрь UtcTimestamp: предполагаем, что у него нормальный __str__()
        (например ISO-строка в UTC). Если ты решишь сериализовать по-другому —
        меняется только UtcTimestamp или слой DTO.
        """
        return {
            "source": self.source,
            "ingested_at": str(self.ingested_at),
            "ingest_id": str(self.ingest_id) if self.ingest_id is not None else None,
            "instrument_key": self.instrument_key,
            "trades_count": self.trades_count,
            "taker_buy_volume_base": self.taker_buy_volume_base,
            "taker_buy_volume_quote": self.taker_buy_volume_quote,
        }

    # Ниже — маленькие приватные методы: они не "геттеры/сеттеры",
    # а часть инвариантов объекта (EO-style).

    def _require_non_negative_int(self, value: Optional[int], field: str) -> None:
        if value is None:
            return
        if value < 0:
            raise ValueError(f"{field} must be >= 0, got {value}")

    def _require_non_negative_float(self, value: Optional[float], field: str) -> None:
        if value is None:
            return
        if value < 0.0:
            raise ValueError(f"{field} must be >= 0, got {value}")
