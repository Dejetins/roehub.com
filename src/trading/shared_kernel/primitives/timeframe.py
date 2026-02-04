from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .utc_timestamp import UtcTimestamp


# Базовый таймфрейм-источник правды (в БД хранится только 1m canonical).
_BASE_CODE = "1m"

# Поддерживаемый набор (можно расширять).
# Значения — длительность в секундах.
_SUPPORTED_SECONDS = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}


@dataclass(frozen=True, slots=True)
class Timeframe:
    """
    Timeframe — параметр расчёта/запроса.
    Хранение: источник правды только 1m, остальные TF derived через rollup.

    Representation:
    - code: "1m", "5m", ...
    """

    code: str

    def __post_init__(self) -> None:
        normalized = self.code.strip().lower()
        object.__setattr__(self, "code", normalized)

        if normalized not in _SUPPORTED_SECONDS:
            raise ValueError(
                f"Unsupported timeframe={normalized!r}. Supported: {sorted(_SUPPORTED_SECONDS.keys())}"  # noqa: E501
            )

        # Любой derived timeframe должен быть кратен 1 минуте.
        seconds = _SUPPORTED_SECONDS[normalized]
        if seconds % _SUPPORTED_SECONDS[_BASE_CODE] != 0:
            raise ValueError(f"Timeframe {normalized!r} must be a multiple of 1m")

    def duration(self) -> timedelta:
        """Длительность таймфрейма как timedelta (это derived-значение)."""
        return timedelta(seconds=_SUPPORTED_SECONDS[self.code])

    def base(self) -> Timeframe:
        """Вернуть базовый TF (1m)."""
        return Timeframe(_BASE_CODE)

    def bucket_open(self, ts: UtcTimestamp) -> UtcTimestamp:
        """
        Вычислить начало бакета (bucket_open) для данного timestamp по правилу epoch-aligned UTC.

        Это не "роллап", а только детерминированное выравнивание времени по TF.
        """
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        dt = ts.value  # уже UTC + ms по инвариантам UtcTimestamp

        delta = dt - epoch
        total_ms = delta // timedelta(milliseconds=1)

        tf_ms = self.duration() // timedelta(milliseconds=1)
        bucket_ms = (total_ms // tf_ms) * tf_ms

        return UtcTimestamp(epoch + timedelta(milliseconds=bucket_ms))

    def bucket_close(self, ts: UtcTimestamp) -> UtcTimestamp:
        """bucket_close = bucket_open + duration."""
        return UtcTimestamp(self.bucket_open(ts).value + self.duration())

    def __str__(self) -> str:
        # Сериализация "как строка"
        return self.code
