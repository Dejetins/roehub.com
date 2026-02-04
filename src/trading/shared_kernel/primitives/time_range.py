from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .utc_timestamp import UtcTimestamp


@dataclass(frozen=True, slots=True)
class TimeRange:
    """
    TimeRange — стандартный диапазон времени.

    Семантика:
    - полуинтервал [start, end): start включён, end не включён.

    Инварианты:
    - start < end
    """

    start: UtcTimestamp
    end: UtcTimestamp

    def __post_init__(self) -> None:
        if self.start.value >= self.end.value:
            raise ValueError(
                f"TimeRange requires start < end, got start={self.start} end={self.end}"
            )

    def duration(self) -> timedelta:
        """Длительность диапазона как timedelta (end - start)."""
        return self.end.value - self.start.value

    def contains(self, ts: UtcTimestamp) -> bool:
        """Проверка попадания точки во временной диапазон с семантикой [start, end)."""
        return self.start.value <= ts.value < self.end.value

    def overlap(self, other: TimeRange) -> bool:
        """Есть ли пересечение двух полуинтервалов [start, end)."""
        return self.start.value < other.end.value and other.start.value < self.end.value

    def intersection(self, other: TimeRange) -> TimeRange:
        """
        Пересечение двух диапазонов.
        Если пересечения нет — бросаем ValueError.
        """
        new_start = self.start if self.start.value >= other.start.value else other.start
        new_end = self.end if self.end.value <= other.end.value else other.end
        return TimeRange(new_start, new_end)
