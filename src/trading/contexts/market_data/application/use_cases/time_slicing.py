from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading.shared_kernel.primitives.time_range import TimeRange
from trading.shared_kernel.primitives.utc_timestamp import UtcTimestamp


def slice_time_range_by_utc_days(time_range: TimeRange, *, max_days: int = 7) -> list[TimeRange]:
    """
    Разбивает полуинтервал времени `[start, end)` на последовательность чанков, выровненных по границам UTC-дней.

    Зачем:
    - ClickHouse raw-таблицы партиционированы по дню (toYYYYMMDD по времени открытия свечи).
    - Чтобы один INSERT не затрагивал слишком много дневных партиций (и не ловить операционные лимиты/ошибки),
      диапазон backfill режется на куски, где каждый кусок пересекает не более `max_days` UTC-дней.

    Контракт:
    - Вход: `time_range` со семантикой полуинтервала `[start, end)`, оба времени — UTC (через `UtcTimestamp`).
    - Выход: список `TimeRange`, покрывающих исходный диапазон без дыр и перекрытий.
      * Первый чанк начинается ровно в `time_range.start`.
      * Последний чанк заканчивается ровно в `time_range.end`.
      * Внутренние границы чанков совпадают с границами суток UTC (00:00:00.000Z).

    Инварианты:
    - `1 <= max_days <= 7` (для v2 используем 7).
    - Результат всегда отсортирован по времени (по возрастанию).
    - Каждый чанк гарантированно не пересекает больше `max_days` дневных партиций.

    Пример:
    - `[2026-02-01T12:00Z, 2026-02-11T12:00Z)` при `max_days=7` →
      `[2026-02-01T12:00Z, 2026-02-08T00:00Z)` и `[2026-02-08T00:00Z, 2026-02-11T12:00Z)`.

    Примечание:
    - Резка делается по UTC-дням независимо от исходной минутной/секундной компоненты `start`.
    """ # noqa: E501
    if max_days <= 0 or max_days > 7:
        raise ValueError(f"max_days must be in [1, 7], got {max_days}")

    end_dt = time_range.end.value

    out: list[TimeRange] = []
    cursor = time_range.start

    while cursor.value < end_dt:
        anchor_dt: datetime = _floor_to_utc_day(cursor)
        boundary_dt: datetime = anchor_dt + timedelta(days=max_days)
        slice_end_dt: datetime = end_dt if end_dt <= boundary_dt else boundary_dt

        chunk = TimeRange(
            start=cursor,
            end=UtcTimestamp(slice_end_dt),
        )
        out.append(chunk)
        cursor = chunk.end

    return out


def _floor_to_utc_day(ts: UtcTimestamp) -> datetime:
    dt_utc = ts.value.astimezone(timezone.utc)
    return dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)
