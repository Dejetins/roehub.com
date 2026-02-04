from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class UtcTimestamp:
    """
    UtcTimestamp — единый тип времени в системе с жёстким требованием UTC.

    Правила (зафиксированы в docs/architecture/shared-kernel-primitives.md):
    - входной datetime должен быть timezone-aware (naive запрещён)
    - храним время в UTC
    - точность приводим к миллисекундам (DateTime64(3, 'UTC'))
    """

    value: datetime

    def __post_init__(self) -> None:
        dt = self.value

        # 1) Запрещаем naive datetime:
        # tzinfo может быть не None, но utcoffset() всё равно может вернуть None.
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("UtcTimestamp requires a timezone-aware datetime (naive datetime is forbidden)")  # noqa: E501

        # 2) Приводим к UTC (разрешаем любой timezone-aware вход, но внутри всегда UTC).
        dt_utc = dt.astimezone(timezone.utc)

        # 3) Приводим точность к миллисекундам:
        # DateTime64(3) хранит миллисекунды, поэтому микросекунды обрезаем вниз.
        ms = (dt_utc.microsecond // 1000) * 1000
        dt_utc_ms = dt_utc.replace(microsecond=ms)

        object.__setattr__(self, "value", dt_utc_ms)

    def __str__(self) -> str:
        """
        Сериализация "как строка" — ISO в UTC с миллисекундами и суффиксом Z.
        Пример: 2026-02-04T12:34:56.789Z
        """
        s = self.value.isoformat(timespec="milliseconds")
        # isoformat() для UTC обычно даёт '+00:00', заменяем на 'Z' для привычного вида.
        if s.endswith("+00:00"):
            s = s[:-6] + "Z"
        return s
