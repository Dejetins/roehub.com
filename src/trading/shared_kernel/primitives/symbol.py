from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Symbol:
    """
    Symbol — обозначение инструмента внутри конкретного MarketId (например "BTCUSDT").

    Правила (зафиксированы в docs/architecture/shared-kernel-primitives.md):
    - нормализация: strip + upper
    - инвариант: после нормализации строка не пустая
    """

    value: str

    def __post_init__(self) -> None:
        # Приводим к каноническому виду: без пробелов по краям и в верхнем регистре.
        normalized = self.value.strip().upper()
        object.__setattr__(self, "value", normalized)

        # Инвариант: символ не может быть пустым после нормализации.
        if not normalized:
            raise ValueError("Symbol must be non-empty after normalization")

    def __str__(self) -> str:
        # Сериализация "как строка" — просто вернуть нормализованное значение.
        return self.value
