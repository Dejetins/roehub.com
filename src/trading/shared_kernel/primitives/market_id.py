from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarketId:
    """
    MarketId — стабильный идентификатор рынка (биржа + тип рынка) из ref_market.market_id.

    DDL: UInt16
    Инварианты:
    - > 0
    - <= 65535 (влезает в UInt16)
    """

    value: int

    def __post_init__(self) -> None:
        # Базовая проверка через сравнения (duck-typing).
        # Если value "не число", сравнение выбросит TypeError — мы превращаем в ValueError с понятным текстом.  # noqa: E501
        try:
            if self.value <= 0:
                raise ValueError(f"MarketId must be > 0, got {self.value}")
            if self.value > 65535:
                raise ValueError(f"MarketId must fit UInt16 (<= 65535), got {self.value}")
        except TypeError as e:
            raise ValueError(f"MarketId must be an integer-like value, got {self.value!r}") from e

        # bool — подкласс int, но для ID это обычно ошибка (True/False не хотим).
        if type(self.value) is bool:  # noqa: E721 (нам здесь осознанно нужен строгий type-check)
            raise ValueError("MarketId must be an int, not a bool")

    def __str__(self) -> str:
        # Сериализация "как число" обычно делает слой DTO, но строковое представление полезно для логов.  # noqa: E501
        return str(self.value)
