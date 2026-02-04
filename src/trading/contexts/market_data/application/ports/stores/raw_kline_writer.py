from __future__ import annotations

from typing import Iterable, Protocol

from trading.contexts.market_data.application.dto import CandleWithMeta


class RawKlineWriter(Protocol):
    """
    RawKlineWriter — порт записи 1m свечей в ClickHouse raw-таблицы (raw_*_klines_1m).

    Важно:
    - canonical формируется автоматически через MV (raw -> canonical)
    - прямых записей в canonical нет

    Contract:
    - write_1m(rows: Iterable[CandleWithMeta]) -> None

    Semantics:
    - адаптер маршрутизирует запись в нужную raw-таблицу (implementation detail)
    - повторные вставки возможны (re-ingestion/backfill); допустимы дубликаты до merge
    """

    def write_1m(self, rows: Iterable[CandleWithMeta]) -> None:
        ...
