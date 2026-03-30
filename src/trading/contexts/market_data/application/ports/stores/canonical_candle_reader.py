from __future__ import annotations

from datetime import timedelta
from typing import Iterator, Protocol

from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    CanonicalCandleBatch1m,
)
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


class CanonicalCandleReader(Protocol):
    """
    CanonicalCandleReader — порт чтения канонических 1m свечей из market_data.canonical_candles_1m.

    Contract:
    - read_1m(instrument_id, time_range) -> Iterator[CandleWithMeta]
    - read_1m_arrays(instrument_id, time_range) -> CanonicalCandleBatch1m

    Semantics:
    - возвращает свечи в пределах полуинтервала [start, end)
    - SHOULD: выдача отсортирована по candle.ts_open по возрастанию

    Dedup rule (контрактный):
    - для части диапазона, пересекающей последние 24 часа относительно clock.now(),
      порт гарантирует:
        - не более одной записи на ключ (instrument_id, candle.ts_open)
        - выбирается "последняя версия" по meta.ingested_at
        - для данных старше 24 часов допускается чтение "как есть" (без дополнительного дедупа в запросе)

    Примечание:
    - clock — зависимость реализации (передаётся через конструктор в адаптере).
      Application-слой не обязан передавать clock в каждый вызов read_1m().
    """  # noqa: E501

    def read_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        ...

    def read_1m_arrays(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> CanonicalCandleBatch1m:
        """
        Return canonical `1m` candles in a columnar batch optimized for precompute workloads.

        Args:
            instrument_id: Requested market/symbol identity.
            time_range: Half-open UTC range `[start, end)`.
        Returns:
            CanonicalCandleBatch1m: Strict columnar candle payload ordered by `ts_open`.
        Assumptions:
            Offline/precompute workloads may bypass row-by-row DTO construction for performance.
        Raises:
            Exception: Propagates storage adapter failures.
        Side Effects:
            Reads canonical storage once for the requested range.
        """
        ...


class DedupCutoff:
    """
    DedupCutoff — маленький помощник application-слоя, чтобы единообразно считать границу хвоста.

    Это не port и не "менеджер": просто неизменяемое правило (24 часа) в одном месте.
    """

    def __init__(self, clock: Clock, tail: timedelta = timedelta(hours=24)) -> None:
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("DedupCutoff requires clock")
        if tail.total_seconds() <= 0:
            raise ValueError("DedupCutoff requires positive tail duration")
        self._clock = clock
        self._tail = tail

    def value(self) -> UtcTimestamp:
        now = self._clock.now().value
        return UtcTimestamp(now - self._tail)
