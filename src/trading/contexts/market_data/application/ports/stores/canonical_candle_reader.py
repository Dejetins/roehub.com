from __future__ import annotations

from typing import Iterator, Protocol

from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    CanonicalCandleBatch1m,
)
from trading.shared_kernel.primitives import InstrumentId, TimeRange


class CanonicalCandleReader(Protocol):
    """
    CanonicalCandleReader — порт чтения канонических 1m свечей из market_data.canonical_candles_1m.

    Contract:
    - read_1m(instrument_id, time_range) -> Iterator[CandleWithMeta]
    - read_1m_arrays(instrument_id, time_range) -> CanonicalCandleBatch1m

    Semantics:
    - возвращает свечи в пределах полуинтервала [start, end)
    - SHOULD: выдача отсортирована по candle.ts_open по возрастанию

    Reader contract:
    - offline/precompute consumers may rely on duplicate-free deterministic reads for the whole
      requested range;
    - concrete storage adapter chooses implementation strategy (`FINAL` or equivalent), without
      exposing dedup mechanics to application consumers.
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
