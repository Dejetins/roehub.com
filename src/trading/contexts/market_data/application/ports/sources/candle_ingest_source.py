from __future__ import annotations

from typing import Iterator, Protocol

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.shared_kernel.primitives import InstrumentId, TimeRange


class CandleIngestSource(Protocol):
    """
    CandleIngestSource — порт получения 1m свечей из внешних источников (ws/rest/file).

    Contract:
    - stream_1m(instrument_id, time_range) -> Iterator[CandleWithMeta]

    Semantics:
    - возвращает свечи в пределах полуинтервала [start, end)
    - SHOULD: выдача отсортирована по candle.ts_open по возрастанию
    - meta.source должен быть одним из: ws | rest | file
    - meta.instrument_key — trace/debug поле, формируется в адаптере
    """

    def stream_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        ...
