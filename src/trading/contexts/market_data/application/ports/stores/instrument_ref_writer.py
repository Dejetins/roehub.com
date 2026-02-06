from __future__ import annotations

from typing import Iterable, Protocol

from trading.contexts.market_data.application.dto.reference_data import InstrumentRefUpsert


class InstrumentRefWriter(Protocol):
    def upsert(self, rows: Iterable[InstrumentRefUpsert]) -> None:
        ...
