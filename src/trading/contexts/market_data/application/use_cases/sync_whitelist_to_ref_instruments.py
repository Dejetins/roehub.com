from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from trading.contexts.market_data.application.dto.reference_data import (
    InstrumentRefUpsert,
    WhitelistInstrumentRow,
)
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.stores.instrument_ref_writer import (
    InstrumentRefWriter,
)


@dataclass(frozen=True, slots=True)
class SyncWhitelistReport:
    rows_total: int
    rows_upserted: int
    enabled_count: int
    disabled_count: int


class SyncWhitelistToRefInstrumentsUseCase:
    """
    Sync whitelist (including disabled) into market_data.ref_instruments using ReplacingMergeTree(updated_at) upserts.
    """  # noqa: E501

    def __init__(
        self,
        *,
        writer: InstrumentRefWriter,
        clock: Clock,
        known_market_ids: set[int],
    ) -> None:
        if writer is None:  # type: ignore[truthy-bool]
            raise ValueError("SyncWhitelistToRefInstrumentsUseCase requires writer")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("SyncWhitelistToRefInstrumentsUseCase requires clock")
        if not known_market_ids:
            raise ValueError("SyncWhitelistToRefInstrumentsUseCase requires non-empty known_market_ids")  # noqa: E501
        self._writer = writer
        self._clock = clock
        self._known_market_ids = known_market_ids

    def run(self, rows: Iterable[WhitelistInstrumentRow]) -> SyncWhitelistReport:
        rows_list = list(rows)
        if not rows_list:
            return SyncWhitelistReport(rows_total=0, rows_upserted=0, enabled_count=0, disabled_count=0)  # noqa: E501

        # Validate market_ids early (fail fast on wrong whitelist)
        bad = sorted({r.instrument_id.market_id.value for r in rows_list if r.instrument_id.market_id.value not in self._known_market_ids})  # noqa: E501
        if bad:
            raise ValueError(f"whitelist contains unknown market_id(s) not present in runtime config: {bad}")  # noqa: E501

        now = self._clock.now()

        upserts: list[InstrumentRefUpsert] = []
        enabled = 0
        disabled = 0

        for r in rows_list:
            if r.is_enabled:
                enabled += 1
                status = "ENABLED"
                is_tradable = 1
            else:
                disabled += 1
                status = "DISABLED"
                is_tradable = 0

            upserts.append(
                InstrumentRefUpsert(
                    market_id=r.instrument_id.market_id,
                    symbol=r.instrument_id.symbol,
                    status=status,
                    is_tradable=is_tradable,
                    updated_at=now,
                )
            )

        self._writer.upsert(upserts)

        return SyncWhitelistReport(
            rows_total=len(rows_list),
            rows_upserted=len(upserts),
            enabled_count=enabled,
            disabled_count=disabled,
        )
