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
from trading.shared_kernel.primitives.market_id import MarketId
from trading.shared_kernel.primitives.symbol import Symbol


@dataclass(frozen=True, slots=True)
class SyncWhitelistReport:
    rows_total: int
    rows_upserted: int
    rows_skipped_unchanged: int
    enabled_count: int
    disabled_count: int


class SyncWhitelistToRefInstrumentsUseCase:
    """
    Sync whitelist -> ref_instruments, inserting ONLY new or changed rows.
    Re-running with the same whitelist is idempotent (0 inserts).
    """

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
            return SyncWhitelistReport(
                rows_total=0,
                rows_upserted=0,
                rows_skipped_unchanged=0,
                enabled_count=0,
                disabled_count=0,
            )

        bad = sorted(
            {
                r.instrument_id.market_id.value
                for r in rows_list
                if r.instrument_id.market_id.value not in self._known_market_ids
            }
        )
        if bad:
            raise ValueError(f"whitelist contains unknown market_id(s) not present in runtime config: {bad}")  # noqa: E501

        now = self._clock.now()

        # group by market_id for efficient CH queries
        by_market: dict[int, list[WhitelistInstrumentRow]] = {}
        for r in rows_list:
            by_market.setdefault(r.instrument_id.market_id.value, []).append(r)

        upserts: list[InstrumentRefUpsert] = []
        enabled = 0
        disabled = 0
        skipped = 0

        for mid_int, group in by_market.items():
            market_id = MarketId(mid_int)
            symbols: list[Symbol] = [g.instrument_id.symbol for g in group]
            existing = self._writer.existing_latest(market_id=market_id, symbols=symbols)

            for r in group:
                sym = str(r.instrument_id.symbol)
                if r.is_enabled:
                    enabled += 1
                    target_status = "ENABLED"
                    target_tradable = 1
                else:
                    disabled += 1
                    target_status = "DISABLED"
                    target_tradable = 0

                cur = existing.get(sym)
                if cur is not None:
                    cur_status, cur_tradable = cur
                    if cur_status == target_status and cur_tradable == target_tradable:
                        skipped += 1
                        continue

                upserts.append(
                    InstrumentRefUpsert(
                        market_id=r.instrument_id.market_id,
                        symbol=r.instrument_id.symbol,
                        status=target_status,
                        is_tradable=target_tradable,
                        updated_at=now,
                    )
                )

        self._writer.upsert(upserts)

        return SyncWhitelistReport(
            rows_total=len(rows_list),
            rows_upserted=len(upserts),
            rows_skipped_unchanged=skipped,
            enabled_count=enabled,
            disabled_count=disabled,
        )
