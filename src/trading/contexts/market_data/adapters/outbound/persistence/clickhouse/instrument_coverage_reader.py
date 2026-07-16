from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.shared_kernel.primitives import InstrumentId


@dataclass(frozen=True, slots=True)
class InstrumentCoverageSnapshot:
    state: str
    percent: float | None


@dataclass(frozen=True, slots=True)
class ClickHouseInstrumentCoverageReader:
    """Read a bounded exact 1m coverage ratio for one confirmed history window."""

    gateway: ClickHouseGateway
    database: str = "market_data"

    def __post_init__(self) -> None:
        if self.gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseInstrumentCoverageReader requires gateway")
        if not self.database.strip():
            raise ValueError("ClickHouseInstrumentCoverageReader requires non-empty database")

    def read(
        self,
        *,
        instrument_id: InstrumentId,
        expected_start_at: datetime,
        expected_end_at: datetime,
    ) -> InstrumentCoverageSnapshot:
        expected_minutes = int(
            (expected_end_at - expected_start_at).total_seconds() // 60
        )
        if expected_minutes <= 0:
            return InstrumentCoverageSnapshot(state="unknown", percent=None)
        rows = self.gateway.select(
            f"""
                SELECT uniqExact(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS candles
                FROM {self.database}.canonical_candles_1m
                WHERE market_id = %(market_id)s
                  AND symbol = %(symbol)s
                  AND ts_open >= fromUnixTimestamp64Milli(%(start_ms)s, 'UTC')
                  AND ts_open < fromUnixTimestamp64Milli(%(end_ms)s, 'UTC')
                SETTINGS max_threads = 1
            """,
            {
                "market_id": instrument_id.market_id.value,
                "symbol": str(instrument_id.symbol),
                "start_ms": _epoch_ms(expected_start_at),
                "end_ms": _epoch_ms(expected_end_at),
            },
        )
        actual_minutes = int(rows[0].get("candles", 0)) if rows else 0
        percent = min(100.0, round((actual_minutes / expected_minutes) * 100, 2))
        return InstrumentCoverageSnapshot(
            state="complete" if actual_minutes >= expected_minutes else "partial",
            percent=percent,
        )


def _epoch_ms(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("coverage timestamps must be timezone-aware")
    return int(value.timestamp() * 1000)
