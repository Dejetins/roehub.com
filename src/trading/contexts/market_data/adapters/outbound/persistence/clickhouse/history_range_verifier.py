"""Repair the derived bitmap and persist verified source absences in bounded ranges."""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseCommandGateway,
)
from trading.contexts.market_data.application.services.source_retry import IncompleteHistoryError
from trading.shared_kernel.primitives import InstrumentId, TimeRange


@dataclass(frozen=True)
class ClickHouseHistoryRangeVerifier:
    gateway: ClickHouseCommandGateway
    database: str = "market_data"

    def confirm(
        self, *, instrument_id: InstrumentId, time_range: TimeRange,
        observed_minutes: set[int], observed_at: datetime, accept_absence: bool,
        check_lease: Callable[[], None],
    ) -> None:
        start = int(time_range.start.value.timestamp() // 60)
        end = int(time_range.end.value.timestamp() // 60)
        if end - start > 7 * 1440:
            raise ValueError("History verification must be bounded to seven days")
        params = dict(market=instrument_id.market_id.value, symbol=str(instrument_id.symbol),
                      start=time_range.start.value, end=time_range.end.value)
        rows = self.gateway.select(
            f"""SELECT DISTINCT toUInt32(intDiv(toUnixTimestamp(ts_open),60)) AS minute
            FROM {self.database}.canonical_candles_1m
            WHERE market_id=%(market)s AND symbol=%(symbol)s
                AND ts_open >= %(start)s AND ts_open < %(end)s
            SETTINGS max_threads=1, max_execution_time=10, max_memory_usage=134217728""",
            params,
        )
        actual = {int(row["minute"]) for row in rows}
        # A returned but unpersisted candle is a storage failure, never an exchange gap.
        if not observed_minutes.issubset(actual):
            raise IncompleteHistoryError()
        check_lease()
        self.gateway.command(
            f"""INSERT INTO {self.database}.canonical_candles_1m_stats
            SELECT market_id, symbol, sumState(toUInt64(0)), minState(ts_open),
                maxState(ts_open), groupBitmapState(toUInt32(intDiv(toUnixTimestamp(ts_open),60)))
            FROM {self.database}.canonical_candles_1m
            WHERE market_id=%(market)s AND symbol=%(symbol)s
                AND ts_open >= %(start)s AND ts_open < %(end)s
                AND NOT bitmapContains((SELECT groupBitmapMergeState(minute_bitmap_state)
                    FROM {self.database}.canonical_candles_1m_stats
                    WHERE market_id=%(market)s AND symbol=%(symbol)s),
                    toUInt32(intDiv(toUnixTimestamp(ts_open),60)))
            GROUP BY market_id, symbol
            SETTINGS max_threads=1, max_execution_time=10, max_memory_usage=134217728""",
            params,
        )
        absent = set(range(start, end)) - actual if accept_absence else set()
        if absent:
            check_lease()
            self.gateway.insert_rows(f"{self.database}.source_absent_candles_1m", [
                dict(market_id=instrument_id.market_id.value, symbol=str(instrument_id.symbol),
                     minute_key=minute, observed_at=observed_at)
                for minute in sorted(absent)
            ])
