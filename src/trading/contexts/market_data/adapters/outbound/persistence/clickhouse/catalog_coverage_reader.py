from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)


@dataclass(frozen=True, slots=True)
class ClickHouseCatalogCoverageReader:
    gateway: ClickHouseGateway
    database: str = "market_data"

    def counts(
        self, *, market_ids: Sequence[int], start_at: datetime, end_at: datetime
    ) -> Mapping[tuple[int, str], int]:
        if not market_ids or len(market_ids) > 4:
            raise ValueError("catalog coverage requires one to four markets")
        minutes = (end_at - start_at).total_seconds() / 60
        if not 0 < minutes <= 10080:
            raise ValueError("catalog coverage range exceeds seven days")
        rows = self.gateway.select(
            f"""
            SELECT market_id, symbol,
                   uniqExact(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS candles
            FROM {self.database}.canonical_candles_1m
            WHERE market_id IN %(markets)s
              AND ts_open >= fromUnixTimestamp64Milli(%(start)s, 'UTC')
              AND ts_open < fromUnixTimestamp64Milli(%(end)s, 'UTC')
            GROUP BY market_id, symbol
            SETTINGS max_threads = 1, max_execution_time = 10,
                     max_result_rows = 80000, result_overflow_mode = 'throw'
            """,
            {"markets": tuple(market_ids), "start": int(start_at.timestamp() * 1000),
             "end": int(end_at.timestamp() * 1000)},
        )
        return {(int(r["market_id"]), str(r["symbol"])): int(r["candles"]) for r in rows}

    def latest(self, *, market_id: int, symbol: str, before: datetime) -> datetime | None:
        rows = self.gateway.select(
            f"""
            SELECT maxOrNull(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS minute
            FROM {self.database}.canonical_candles_1m
            WHERE market_id = %(market)s AND symbol = %(symbol)s
              AND ts_open < fromUnixTimestamp64Milli(%(before)s, 'UTC')
            SETTINGS max_threads = 1, max_execution_time = 10
            """,
            {"market": market_id, "symbol": symbol, "before": int(before.timestamp() * 1000)},
        )
        minute = rows[0].get("minute") if rows else None
        return datetime.fromtimestamp(int(minute) * 60, UTC) if minute is not None else None
