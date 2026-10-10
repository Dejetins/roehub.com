"""Bounded keyset observations from the rebuildable actual-minute bitmap index.

No canonical-scan fallback: unavailable index queries fail visibly; missing states
require the existing bounded history verifier. Administrative deletes require rebuild.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from typing import Any, Mapping

from trading.contexts.backtest_artifacts.application.services.base_source_reconciliation import (
    StoredSourceObservation,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp


@dataclass(frozen=True)
class ClickHouseStoredArtifactCoordinates:
    gateway: ClickHouseGateway
    database: str = "market_data"

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.database):
            raise ValueError("invalid_database")

    def page(self, *, after: tuple[int, str] | None, limit: int = 50):
        """Compatibility coordinate-only view; normal reconciliation uses observation_page."""
        self._validate_limit(limit)
        market, symbol = after or (0, "")
        rows = self.gateway.select(
            f"""SELECT market_id,symbol FROM {self.database}.canonical_candles_1m_stats
            WHERE (market_id,symbol) > (%(market)s,%(symbol)s)
            AND market_id IN %(markets)s
            GROUP BY market_id,symbol ORDER BY market_id,symbol LIMIT %(limit)s
            SETTINGS max_threads=1, max_execution_time=10, max_memory_usage=134217728""",
            dict(market=market, symbol=symbol, limit=limit,
                 markets=tuple(SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2)),
        )
        return tuple(self._instrument(row) for row in rows)

    def observation_page(
        self, *, after: tuple[int, str] | None, before: UtcTimestamp, limit: int = 50,
    ) -> tuple[StoredSourceObservation, ...]:
        self._validate_limit(limit)
        market, symbol = after or (0, "")
        return self._read(
            predicate="(market_id,symbol) > (%(market)s,%(symbol)s)",
            parameters=dict(market=market, symbol=symbol), before=before, limit=limit,
        )

    def observation(
        self, *, instrument: InstrumentId, before: UtcTimestamp,
    ) -> StoredSourceObservation:
        rows = self._read(
            predicate="market_id=%(market)s AND symbol=%(symbol)s",
            parameters=dict(market=instrument.market_id.value, symbol=str(instrument.symbol)),
            before=before, limit=1,
        )
        return rows[0] if rows else StoredSourceObservation(instrument, None, None)

    def _read(
        self, *, predicate: str, parameters: Mapping[str, Any],
        before: UtcTimestamp, limit: int,
    ) -> tuple[StoredSourceObservation, ...]:
        # Clip the exact merged keys, not aggregate max timestamps. ceil preserves
        # ts_open < before when before falls inside a minute, including epoch zero.
        rows = self.gateway.select(
            f"""SELECT market_id,symbol,
                if(bitmapCardinality(minutes)=0,NULL,bitmapMin(minutes)) AS first_minute,
                if(bitmapCardinality(minutes)=0,NULL,bitmapMax(minutes)) AS last_minute
            FROM (
                SELECT market_id,symbol,
                    bitmapSubsetInRange(groupBitmapMergeState(minute_bitmap_state),
                        toUInt32(0),toUInt32(%(before_minute)s)) AS minutes
                FROM {self.database}.canonical_candles_1m_stats
                WHERE {predicate} AND market_id IN %(markets)s
                GROUP BY market_id,symbol ORDER BY market_id,symbol LIMIT %(limit)s
            ) ORDER BY market_id,symbol
            SETTINGS max_threads=1, max_execution_time=10, max_memory_usage=134217728""",
            {**parameters, "before_minute": max(0, ceil(before.value.timestamp()/60)),
             "limit": limit, "markets": tuple(SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2)},
        )
        return tuple(StoredSourceObservation(
            self._instrument(row), self._timestamp(row["first_minute"]),
            self._timestamp(row["last_minute"]),
        ) for row in rows)

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if not 1 <= limit <= 50:
            raise ValueError("invalid_reconciliation_limit")

    @staticmethod
    def _instrument(row: Mapping[str, Any]) -> InstrumentId:
        return InstrumentId(MarketId(int(row["market_id"])), Symbol(str(row["symbol"])))

    @staticmethod
    def _timestamp(minute: Any) -> UtcTimestamp | None:
        return None if minute is None else UtcTimestamp(datetime.fromtimestamp(int(minute)*60, UTC))
