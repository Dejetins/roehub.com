from dataclasses import dataclass

from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp

from .canonical_candle_index_reader import PostgresCanonicalCandleIndexReader
from .history_coverage_reader import PostgresHistoryCoverageReader
from .ingestion_gateway import IngestionGateway
from .query_support import CoverageUnavailable, rows


@dataclass(frozen=True)
class PostgresCatalogCoverageReader:
    gateway: IngestionGateway

    def counts(self, *, market_ids, start_at, end_at):
        if (
            not 1 <= len(market_ids) <= 4
            or not 0 < (end_at - start_at).total_seconds() <= 7 * 86400
        ):
            raise ValueError("catalog requires <=4 markets and <=7 days")
        keys = self.gateway.run(
            lambda c: [
                (r["market_id"], r["symbol"])
                for r in rows(
                    c,
                    "SELECT market_id,symbol FROM market_data.instrument_generations "
                    "WHERE market_id=ANY(%s) ORDER BY market_id,symbol LIMIT 80001",
                    (list(market_ids),),
                )
            ]
        )
        return PostgresHistoryCoverageReader(self.gateway).counts(
            starts={k: start_at for k in keys}, end_at=end_at
        )

    def latest(self, *, market_id, symbol, before):
        try:
            value = PostgresCanonicalCandleIndexReader(self.gateway).max_ts_open_lt(
                instrument_id=InstrumentId(MarketId(market_id), Symbol(symbol)),
                before=UtcTimestamp(before),
            )
        except CoverageUnavailable:
            # This optional display field makes no claim that history is complete.
            # Fresh catalog instruments have no generation/index until their first write.
            return None
        return value.value if value else None
