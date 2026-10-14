"""Fixed-cutoff, generation/revision-aware observations from compact actual indexes."""

from dataclasses import dataclass

from trading.contexts.backtest_artifacts.application.services.base_source_reconciliation import (
    StoredSourceObservation,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.query_support import (
    observations,
    rows,
    timestamp,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp


@dataclass(frozen=True)
class PostgresStoredArtifactCoordinates:
    gateway: IngestionGateway

    def page(self, *, after, limit=50):
        self._validate(limit)
        return self.gateway.run(
            lambda c: tuple(
                InstrumentId(MarketId(r["market_id"]), Symbol(r["symbol"]))
                for r in rows(
                    c,
                    "SELECT market_id,symbol FROM market_data.instrument_generations "
                    "WHERE (market_id,symbol)>(%s,%s) AND market_id IN (1,2,3,4) "
                    "ORDER BY market_id,symbol LIMIT %s",
                    (*(after or (0, "")), limit),
                )
            )
        )

    def observation_page(self, *, after, before, limit=50):
        self._validate(limit)
        return self.gateway.run(
            lambda c: tuple(
                self._map(r)
                for r in observations(c, after=after or (0, ""), before=before.value, limit=limit)
            )
        )

    def observation(self, *, instrument, before):
        result = self.gateway.run(
            lambda c: observations(c, instrument=instrument, before=before.value, limit=1)
        )
        return (
            self._map(result[0])
            if result
            else StoredSourceObservation(instrument, None, None, index_state="unknown")
        )

    @staticmethod
    def _validate(limit):
        if not 1 <= limit <= 50:
            raise ValueError("invalid_reconciliation_limit")

    @staticmethod
    def _map(r):
        ready = r["clipped_count"] is not None

        def ts(k):
            return UtcTimestamp(timestamp(int(r[k]))) if ready and r[k] is not None else None

        return StoredSourceObservation(
            InstrumentId(MarketId(r["market_id"]), Symbol(r["symbol"])),
            ts("first_minute"),
            ts("last_minute"),
            r["generation"],
            r["content_revision"],
            r["funding_revision"],
            int(r["clipped_count"]) if ready else None,
            "ready" if ready else "unknown",
        )
