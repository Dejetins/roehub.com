"""Funding writes share the candle generation lock and durable revision contract."""

from collections.abc import Sequence

from trading.contexts.market_data.application.dto import FundingRateRecord

from .ingestion_common import bounded, lock_generations, stage, upsert
from .ingestion_gateway import IngestionGateway


class PostgresFundingRateWriter:
    def __init__(self, gateway: IngestionGateway):
        self._gateway = gateway

    def write_funding_rates(self, rows: Sequence[FundingRateRecord]) -> None:
        batch = bounded(rows, _payload, max_rows=1000)
        if batch:
            self._gateway.prepare_months(r["funding_time"] for r in batch)
            self._gateway.run(lambda conn: self._write(conn, batch))

    def _write(self, conn, batch):
        lock_generations(conn, ((r["market_id"], r["symbol"]) for r in batch))
        canonical = [
            {k: v for k, v in r.items() if k not in ("mark_price", "category")} for r in batch
        ]
        # Winners use the exact same whole-record precedence for canonical and raw,
        # including provider-only fields; canonical stores that same order token.
        from .ingestion_common import payload_order

        for source, target in zip(batch, canonical):
            target["payload_order"] = payload_order(
                v for k, v in source.items() if k not in ("ingested_at", "ingest_id")
            )
        stage(conn, "canonical_funding_rates", canonical, name="funding_stage")
        conn.execute(
            "CREATE TEMP TABLE funding_winners ON COMMIT DROP AS "
            "SELECT DISTINCT ON (market_id,symbol,funding_time) * FROM funding_stage "
            "ORDER BY market_id,symbol,funding_time,ingested_at DESC,payload_order DESC"
        )
        changed = conn.execute("""SELECT DISTINCT s.market_id,s.symbol FROM funding_winners s
            LEFT JOIN market_data.canonical_funding_rates c USING (market_id,symbol,funding_time)
            WHERE (c.funding_time IS NULL OR (s.ingested_at,s.payload_order) >
                (c.ingested_at,c.payload_order)) AND (c.funding_time IS NULL OR
                ROW(s.instrument_key,s.funding_rate,s.funding_interval_minutes,
                    s.funding_interval_source) IS DISTINCT FROM
                ROW(c.instrument_key,c.funding_rate,c.funding_interval_minutes,
                    c.funding_interval_source))""").fetchall()
        upsert(
            conn,
            "canonical_funding_rates",
            canonical,
            keys=("market_id", "symbol", "funding_time"),
            preserve_order=True,
        )
        for market, table, omitted in (
            (2, "raw_binance_funding_rates", "category"),
            (4, "raw_bybit_funding_rates", "mark_price"),
        ):
            raw = [
                {k: v for k, v in r.items() if k != omitted}
                for r in batch
                if r["market_id"] == market
            ]
            for r in raw:
                r["payload_order"] = payload_order(
                    v for k, v in r.items() if k not in ("ingested_at", "ingest_id")
                )
            upsert(
                conn,
                table,
                raw,
                keys=("market_id", "symbol", "funding_time"),
                name=table + "_stage",
                preserve_order=True,
            )
        for market, symbol in changed:
            conn.execute(
                "UPDATE market_data.instrument_generations "
                "SET funding_revision=funding_revision+1,updated_at=clock_timestamp() "
                "WHERE market_id=%s AND symbol=%s",
                (market, symbol),
            )


def _payload(row: FundingRateRecord):
    market = row.instrument_id.market_id.value
    if market not in (2, 4):
        raise ValueError("funding requires a supported futures market")
    result = dict(
        market_id=market,
        symbol=str(row.instrument_id.symbol),
        instrument_key=row.instrument_key,
        funding_time=row.funding_time.value,
        funding_rate=float(row.funding_rate),
        funding_interval_minutes=row.funding_interval_minutes,
        funding_interval_source=row.funding_interval_source,
        source=row.source,
        ingested_at=row.ingested_at.value,
        ingest_id=row.ingest_id,
    )
    if market == 2:
        result["mark_price"] = row.mark_price
    else:
        result["category"] = row.bybit_category or "linear"
    order = (
        "market_id",
        "symbol",
        "instrument_key",
        "category",
        "funding_time",
        "funding_rate",
        "funding_interval_minutes",
        "funding_interval_source",
        "mark_price",
        "source",
        "ingested_at",
        "ingest_id",
    )
    return {k: result[k] for k in order if k in result}
