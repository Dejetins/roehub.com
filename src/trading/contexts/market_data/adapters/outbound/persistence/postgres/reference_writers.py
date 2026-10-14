"""Bounded versioned reference writes; omitted enrichment is preserved, NULL clears."""

from dataclasses import fields

from trading.contexts.market_data.application.dto import FundingInstrument
from trading.contexts.market_data.application.dto.reference_data import (
    InstrumentRefEnrichmentSnapshot,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp

from .ingestion_common import MAX_ROWS, bounded, upsert
from .ingestion_gateway import IngestionGateway


def _reference(row):
    result = {f.name: getattr(row, f.name) for f in fields(row)}
    if "instrument_id" in result:
        instrument = result.pop("instrument_id")
        result["market_id"], result["symbol"] = instrument.market_id.value, str(instrument.symbol)
    else:
        result["market_id"] = result["market_id"].value
        if "symbol" in result:
            result["symbol"] = str(result["symbol"])
    result["updated_at"] = result["updated_at"].value
    order = (
        "market_id",
        "symbol",
        "exchange_name",
        "market_type",
        "market_code",
        "instrument_key",
        "exchange",
        "status",
        "is_tradable",
        "is_enabled",
        "count_symbols",
        "base_asset",
        "quote_asset",
        "price_step",
        "qty_step",
        "min_notional",
        "funding_interval_minutes",
        "funding_interval_source",
        "funding_cap",
        "funding_floor",
        "updated_at",
    )
    if "instrument_id" in {f.name for f in fields(row)}:
        order = (
            "market_id",
            "symbol",
            "instrument_key",
            "exchange",
            "market_type",
            "status",
            "is_tradable",
            "base_asset",
            "quote_asset",
            "funding_interval_minutes",
            "funding_interval_source",
            "funding_cap",
            "funding_floor",
            "updated_at",
        )
    return {k: result[k] for k in order if k in result}


class PostgresMarketRefWriter:
    def __init__(self, gateway: IngestionGateway):
        self._gateway = gateway

    def insert(self, rows):
        batch = bounded(rows, _reference, max_rows=4)
        if batch:
            self._gateway.run(
                lambda c: upsert(c, "ref_market", batch, keys=("market_id",), version="updated_at")
            )

    def existing_market_ids(self, ids):
        values = [i.value for i in ids]
        if len(values) > MAX_ROWS:
            raise ValueError("reference read exceeds bound")
        return self._gateway.run(
            lambda c: {
                r[0]
                for r in c.execute(
                    "SELECT market_id FROM market_data.ref_market WHERE market_id=ANY(%s)",
                    (values,),
                )
            }
        )


class PostgresInstrumentRefWriter:
    def __init__(self, gateway: IngestionGateway):
        self._gateway = gateway

    def upsert(self, rows):
        self._write(bounded(rows, _reference, max_rows=1000), enrichment=False)

    def upsert_enrichment(self, rows):
        self._write(bounded(rows, _reference, max_rows=1000), enrichment=True)

    def _write(self, batch, *, enrichment):
        if not batch:
            return

        def operation(conn):
            status = [
                {k: r[k] for k in ("market_id", "symbol", "status", "is_tradable", "updated_at")}
                for r in batch
            ]
            upsert(
                conn,
                "ref_instruments",
                status,
                keys=("market_id", "symbol"),
                version="updated_at",
                name="status_stage",
            )
            if enrichment:
                details = [
                    {k: v for k, v in r.items() if k not in ("status", "is_tradable", "updated_at")}
                    for r in batch
                ]
                for r, original in zip(details, batch):
                    r["enrichment_updated_at"] = original["updated_at"]
                upsert(
                    conn,
                    "ref_instruments",
                    details,
                    keys=("market_id", "symbol"),
                    version="enrichment_updated_at",
                    order="enrichment_order",
                    name="enrichment_stage",
                )

        self._gateway.run(operation)

    def existing_latest_enrichment(self, *, market_id, symbols):
        if len(symbols) > MAX_ROWS:
            raise ValueError("reference read exceeds bound")
        return self._gateway.run(
            lambda c: {
                r[0]: InstrumentRefEnrichmentSnapshot(*r[1:])
                for r in c.execute(
                    "SELECT symbol,status,is_tradable,base_asset,quote_asset,"
                    "price_step,qty_step,min_notional FROM market_data.ref_instruments "
                    "WHERE market_id=%s AND symbol=ANY(%s)",
                    (market_id.value, list(map(str, symbols))),
                )
            }
        )

    def existing_latest(self, *, market_id, symbols):
        return {
            k: (v.status, v.is_tradable)
            for k, v in self.existing_latest_enrichment(
                market_id=market_id, symbols=symbols
            ).items()
        }


class PostgresFundingInstrumentStore:
    def __init__(self, gateway: IngestionGateway):
        self._gateway = gateway

    def upsert_funding_instruments(self, rows):
        batch = bounded(rows, _reference, max_rows=1000)
        if batch:
            self._gateway.run(
                lambda c: upsert(
                    c,
                    "funding_instrument_universe",
                    batch,
                    keys=("market_id", "symbol"),
                    version="updated_at",
                )
            )

    def _read(self, predicate, params):
        from psycopg import sql

        columns = (
            "market_id,symbol,instrument_key,exchange,market_type,status,is_tradable,"
            "base_asset,quote_asset,funding_interval_minutes,funding_interval_source,"
            "funding_cap,funding_floor,updated_at"
        )

        def operation(c):
            query = sql.SQL(
                "SELECT {} FROM market_data.funding_instrument_universe WHERE {} "
                "ORDER BY market_id,symbol"
            ).format(sql.SQL(columns), predicate)
            def streamed():
                with c.cursor(name="funding_universe") as cursor:
                    cursor.execute(query,params)
                    while batch := cursor.fetchmany(4096):
                        yield from batch
            return tuple(
                FundingInstrument(
                    InstrumentId(MarketId(r[0]), Symbol(r[1])),
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6],
                    r[7],
                    r[8],
                    r[9],
                    r[10],
                    r[11],
                    r[12],
                    UtcTimestamp(r[13]),
                )
                for r in streamed()
            )

        return self._gateway.run(operation)

    def list_tradable_funding_instruments(self, *, market_ids):
        from psycopg import sql

        if len(market_ids) > 4:
            raise ValueError("unsupported market universe")
        return self._read(
            sql.SQL("market_id=ANY(%s) AND is_tradable=1"), ([m.value for m in market_ids],)
        )

    def get_funding_instrument(self, instrument_id):
        from psycopg import sql

        rows = self._read(
            sql.SQL("market_id=%s AND symbol=%s AND is_tradable=1"),
            (instrument_id.market_id.value, str(instrument_id.symbol)),
        )
        return rows[0] if rows else None
