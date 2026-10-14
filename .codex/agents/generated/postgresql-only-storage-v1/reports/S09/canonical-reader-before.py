"""Snapshot server cursors with bounded conversion; outputs use anonymous disk-backed arrays."""

from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass

import numpy as np

from trading.contexts.market_data.application.dto import CandleWithMeta, CanonicalCandleBatch1m
from trading.shared_kernel.primitives import Candle, CandleMeta, UtcTimestamp

from .ingestion_gateway import IngestionGateway
from .query_support import BATCH_ROWS, key, rows

CANDLE_RANGE = """SELECT market_id,symbol,instrument_key,ts_open,ts_close,
open,high,low,close,volume_base,volume_quote,trades_count,taker_buy_volume_base,
taker_buy_volume_quote,source,ingested_at,ingest_id
FROM market_data.canonical_candles_1m
WHERE market_id=%s AND symbol=%s AND ts_open>=%s AND ts_open<%s ORDER BY ts_open"""
ARRAY_RANGE = """SELECT (extract(epoch FROM ts_open)*1000)::bigint AS t,
(extract(epoch FROM ts_close)*1000)::bigint AS e,open,high,low,close,volume_base
FROM market_data.canonical_candles_1m
WHERE market_id=%s AND symbol=%s AND ts_open>=%s AND ts_open<%s ORDER BY ts_open"""


@dataclass(frozen=True)
class PostgresCanonicalCandleReader:
    gateway: IngestionGateway

    def read_1m(self, instrument_id, time_range):
        # Spool owns no live SQL session while the consumer is idle. Closing the
        # iterator closes the anonymous file; transport retries truncate partial data.
        with tempfile.TemporaryFile() as output:

            def read(c):
                output.seek(0)
                output.truncate()
                with c.cursor().copy(
                    "COPY (" + CANDLE_RANGE + ") TO STDOUT (FORMAT BINARY)",
                    (*key(instrument_id), time_range.start.value, time_range.end.value),
                ) as stream:
                    stream.set_types(
                        [
                            "int4",
                            "text",
                            "text",
                            "timestamptz",
                            "timestamptz",
                            "float8",
                            "float8",
                            "float8",
                            "float8",
                            "float8",
                            "float8",
                            "int8",
                            "float8",
                            "float8",
                            "text",
                            "timestamptz",
                            "uuid",
                        ]
                    )
                    names = (
                        "market_id",
                        "symbol",
                        "instrument_key",
                        "ts_open",
                        "ts_close",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume_base",
                        "volume_quote",
                        "trades_count",
                        "taker_buy_volume_base",
                        "taker_buy_volume_quote",
                        "source",
                        "ingested_at",
                        "ingest_id",
                    )
                    for values in stream.rows():
                        pickle.dump(dict(zip(names, values)), output, protocol=5)

            self.gateway.run(read)
            output.seek(0)
            while True:
                try:
                    r = pickle.load(output)
                except EOFError:
                    return
                yield CandleWithMeta(
                    Candle(
                        instrument_id,
                        UtcTimestamp(r["ts_open"]),
                        UtcTimestamp(r["ts_close"]),
                        r["open"],
                        r["high"],
                        r["low"],
                        r["close"],
                        r["volume_base"],
                        r["volume_quote"],
                    ),
                    CandleMeta(
                        source=r["source"],
                        ingested_at=UtcTimestamp(r["ingested_at"]),
                        ingest_id=r["ingest_id"],
                        instrument_key=r["instrument_key"],
                        trades_count=r["trades_count"],
                        taker_buy_volume_base=r["taker_buy_volume_base"],
                        taker_buy_volume_quote=r["taker_buy_volume_quote"],
                    ),
                )

    def read_1m_arrays(self, instrument_id, time_range):
        with (
            tempfile.TemporaryFile() as opens,
            tempfile.TemporaryFile() as closes,
            tempfile.TemporaryFile() as values,
        ):

            def read(c):
                for f in (opens, closes, values):
                    f.seek(0)
                    f.truncate()
                count = 0
                batch = []

                def flush():
                    np.asarray([r["t"] for r in batch], dtype=np.int64).tofile(opens)
                    np.asarray([r["e"] for r in batch], dtype=np.int64).tofile(closes)
                    np.asarray(
                        [
                            [r[k] for k in ("open", "high", "low", "close", "volume_base")]
                            for r in batch
                        ],
                        dtype=np.float32,
                    ).tofile(values)

                for r in rows(
                    c,
                    ARRAY_RANGE,
                    (*key(instrument_id), time_range.start.value, time_range.end.value),
                ):
                    batch.append(r)
                    count += 1
                    if len(batch) == BATCH_ROWS:
                        flush()
                        batch.clear()
                if batch:
                    flush()
                return count

            n = self.gateway.run(read)
            if not n:
                return CanonicalCandleBatch1m(
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    np.empty((0, 5), dtype=np.float32),
                )
            return CanonicalCandleBatch1m(
                np.memmap(opens, dtype=np.int64, mode="r", shape=(n,)),
                np.memmap(closes, dtype=np.int64, mode="r", shape=(n,)),
                np.memmap(values, dtype=np.float32, mode="r", shape=(n, 5)),
            )

    def latest(self, *, instrument_id, before, limit=100):
        if not 1 <= limit <= 8192:
            raise ValueError("invalid_tail_limit")
        # Summary/index identifies occupied bounds; timestamp predicates prune partitions.

        from .canonical_candle_index_reader import PostgresCanonicalCandleIndexReader

        index = PostgresCanonicalCandleIndexReader(self.gateway)
        first, last = index.bounds_1m(instrument_id=instrument_id, before=before)
        if first is None:
            return ()

        def read(c):
            query = CANDLE_RANGE.replace("ORDER BY ts_open", "ORDER BY ts_open DESC LIMIT %s")
            return list(rows(c, query, (*key(instrument_id), first.value, before.value, limit)))

        return tuple(reversed(self.gateway.run(read)))
