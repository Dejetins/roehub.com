"""Explicit research export queries; no storage SQL lives in RL entrypoints."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from typing import Any, overload

import numpy as np

from .query_support import BATCH_ROWS, rows

_FIELDS = (
    "open_time_ms",
    "close_time_ms",
    "open_f32",
    "high_f32",
    "low_f32",
    "close_f32",
    "volume_base_f32",
    "volume_quote_f32",
    "trades_count_i64",
)
_DTYPE = np.dtype(
    [(name, "i8" if name.endswith(("ms", "i64")) else "f4") for name in _FIELDS] + [("nulls", "u2")]
)


class _FeatureRows(Sequence[Mapping[str, Any]]):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    @overload
    def __getitem__(self, index: int) -> Mapping[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> "_FeatureRows": ...

    def __getitem__(self, index) -> Mapping[str, Any] | "_FeatureRows":
        if isinstance(index, slice):
            return _FeatureRows(self.data[index])
        row = self.data[index]
        return {
            name: None if int(row["nulls"]) & (1 << k) else row[name].item()
            for k, name in enumerate(_FIELDS)
        }


FEATURES = """SELECT (extract(epoch FROM ts_open)*1000)::bigint AS open_time_ms,
(extract(epoch FROM ts_close)*1000)::bigint AS close_time_ms,
open::real AS open_f32,high::real AS high_f32,low::real AS low_f32,close::real AS close_f32,
volume_base::real AS volume_base_f32,volume_quote::real AS volume_quote_f32,
trades_count AS trades_count_i64 FROM market_data.canonical_candles_1m
WHERE market_id=%s AND symbol=%s AND ts_open>=%s AND ts_open<%s ORDER BY ts_open"""
COVERAGE = """SELECT floor(extract(epoch FROM min(ts_open))/60)::bigint AS first_minute_key,
floor(extract(epoch FROM max(ts_open))/60)::bigint AS last_minute_key,
count(*) AS physical_rows,count(*) AS distinct_minutes,
count(volume_quote) AS volume_quote_rows,count(trades_count) AS trades_count_rows,
count(*) FILTER(WHERE volume_base=0) AS zero_volume_rows,
count(*) FILTER(WHERE volume_base>0 AND volume_quote IS NOT NULL) AS vwap_computable_rows
FROM market_data.canonical_candles_1m
WHERE market_id=%s AND symbol=%s AND ts_open>=%s AND ts_open<%s"""


def feature_rows(gateway, *, market_id, symbol, start, end):
    if not 0 < (end - start).total_seconds() <= 366 * 86400:
        raise ValueError("research_slab_requires_at_most_one_year")
    with tempfile.TemporaryFile() as output:

        def read(c):
            output.seek(0)
            output.truncate()
            batch = []
            count = 0
            for row in rows(c, FEATURES, (market_id, symbol, start, end)):
                nulls = sum(1 << k for k, name in enumerate(_FIELDS) if row[name] is None)
                batch.append(
                    tuple(0 if row[name] is None else row[name] for name in _FIELDS) + (nulls,)
                )
                count += 1
                if len(batch) == BATCH_ROWS:
                    np.asarray(batch, dtype=_DTYPE).tofile(output)
                    batch.clear()
            if batch:
                np.asarray(batch, dtype=_DTYPE).tofile(output)
            return count

        count = gateway.run(read, read_only=True)
        output.flush()
        data = (
            np.memmap(output, dtype=_DTYPE, mode="r", shape=(count,))
            if count
            else np.empty(0, dtype=_DTYPE)
        )
        return _FeatureRows(data)


def coverage_row(gateway, *, market_id, symbol, start, end):
    return gateway.run(
        lambda c: next(rows(c, COVERAGE, (market_id, symbol, start, end))), read_only=True
    )


def latest_minute(gateway, market_id):
    def read(c):
        unknown = c.execute(
            "SELECT 1 FROM market_data.instrument_generations "
            "WHERE market_id=%s AND index_state<>'ready' LIMIT 1",
            (market_id,),
        ).fetchone()
        if unknown:
            raise ValueError("canonical_coverage_unknown")
        return c.execute(
            "SELECT floor(extract(epoch FROM max(last_ts))/60)::bigint "
            "FROM market_data.candle_coverage_summary WHERE market_id=%s",
            (market_id,),
        ).fetchone()[0]

    return gateway.run(read, read_only=True)
