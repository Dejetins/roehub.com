"""Exact hash semantics, independent of Arrow layout and chunk boundaries."""

import hashlib
import json
import struct
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pytest

from tools.qa.postgres_production_hash import CanonicalHasher, require_exact_millisecond_timestamps


def digest(table, chunk_size):
    h = CanonicalHasher()
    for batch in table.to_batches(max_chunksize=chunk_size):
        h.update_batch(batch)
    return h.hexdigest()


def fixture():
    return pa.table(
        {
            "price": pa.array([1.0, None, -0.0, 0.0, np.nextafter(1.0, 2.0)], pa.float64()),
            "time": pa.array([0, 1, None, -1, 123456789], pa.timestamp("ms", "UTC")),
            "symbol": pa.array(["BTC", None, "", "Б", "ETH"]),
            "count": pa.array([0, 1, None, 3, 4], pa.uint32()),
        }
    )


def test_chunk_and_buffer_layout_independence():
    original = fixture()
    # Rebuild with separate allocations, nonzero array offsets and different chunks.
    padded = pa.concat_tables([original.slice(0, 2), original, original.slice(0, 1)])
    sliced = padded.slice(2, original.num_rows)
    rebuilt = pa.Table.from_pydict(original.to_pydict(), schema=original.schema)
    expected = digest(original, 5)
    for table in (
        original,
        sliced,
        rebuilt,
        pa.concat_tables([original.slice(0, 1), original.slice(1)]),
    ):
        for size in (1, 2, 3, 100):
            assert digest(table, size) == expected


def test_exact_value_null_order_and_schema_mutations_change_hash():
    original = fixture()
    expected = digest(original, 2)
    mutations = []
    for replacement in (
        [np.nextafter(1.0, 2.0), None, -0.0, 0.0, np.nextafter(1.0, 2.0)],
        [1.0, 0.0, -0.0, 0.0, np.nextafter(1.0, 2.0)],
        [1.0, None, 0.0, 0.0, np.nextafter(1.0, 2.0)],
    ):
        mutations.append(original.set_column(0, "price", pa.array(replacement, pa.float64())))
    mutations.append(original.take(pa.array([4, 3, 2, 1, 0])))
    mutations.append(original.rename_columns(["other", "time", "symbol", "count"]))
    mutations.append(original.set_column(0, "price", original.column(0).cast(pa.float32())))
    for mutation in mutations:
        assert digest(mutation, 3) != expected


def test_independent_byte_specification():
    table = pa.table({"x": pa.array([1.0, None, -0.0], pa.float64())})
    result = hashlib.sha256(b"roehub-s09-canonical-values/v1\0")
    result.update(json.dumps([("x", "double", True)], separators=(",", ":")).encode())
    result.update(struct.pack("<Q", 3))
    result.update(hashlib.sha256(bytes([0, 1, 0])).digest())
    result.update(hashlib.sha256(struct.pack("<ddd", 1.0, 0.0, -0.0)).digest())
    assert digest(table, 1) == result.hexdigest()


def test_database_submillisecond_difference_is_rejected_before_arrow_cast():
    schema = pa.schema([("ts_open", pa.timestamp("ms", "UTC"))])
    exact = datetime(2026, 1, 1, microsecond=123000, tzinfo=UTC)
    changed = exact.replace(microsecond=123001)
    # Arrow alone erases the difference; verifier must reject before conversion.
    assert pa.array([exact], schema.field(0).type).equals(pa.array([changed], schema.field(0).type))
    require_exact_millisecond_timestamps([(exact,)], schema)
    with pytest.raises(ValueError, match="represented exactly"):
        require_exact_millisecond_timestamps([(changed,)], schema)
