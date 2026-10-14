"""Chunk-independent exact Arrow value hash; explicit schema/null/value byte streams."""

from __future__ import annotations

import hashlib
import json
import struct

import pyarrow as pa


class CanonicalHasher:
    """Each field hashes null bytes and little-endian values independently of buffers.

    Final SHA256 binds ordered names/types/nullability, row count, and each field's
    two SHA256 streams. Timestamp units are preserved (the source schema is UTC ms).
    Strings use signed int64 byte lengths, -1 for NULL, followed by exact UTF-8.
    Numeric NULL payload is explicitly zero; its null byte remains one. Float64
    bytes are never rounded or cast to a lower precision.
    """

    def __init__(self):
        self.schema = None
        self.rows = 0
        self.nulls = []
        self.values = []

    def update_batch(self, batch):
        schema = [(f.name, str(f.type), f.nullable) for f in batch.schema]
        if self.schema is None:
            self.schema = schema
            self.nulls = [hashlib.sha256() for _ in schema]
            self.values = [hashlib.sha256() for _ in schema]
        assert schema == self.schema
        self.rows += batch.num_rows
        for field, column, nulls, values in zip(
            batch.schema, batch.columns, self.nulls, self.values
        ):
            nulls.update(column.is_null().to_numpy(zero_copy_only=False).astype("u1").tobytes())
            if pa.types.is_timestamp(field.type):
                values.update(
                    column.cast(pa.int64()).fill_null(0).to_numpy().astype("<i8").tobytes()
                )
            elif pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                a = column.fill_null(0).to_numpy()
                values.update(a.astype(a.dtype.newbyteorder("<")).tobytes())
            elif pa.types.is_string(field.type):
                encoded = bytearray()
                for value in column.to_pylist():
                    data = b"" if value is None else value.encode("utf8")
                    encoded.extend(struct.pack("<q", -1 if value is None else len(data)))
                    encoded.extend(data)
                values.update(encoded)
            else:
                raise ValueError(f"unsupported hash type: {field.type}")

    def hexdigest(self):
        result = hashlib.sha256(b"roehub-s09-canonical-values/v1\0")
        result.update(json.dumps(self.schema, separators=(",", ":")).encode("utf8"))
        result.update(struct.pack("<Q", self.rows))
        for nulls, values in zip(self.nulls, self.values):
            result.update(nulls.digest())
            result.update(values.digest())
        return result.hexdigest()


def require_exact_millisecond_timestamps(rows, schema):
    """Reject DB precision loss before casting datetimes to source UTC-ms Arrow."""
    positions = [i for i, field in enumerate(schema) if pa.types.is_timestamp(field.type)]
    for row in rows:
        for index in positions:
            value = row[index]
            if value is not None and value.microsecond % 1000:
                raise ValueError("database timestamp cannot be represented exactly in milliseconds")
