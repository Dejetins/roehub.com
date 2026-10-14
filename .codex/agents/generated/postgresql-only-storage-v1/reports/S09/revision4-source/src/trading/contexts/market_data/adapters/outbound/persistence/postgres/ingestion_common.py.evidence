"""Normalized payload order and bounded COPY helpers shared by market writers."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from psycopg import sql

MAX_ROWS = 8192
MAX_BYTES = 8 * 1024 * 1024


def utc_ms(value: datetime) -> datetime:
    value = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return value.replace(microsecond=value.microsecond // 1000 * 1000)


def normalized(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc_ms(value)
    if isinstance(value, float):
        if math.isnan(value):
            return float("nan")
        return 0.0 if value == 0 else value
    return value


def payload_order(values: Iterable[Any]) -> bytes:
    result = bytearray(b"\x01")
    for value in values:
        value = normalized(value)
        if value is None:
            body = b"n"
        elif isinstance(value, datetime):
            epoch = value - datetime(1970, 1, 1, tzinfo=timezone.utc)
            body = b"t" + struct.pack(
                ">q", epoch.days * 86400000 + epoch.seconds * 1000 + epoch.microseconds // 1000
            )
        elif isinstance(value, (int, bool)):
            body = b"i" + struct.pack(">q", value)
        elif isinstance(value, float):
            body = b"f" + (
                bytes.fromhex("7ff8000000000000") if math.isnan(value) else struct.pack(">d", value)
            )
        elif isinstance(value, str):
            body = b"s" + value.encode("utf8")
        else:
            raise TypeError("unsupported payload field")
        result.extend(struct.pack(">I", len(body)))
        result.extend(body)
    return bytes(result)


def bounded(rows: Iterable[Any], convert, *, max_rows: int = MAX_ROWS) -> list[dict[str, Any]]:
    result = []
    size = 0
    for row in rows:
        if len(result) >= max_rows:
            raise ValueError(f"batch exceeds {max_rows} rows; caller must split before writing")
        item = {k: normalized(v) for k, v in convert(row).items()}
        # Includes UUID text and transport overhead; fail before any SQL side effect.
        size += 4 * sum(len(str(v).encode("utf8")) + 16 for v in item.values())
        if size > MAX_BYTES:
            raise ValueError("batch exceeds 8 MiB")
        result.append(item)
    return result


def stage(
    conn, table: str, rows: list[dict[str, Any]], *, name: str = "ingest_stage", reuse: bool = False
) -> str:
    columns = list(rows[0])
    conn.execute(
        sql.SQL(
            "CREATE TEMP TABLE {}{} (LIKE market_data.{} INCLUDING DEFAULTS) ON COMMIT {}"
        ).format(
            sql.SQL("IF NOT EXISTS " if reuse else ""),
            sql.Identifier("pg_temp", name),
            sql.Identifier(table),
            sql.SQL("DELETE ROWS" if reuse else "DROP"),
        )
    )
    if reuse:
        conn.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier("pg_temp", name)))
    with conn.cursor().copy(
        sql.SQL("COPY {} ({}) FROM STDIN").format(
            sql.Identifier("pg_temp", name), sql.SQL(",").join(map(sql.Identifier, columns))
        )
    ) as copy:
        for row in rows:
            copy.write_row(tuple(row[c] for c in columns))
    return name


def upsert(
    conn,
    table: str,
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    version: str = "ingested_at",
    name: str = "ingest_stage",
    preserve_order: bool = False,
    order: str = "payload_order",
) -> None:
    if not rows:
        return
    for row in rows:
        if not preserve_order:
            row[order] = payload_order(
                v for k, v in row.items() if k not in (version, "ingest_id", order)
            )
    stage(conn, table, rows, name=name)
    upsert_staged(conn, table, list(rows[0]), keys=keys, version=version, name=name, order=order)


def upsert_staged(
    conn,
    table: str,
    columns: list[str],
    *,
    keys: tuple[str, ...],
    version: str = "ingested_at",
    name: str = "ingest_stage",
    order: str = "payload_order",
) -> None:
    """Apply the same versioned winner policy to an already populated local stage."""
    cols = columns

    def identifiers(names):
        return sql.SQL(",").join(map(sql.Identifier, names))

    assignments = sql.SQL(",").join(
        sql.SQL("{}=EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
        for c in cols
        if c not in keys
    )
    conn.execute(
        sql.SQL(
            "INSERT INTO market_data.{table} AS target ({cols}) "
            "SELECT DISTINCT ON ({keys}) {cols} FROM {stage} "
            "ORDER BY {keys}, {version} DESC, {order} DESC "
            "ON CONFLICT ({keys}) DO UPDATE SET {assignments} "
            "WHERE (EXCLUDED.{version},EXCLUDED.{order}) > "
            "(target.{version},target.{order})"
        ).format(
            table=sql.Identifier(table),
            cols=identifiers(cols),
            keys=identifiers(keys),
            stage=sql.Identifier("pg_temp", name),
            version=sql.Identifier(version),
            order=sql.Identifier(order),
            assignments=assignments,
        )
    )


def lock_generations(conn, keys):
    result = {}
    for market, symbol in sorted(set(keys)):
        conn.execute(
            "INSERT INTO market_data.instrument_generations (market_id,symbol) "
            "VALUES (%s,%s) ON CONFLICT DO NOTHING",
            (market, symbol),
        )
        row = conn.execute(
            "SELECT generation,index_state,invalidation_reason FROM "
            "market_data.instrument_generations WHERE market_id=%s AND symbol=%s "
            "FOR UPDATE",
            (market, symbol),
        ).fetchone()
        assert row is not None
        if row[1] != "ready" and row[2] != "initial":
            raise RuntimeError("source generation requires bounded index rebuild")
        result[(market, symbol)] = row[0]
    return result
