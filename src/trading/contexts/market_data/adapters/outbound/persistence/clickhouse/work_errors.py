"""Allowlisted driver failures; never persist driver messages or connection URLs."""

import re

from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError


def transient_clickhouse_error(error: Exception) -> str | None:
    if not isinstance(error, DatabaseError):
        return None
    # clickhouse-connect exposes the response code in its formatted exception,
    # not as a structured attribute. Inspect in memory only; do not log this text.
    match = re.search(r"(?:ClickHouse error code |(?:^|\n)\s*Code:\s*)(\d+)\b", str(error))
    if match:
        code = int(match.group(1))
        if code == 241:
            return "storage_memory_pressure"
        if code in {159, 209, 210}:
            return "storage_unavailable"
        return None
    return "storage_unavailable" if isinstance(error, OperationalError) else None
