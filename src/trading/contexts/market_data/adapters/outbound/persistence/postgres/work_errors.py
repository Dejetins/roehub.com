"""Sanitized PostgreSQL retry classification; driver messages are never persisted."""

import psycopg

from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
)

from .ingestion_gateway import retryable
from .query_support import CoverageUnavailable


def transient_postgres_error(error: Exception) -> str | None:
    if isinstance(error, TemporaryIngestionError):
        return (
            "storage_busy"
            if "saturat" in str(error) or "admission" in str(error)
            else "storage_unavailable"
        )
    if isinstance(error, CoverageUnavailable) and str(error) == "verification_lease_busy":
        return "storage_busy"
    state = getattr(error, "sqlstate", None)
    if (
        state == "23514"
        and isinstance(error, psycopg.Error)
        and (error.diag.message_primary or "").startswith("no partition of relation ")
        and error.diag.schema_name == "market_data"
        and error.diag.table_name
        in {
            "raw_binance_klines_1m",
            "raw_bybit_klines_1m",
            "canonical_candles_1m",
            "raw_binance_funding_rates",
            "raw_bybit_funding_rates",
            "canonical_funding_rates",
        }
    ):
        return "storage_partition_pending"
    if state in {"53300", "55P03", "57014"}:
        return "storage_busy"
    if state == "53200":
        return "storage_memory_pressure"
    return "storage_unavailable" if isinstance(error, psycopg.Error) and retryable(error) else None
