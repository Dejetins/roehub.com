"""Lifecycle-owned PostgreSQL market gateways shared by composition roots."""

from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass
from typing import Mapping

from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)

_LOCK = threading.Lock()
_GATEWAYS: dict[tuple[str, str], IngestionGateway] = {}
_ALIASES = {"stream": "ingestion", "backfill": "ingestion", "artifacts": "bulk"}
_DEADLINES = {"ingestion": 5, "interactive": 1, "bulk": 30, "reconcile": 10, "reference": 5}


def market_gateway(environ: Mapping[str, str], *, workload: str = "reference") -> IngestionGateway:
    dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not dsn:
        raise ValueError("STRATEGY_PG_DSN is required for canonical PostgreSQL storage")
    workload = _ALIASES.get(workload, workload)
    if workload not in _DEADLINES:
        raise ValueError("invalid PostgreSQL market workload")
    key = (dsn, workload)
    with _LOCK:
        if key not in _GATEWAYS:
            _GATEWAYS[key] = IngestionGateway(
                dsn, workload=workload, deadline_s=_DEADLINES[workload]
            )
        return _GATEWAYS[key]


def close_market_gateways() -> None:
    with _LOCK:
        for gateway in _GATEWAYS.values():
            gateway.close()
        _GATEWAYS.clear()


atexit.register(close_market_gateways)


@dataclass(frozen=True)
class PostgresWriterFactory:
    environ: Mapping[str, str]

    def writer(self) -> PostgresRawKlineWriter:
        return PostgresRawKlineWriter(market_gateway(self.environ, workload="ingestion"))
