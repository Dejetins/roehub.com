"""Deduplicated public history discovery, consumed only by the trusted scheduler."""

from datetime import datetime
from typing import Any, Mapping
from uuid import uuid4

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)


class PostgresHistoryBoundsRepository:
    def __init__(self, *, gateway: BacktestPostgresGateway) -> None:
        self._gateway = gateway

    def read_or_request(
        self, *, market_id: int, symbol: str, now: datetime
    ) -> Mapping[str, Any] | None:
        self._gateway.execute(
            query="""SELECT request_market_data_history_bounds(
                %(market)s::smallint, %(symbol)s::text, %(now)s::timestamptz)""",
            parameters={"market": market_id, "symbol": symbol, "now": now},
        )
        return self._gateway.fetch_one(
            query="""SELECT state, first_open_at, updated_at FROM market_data_history_bounds
                     WHERE market_id = %(market)s AND symbol = %(symbol)s""",
            parameters={"market": market_id, "symbol": symbol},
        )

    def claim(self, *, now: datetime) -> Mapping[str, Any] | None:
        return self._gateway.fetch_one(
            query="""
            WITH candidate AS (
                SELECT market_id, symbol FROM market_data_history_bounds
                WHERE state = 'queued' OR (state = 'running'
                    AND updated_at < %(now)s::timestamptz - INTERVAL '5 minutes')
                ORDER BY updated_at FOR UPDATE SKIP LOCKED LIMIT 1
            )
            UPDATE market_data_history_bounds b SET state = 'running',
                updated_at = %(now)s, worker_token = %(token)s
            FROM candidate c WHERE b.market_id = c.market_id AND b.symbol = c.symbol RETURNING b.*
            """,
            parameters={"now": now, "token": uuid4()},
        )

    def finish(
        self, *, job: Mapping[str, Any], first_open_at: datetime | None, now: datetime
    ) -> None:
        self._gateway.execute(
            query="""
            UPDATE market_data_history_bounds SET state = %(state)s, first_open_at = %(first)s,
                updated_at = %(now)s, worker_token = NULL
            WHERE market_id = %(market)s AND symbol = %(symbol)s
              AND state = 'running' AND worker_token = %(token)s
            """,
            parameters={
                "market": job["market_id"],
                "symbol": job["symbol"],
                "token": job["worker_token"],
                "now": now,
                "first": first_open_at,
                "state": "ready" if first_open_at is not None else "unavailable",
            },
        )
