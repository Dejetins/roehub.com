from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.market_data.application.dto import InstrumentRefEnrichmentUpsert
from trading.shared_kernel.primitives import MarketId


class PostgresCatalogSnapshotRepository:
    """Immutable metadata projections; never stores credentials or candle rows."""

    def __init__(self, *, gateway: BacktestPostgresGateway) -> None:
        self._gateway = gateway

    def publish(
        self,
        *,
        market_ids: Sequence[MarketId],
        rows: Sequence[InstrumentRefEnrichmentUpsert],
        refreshed_at: datetime,
    ) -> None:
        for market in market_ids:
            items = [
                dict(
                    symbol=str(r.symbol),
                    base_asset=r.base_asset,
                    quote_asset=r.quote_asset,
                    price_step=r.price_step,
                    qty_step=r.qty_step,
                    min_notional=r.min_notional,
                )
                for r in rows
                if r.market_id == market and r.status == "ENABLED" and r.is_tradable
            ]
            items.sort(key=lambda r: str(r["symbol"]))
            if len(items) > 20000 or len({r["symbol"] for r in items}) != len(items):
                raise ValueError("catalog snapshot exceeds limits or contains duplicate symbols")
            self._gateway.execute(
                query="""
                INSERT INTO market_data_catalog_snapshots
                    (snapshot_id, market_id, refreshed_at, items)
                VALUES (%(id)s, %(market)s, %(at)s, %(items)s::jsonb)
                """,
                parameters={
                    "id": uuid4(),
                    "market": market.value,
                    "at": refreshed_at,
                    "items": json.dumps(items, allow_nan=False),
                },
            )
            # Retain only three public projections per market. Expired cursors
            # receive 409; candle history and organization selections are untouched.
            self._gateway.execute(
                query="""
                DELETE FROM market_data_catalog_snapshots
                WHERE market_id = %(market)s AND snapshot_id NOT IN (
                    SELECT snapshot_id FROM market_data_catalog_snapshots
                    WHERE market_id = %(market)s
                    ORDER BY refreshed_at DESC, snapshot_id DESC LIMIT 3
                )
                """, parameters={"market": market.value},
            )

    def latest(
        self, *, market_id: int, snapshot_id: UUID | None = None
    ) -> Mapping[str, Any] | None:
        return self._gateway.fetch_one(
            query="""
            SELECT snapshot_id, market_id, refreshed_at, jsonb_array_length(items) AS total
            FROM market_data_catalog_snapshots
            WHERE market_id = %(market)s AND (%(id)s::uuid IS NULL OR snapshot_id = %(id)s)
            ORDER BY refreshed_at DESC, snapshot_id DESC LIMIT 1
            """,
            parameters={"market": market_id, "id": snapshot_id},
        )

    def current_catalogs(
        self, *, market_ids: Sequence[int], snapshot_ids: Sequence[UUID] = ()
    ) -> tuple[Mapping[str, Any], ...]:
        """At most four immutable, already bounded catalog projections."""
        if not 0 < len(market_ids) <= 4:
            raise ValueError("invalid catalog market count")
        return self._gateway.fetch_all(
            query="""
            SELECT DISTINCT ON (market_id) snapshot_id, market_id, refreshed_at, items
            FROM market_data_catalog_snapshots
            WHERE market_id = ANY(%(markets)s)
              AND (%(ids)s::uuid[] IS NULL OR snapshot_id = ANY(%(ids)s::uuid[]))
            ORDER BY market_id, refreshed_at DESC, snapshot_id DESC
            """,
            parameters={"markets": list(market_ids), "ids": list(snapshot_ids) or None},
        )

    def page(
        self,
        *,
        snapshot_id: UUID,
        prefix: str,
        after: str,
        limit: int,
        selected_symbols: Sequence[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        return self._gateway.fetch_all(
            query="""
            SELECT item FROM market_data_catalog_snapshots s,
                LATERAL jsonb_array_elements(s.items) item
            WHERE s.snapshot_id = %(id)s AND item->>'symbol' > %(after)s
              AND starts_with(item->>'symbol', %(prefix)s)
              AND (%(selected)s::text[] IS NULL OR item->>'symbol' = ANY(%(selected)s))
            ORDER BY item->>'symbol' LIMIT %(limit)s
            """,
            parameters={
                "id": snapshot_id,
                "prefix": prefix,
                "after": after,
                "limit": limit,
                "selected": selected_symbols,
            },
        )
