"""Recover five selected historical metadata records, retaining original freshness."""

import json
import os

import psycopg

from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.reference_writers import (
    PostgresInstrumentRefWriter,
)
from trading.contexts.market_data.application.dto.reference_data import (
    InstrumentRefEnrichmentUpsert,
)
from trading.platform.postgres_budget import connect
from trading.shared_kernel.primitives import MarketId, Symbol, UtcTimestamp

with connect(os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin") as target:
    selected = {
        (m, s)
        for m, s in target.execute(
            "SELECT DISTINCT market_id,symbol FROM market_data_instrument_selections"
        )
    }
# This isolated restore database is evidence, not an installation workload.
with psycopg.connect("host=roehub-s10-recovery-verify dbname=restore user=postgres") as recovered:
    snapshots = recovered.execute(
        "SELECT DISTINCT ON (market_id) market_id,refreshed_at,items FROM market_data_catalog_snapshots ORDER BY market_id,refreshed_at DESC,snapshot_id DESC"  # noqa: E501
    ).fetchall()
    rows = []
    for market, stamp, items in snapshots:
        for item in items:
            if (market, item["symbol"]) in selected:
                rows.append(
                    InstrumentRefEnrichmentUpsert(
                        MarketId(market),
                        Symbol(item["symbol"]),
                        "ENABLED",
                        1,
                        item.get("base_asset"),
                        item.get("quote_asset"),
                        item.get("price_step"),
                        item.get("qty_step"),
                        item.get("min_notional"),
                        UtcTimestamp(stamp),
                    )
                )
    assert len(rows) == 5
    gateway = IngestionGateway(os.environ["STRATEGY_PG_DSN"], workload="reference")
    try:
        PostgresInstrumentRefWriter(gateway).upsert_enrichment(rows)
    finally:
        gateway.close()
    print(
        json.dumps(
            {
                "recovered_selected_metadata": len(rows),
                "original_updated_at": [r.updated_at.value for r in rows],
                "current_catalog_published": False,
                "live_sync_claimed": False,
            },
            default=str,
        )
    )
