"""Real PG consumer fixture: synthetic provider input, production writes/jobs/events."""

import json
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg

from tests.real_infra.market_data.test_postgres_consumers import FixedClock
from tests.real_infra.market_data.test_transactional_ingestion_postgres import candle
from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_index_reader import (  # noqa: E501
    PostgresCanonicalCandleIndexReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.catalog_snapshot_repository import (  # noqa: E501
    PostgresCatalogSnapshotRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_range_verifier import (  # noqa: E501
    PostgresHistoryRangeVerifier,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.work_request_repository import (  # noqa: E501
    PostgresMarketDataWorkRequestRepository,
)
from trading.contexts.market_data.application.dto import InstrumentRefEnrichmentUpsert
from trading.contexts.market_data.application.services.work_requests import (
    MarketDataWorkRequestRunner,
)
from trading.contexts.market_data.application.use_cases.rest_fill_range_1m import (
    RestFillRange1mUseCase,
)
from trading.shared_kernel.primitives import MarketId, Symbol, UtcTimestamp

private = json.loads(Path(".local_artifacts/s08-browser/credentials.json").read_text())
dsn = private["dsn"]
gateway = PsycopgBacktestPostgresGateway(dsn=dsn)
if sys.argv[1] == "seed":
    now = datetime.now(UTC)
    PostgresCatalogSnapshotRepository(gateway=gateway).publish(
        market_ids=[MarketId(1)],
        rows=[
            InstrumentRefEnrichmentUpsert(
                MarketId(1),
                Symbol("BTCUSDT"),
                "ENABLED",
                1,
                "BTC",
                "USDT",
                0.01,
                0.001,
                5.0,
                UtcTimestamp(now),
            )
        ],
        refreshed_at=now,
    )
    with psycopg.connect(dsn) as c:
        c.execute("SELECT market_data.ensure_partitions('2026-10-01',1)")
        c.execute(
            "INSERT INTO market_data_stream_connections VALUES (1,'BTCUSDT',true,1,now()) "
            "ON CONFLICT (market_id,symbol) DO UPDATE SET connected=true,observed_at=now()"
        )
    print("seeded production catalog snapshot and real connection transition event")
elif sys.argv[1] == "range":
    with psycopg.connect(dsn) as c:
        last = c.execute(
            "SELECT max(end_at) FROM market_data_work_requests WHERE kind='candle_ingestion'"
        ).fetchone()[0]
    start = last or datetime(2026, 10, 12, tzinfo=UTC)
    print(
        json.dumps(
            {
                "start": start.strftime("%Y-%m-%dT%H:%M"),
                "end": (start + timedelta(minutes=6)).strftime("%Y-%m-%dT%H:%M"),
            }
        )
    )
else:

    class Provider:
        def stream_1m(self, instrument_id, time_range):
            start = time_range.start.value
            while start < time_range.end.value:
                row = candle(str(instrument_id.symbol))
                yield replace(
                    row,
                    candle=replace(
                        row.candle,
                        instrument_id=instrument_id,
                        ts_open=UtcTimestamp(start),
                        ts_close=UtcTimestamp(start + timedelta(minutes=1)),
                    ),
                )
                start += timedelta(minutes=1)

    ingestion = IngestionGateway(dsn)
    fill = RestFillRange1mUseCase(
        Provider(),
        PostgresRawKlineWriter(ingestion),
        FixedClock(datetime.now(UTC)),
        1,
        500,
        PostgresCanonicalCandleIndexReader(ingestion),
        PostgresHistoryRangeVerifier(ingestion),
    )
    runner = MarketDataWorkRequestRunner(
        store=PostgresMarketDataWorkRequestRepository(gateway=gateway, execution_dsn=dsn),
        fill=fill,
        refresh_catalog=lambda market, check: check(),
        max_windows=1,
        window_minutes=2,
    )
    assert runner.run_once()
    with psycopg.connect(dsn) as c:
        rows = c.execute(
            "SELECT state,completed_units,total_units,rows_read,rows_written "
            "FROM market_data_work_requests ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        bounds = c.execute(
            "SELECT start_at,end_at FROM market_data_work_requests ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        canonical = c.execute(
            "SELECT count(*) FROM market_data.canonical_candles_1m "
            "WHERE symbol='BTCUSDT' AND ts_open >= %s AND ts_open < %s",
            bounds,
        ).fetchone()[0]
    print(json.dumps({"job": rows, "canonical_count": canonical}))
    ingestion.close()
