"""Disposable S06 proof of the current market schema across an encrypted backup."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from apps.migrations.storage import apply_postgres_migrations
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.stored_coordinates import (  # noqa: E501
    PostgresStoredArtifactCoordinates,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_coverage_reader import (  # noqa: E501
    PostgresHistoryCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_range_verifier import (  # noqa: E501
    PostgresHistoryRangeVerifier,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.query_support import minute
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.platform.postgres_budget import admin_connect
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)

BASE = datetime(2026, 2, 1, tzinfo=UTC)
INSTRUMENT = InstrumentId(MarketId(1), Symbol("S06BACKUP"))
END = BASE + timedelta(minutes=10)
TABLES = (
    "raw_binance_klines_1m",
    "canonical_candles_1m",
    "instrument_generations",
    "candle_coverage_days",
    "candle_coverage_summary",
    "source_absent_candles_1m",
    "candle_history_verification",
    "verification_leases",
)


def candle(offset: int, *, revision: int = 0) -> CandleWithMeta:
    opened = BASE + timedelta(minutes=offset)
    return CandleWithMeta(
        Candle(
            INSTRUMENT,
            UtcTimestamp(opened),
            UtcTimestamp(opened + timedelta(minutes=1)),
            10.0,
            20.0,
            1.0,
            11.0 + revision,
            2.0,
            3.0,
        ),
        CandleMeta(
            source="rest",
            ingested_at=UtcTimestamp(BASE + timedelta(days=2, seconds=revision)),
            instrument_key="binance:spot:S06BACKUP",
            ingest_id=uuid4(),
            trades_count=None,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        ),
    )


def run(mode: str) -> dict:
    if os.environ.get("ROEHUB_DISPOSABLE_STORAGE_PROOF") != "1":
        raise ValueError("disposable proof flag required")
    dsn = os.environ["ROEHUB_STORAGE_POSTGRES_DSN"]
    if mode == "seed":
        root = Path(__file__).resolve().parents[2]
        apply_postgres_migrations(
            dsn, repo_root=root, manifest_path=root / "migrations/postgres/manifest.json"
        )
        with admin_connect(dsn) as c:
            c.execute("SELECT market_data.ensure_partitions(%s::date,1)", (BASE.date(),))
    gateway = IngestionGateway(dsn, deadline_s=30)
    try:
        writer = PostgresRawKlineWriter(gateway)
        if mode == "seed":
            writer.write_1m([candle(0), candle(5)])
            writer.write_1m([candle(0, revision=1)])
            verifier = PostgresHistoryRangeVerifier(gateway)
            fence = verifier.acquire(instrument_id=INSTRUMENT)
            verifier.confirm(
                instrument_id=INSTRUMENT,
                time_range=TimeRange(UtcTimestamp(BASE), UtcTimestamp(END)),
                observed_minutes={minute(BASE), minute(BASE) + 5},
                observed_at=BASE,
                accept_absence=True,
                check_lease=lambda: None,
                fence=fence,
                origin=BASE,
            )
            verifier.release(instrument_id=INSTRUMENT, fence=fence)
        coverage = PostgresHistoryCoverageReader(gateway)
        coordinates = PostgresStoredArtifactCoordinates(gateway)
        before = coordinates.observation(instrument=INSTRUMENT, before=UtcTimestamp(END))
        actual = coverage.counts(starts={(1, "S06BACKUP"): BASE}, end_at=END)[(1, "S06BACKUP")]
        absent = coverage.source_absent_counts(starts={(1, "S06BACKUP"): BASE}, end_at=END)[
            (1, "S06BACKUP")
        ]
        assert actual == 2 and absent == 8 and before.actual_count == 2
        with admin_connect(dsn) as c:
            frontier_row = c.execute(
                "SELECT verified_until FROM market_data.candle_history_verification "
                "WHERE symbol='S06BACKUP'"
            ).fetchone()
            assert frontier_row is not None
            frontier = frontier_row[0]
            assert frontier == END
            checksums = {}
            for table in TABLES:
                rows = c.execute(
                    f"SELECT row_to_json(t)::text FROM market_data.{table} AS t "
                    "ORDER BY row_to_json(t)::text"
                ).fetchall()
                checksums[table] = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
            indexes = c.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname='market_data' ORDER BY indexname"
            ).fetchall()
        result = {
            "snapshot": {
                "generation": str(before.generation),
                "content_revision": before.content_revision,
                "funding_revision": before.funding_revision,
                "index_state": before.index_state,
                "actual_count": actual,
                "absent_count": absent,
                "frontier": frontier.isoformat(),
                "table_sha256": checksums,
                "index_sha256": hashlib.sha256(json.dumps(indexes).encode()).hexdigest(),
            }
        }
        if mode == "verify":
            writer.write_1m([candle(0, revision=2)])
            after = coordinates.observation(instrument=INSTRUMENT, before=UtcTimestamp(END))
            assert (before.first, before.last, before.actual_count, before.generation) == (
                after.first,
                after.last,
                after.actual_count,
                after.generation,
            )
            assert after.content_revision is not None and before.content_revision is not None
            assert after.content_revision > before.content_revision
            writer.write_1m([candle(1, revision=2)])
            assert (
                coverage.source_absent_counts(starts={(1, "S06BACKUP"): BASE}, end_at=END)[
                    (1, "S06BACKUP")
                ]
                == 7
            )
            with admin_connect(dsn) as c:
                verification_row = c.execute(
                        "SELECT count(*) FROM market_data.candle_history_verification "
                        "WHERE symbol='S06BACKUP'"
                    ).fetchone()
                assert verification_row is not None and verification_row[0] == 0
            result["post_restore_semantics"] = {
                "same_bounds_correction_changes_revision": True,
                "late_actual_removes_absence_and_invalidates_frontier": True,
            }
        return result
    finally:
        gateway.close()


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), sort_keys=True))
