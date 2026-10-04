"""Read the bounded proof range through the existing scoped Backtests candle consumer."""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import psycopg

from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client
from trading.contexts.backtest.adapters.outbound import (
    PostgresResearchOrganizationScopeResolver,
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest.application.services import OrganizationScopedCanonicalCandleReader
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleReader,
    ThreadLocalClickHouseConnectGateway,
)
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UserId,
    UtcTimestamp,
)

c = json.loads(Path(".local_artifacts/overview-preview-sep26/credentials.json").read_text())
port = (
    subprocess.check_output(["docker", "port", "roehub-navigator-ch", "8123/tcp"], text=True)
    .strip()
    .rsplit(":", 1)[1]
)
env = {
    **os.environ,
    "CH_HOST": "127.0.0.1",
    "CH_PORT": port,
    "CH_USER": "proof",
    "CH_PASSWORD": unquote(urlparse(c["dsn"]).password),
    "CH_DATABASE": "market_data",
}
settings = ClickHouseSettingsLoader(env).load()
with psycopg.connect(c["dsn"]) as db:
    actor = db.execute(
        "SELECT actor_user_id FROM market_data_work_requests WHERE job_id=%s",
        ("f956150f-52dc-4f14-8bde-0e4304bdae5a",),
    ).fetchone()[0]
reader = OrganizationScopedCanonicalCandleReader(
    scope_resolver=PostgresResearchOrganizationScopeResolver(
        gateway=PsycopgBacktestPostgresGateway(dsn=c["dsn"])
    ),
    canonical_reader=ClickHouseCanonicalCandleReader(
        ThreadLocalClickHouseConnectGateway(client_factory=lambda: _clickhouse_client(settings)),
        settings.database,
    ),
)
result = reader.read_1m_arrays(
    user_id=UserId(actor),
    instrument_id=InstrumentId(MarketId(1), Symbol("BTCUSDT")),
    time_range=TimeRange(
        UtcTimestamp(datetime(2024, 1, 1, tzinfo=UTC)),
        UtcTimestamp(datetime(2024, 1, 1, 0, 5, tzinfo=UTC)),
    ),
)
batch = result.candles
assert len(batch.open_time_ms) == 5
assert np.all(np.diff(batch.open_time_ms) == 60000)
assert np.isfinite(batch.ohlcv_f32).all()
print(
    json.dumps(
        {
            "reader": "OrganizationScopedCanonicalCandleReader",
            "source": "ClickHouseCanonicalCandleReader FINAL",
            "market": "binance:spot",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "range": "2024-01-01T00:00Z/2024-01-01T00:05Z",
            "rows": len(batch.open_time_ms),
            "monotonic": True,
            "finite_ohlcv": True,
            "proof_boundary": (
                "existing scoped offline/precompute reader; "
                "artifact publication and strategy execution not performed"
            ),
        }
    )
)
