"""Explicit disposable server proof of class grants, MV writes and forbidden overrides."""

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import clickhouse_connect
import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from apps.migrations.storage import split_clickhouse_sql
from tools.installation.clickhouse_workloads import WORKLOADS, grants
from tools.qa.market_data_mixed_load import raw_insert
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.funding_rate_store import (  # noqa: E501
    ClickHouseFundingRateStore,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseConnectGateway,
)
from trading.contexts.market_data.application.dto import FundingInstrument, FundingRateRecord
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp

pytestmark = pytest.mark.skipif(
    os.environ.get("ROEHUB_S02_DISPOSABLE") != "1",
    reason="explicit disposable class identities required",
)


@pytest.fixture
def workloads():
    admin = clickhouse_connect.get_client(
        host=os.environ["CH_HOST"],
        port=int(os.environ["CH_PORT"]),
        username=os.environ["CH_USER"],
        password=os.environ["CH_PASSWORD"],
    )
    db = "resource_s02_grants_" + uuid4().hex
    for filename in (
        "market_data_ddl.sql",
        "funding_rates_ddl.sql",
        "history_source_absences_ddl.sql",
    ):
        for sql in split_clickhouse_sql(
            (Path("migrations/clickhouse") / filename).read_text().replace("market_data", db)
        ):
            admin.command(sql)
    clients = {}
    try:
        for w in WORKLOADS:
            for grant in grants(w, db):
                admin.command(grant + " TO roehub_" + w)
            clients[w] = clickhouse_connect.get_client(
                host=os.environ["CH_HOST"],
                port=int(os.environ["CH_PORT"]),
                username="roehub_" + w,
                password=Path(os.environ["ROEHUB_CH_" + w.upper() + "_PASSWORD_FILE"]).read_text(),
            )
        yield admin, db, clients
    finally:
        for client in clients.values():
            client.close()
        admin.command(f"DROP DATABASE {db}")
        admin.close()


def test_real_grants_mv_and_normal_scheduler_funding(workloads):
    admin, db, clients = workloads
    for w in ("stream", "backfill"):
        raw_insert(clients[w], db, "BTCUSDT", 29453760, 2, "s02_grants")
    assert admin.query(f"SELECT uniqExact(ts_open) FROM {db}.canonical_candles_1m").result_rows == [
        (2,)
    ]
    assert admin.query(
        f"SELECT groupBitmapMerge(minute_bitmap_state) FROM {db}.canonical_candles_1m_stats"
    ).result_rows == [(2,)]
    backfill = clients["backfill"]
    backfill.insert(
        f"{db}.ref_market",
        [(1, "binance", "spot", "binance:spot")],
        column_names=["market_id", "exchange_name", "market_type", "market_code"],
    )
    backfill.insert(f"{db}.ref_instruments", [(1, "BTCUSDT")], column_names=["market_id", "symbol"])
    store = ClickHouseFundingRateStore(gateway=ClickHouseConnectGateway(backfill), database=db)
    ts = UtcTimestamp(datetime(2026, 1, 1, tzinfo=UTC))
    for market, exchange in [(2, "binance"), (4, "bybit")]:
        instrument = InstrumentId(MarketId(market), Symbol("BTCUSDT"))
        store.upsert_funding_instruments(
            [
                FundingInstrument(
                    instrument_id=instrument,
                    instrument_key=f"{exchange}:futures:BTCUSDT",
                    exchange=exchange,
                    market_type="futures",
                    status="TRADING",
                    is_tradable=1,
                    base_asset="BTC",
                    quote_asset="USDT",
                    funding_interval_minutes=480,
                    funding_interval_source="test",
                    funding_cap=None,
                    funding_floor=None,
                    updated_at=ts,
                )
            ]
        )
        store.write_funding_rates(
            [
                FundingRateRecord(
                    instrument_id=instrument,
                    instrument_key=f"{exchange}:futures:BTCUSDT",
                    funding_time=ts,
                    funding_rate=0.001,
                    funding_interval_minutes=480,
                    funding_interval_source="test",
                    source="test",
                    ingested_at=ts,
                    ingest_id=str(uuid4()),
                    bybit_category="linear" if market == 4 else None,
                )
            ]
        )
        read = ClickHouseFundingRateStore(
            gateway=ClickHouseConnectGateway(clients["artifacts"]), database=db
        )
        assert read.latest_funding_time(instrument) == ts
    for w in ("interactive", "artifacts"):
        with pytest.raises(DatabaseError, match="497"):
            raw_insert(clients[w], db, "BTCUSDT", 29453762, 1, "denied")
    with pytest.raises(DatabaseError, match="497"):
        clients["stream"].command(f"INSERT INTO {db}.ref_market (market_id) VALUES (4)")


@pytest.mark.parametrize(
    "query",
    [
        "SET async_insert=1",
        "SET wait_for_async_insert=0",
        "SELECT 1 SETTINGS async_insert=1",
        "SET profile='default'",
        "SELECT 1 SETTINGS profile='default'",
        "SET max_memory_usage=0",
        "SET max_memory_usage_for_user=0",
        "SET max_concurrent_queries_for_user=0",
        "SET max_threads=100",
        "SET max_execution_time=0",
        "SET timeout_overflow_mode='break'",
    ],
)
def test_real_cap_overrides_are_denied(workloads, query):
    _, _, clients = workloads
    for c in clients.values():
        with pytest.raises(DatabaseError, match="452"):
            c.command(query)
