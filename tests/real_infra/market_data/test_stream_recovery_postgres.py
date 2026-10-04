"""Real PostgreSQL journal proof in an isolated disposable schema."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import LiteralString, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from trading.contexts.market_data.adapters.outbound.persistence.postgres.stream_recovery_store import (  # noqa: E501
    PostgresStreamRecoveryStore,
)
from trading.contexts.market_data.application.dto import RestFillTask
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

DSN = os.environ.get("ROEHUB_MARKET_DATA_TEST_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PostgreSQL target required")


@pytest.fixture
def store():
    schema = "stream_recovery_test_" + uuid4().hex
    with psycopg.connect(DSN) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        migration = Path("migrations/postgres/0027_market_data_stream_recovery_v1.sql").read_text()
        conn.execute(sql.SQL(cast(LiteralString, migration)))
    repo = PostgresStreamRecoveryStore(
        make_conninfo(DSN, options=f"-c search_path={schema}"), max_entries=2
    )
    try:
        yield repo
    finally:
        with psycopg.connect(DSN) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def item(symbol="AAVEUSDT"):
    now = datetime(2026, 10, 4, tzinfo=UTC)
    return RestFillTask(
        InstrumentId(MarketId(1), Symbol(symbol)),
        TimeRange(UtcTimestamp(now), UtcTimestamp(now + timedelta(minutes=1))),
        "ws_recovery",
    )


def test_restart_and_fenced_ack(store):
    first = store.remember(item())
    assert store.remember(item()).token == first.token
    restarted = PostgresStreamRecoveryStore(store.dsn)
    assert restarted.pending() == [first]
    store.complete(first)
    second = store.remember(item())
    assert second.token != first.token
    store.complete(first)  # late ack may not erase the new receipt
    assert store.pending() == [second]
    store.complete_many([first, second])
    assert not store.pending()


def test_capacity_terminal_state_and_explicit_rearm(store):
    first = store.remember(item())
    second = store.remember(item("ADAUSDT"))
    with pytest.raises(RuntimeError, match="capacity"):
        store.remember(item("ALGOUSDT"))
    store.fail_many([first], code="non_retryable_ingestion_error")
    assert store.pending() == [second]
    with pytest.raises(RuntimeError, match="requires_rearm"):
        store.remember(item())
    assert store.rearm_failed() == 1
    assert len(store.pending()) == 2
    store.complete(first)
    assert len(store.pending()) == 2  # pre-rearm ack is stale


def test_retry_after_persists_and_live_collector_is_exclusive(store):
    from dataclasses import replace

    first = store.remember(item())
    next_at = datetime.now(UTC) + timedelta(seconds=60)
    store.retry(replace(first, attempt=3), next_retry_at=next_at, code="source_rate_limited")
    entry = store.pending()[0]
    assert entry.attempt == 3 and entry.next_retry_at == next_at
    other = PostgresStreamRecoveryStore(store.dsn)
    with store.lease():
        with pytest.raises(RuntimeError, match="already running"):
            with other.lease():
                pass
    with other.lease():
        assert other.pending()[0] == entry


def test_window_checkpoint_keeps_identity_and_survives_reopen(store):
    from dataclasses import replace

    original = item()
    original = replace(
        original,
        time_range=TimeRange(
            original.time_range.start,
            UtcTimestamp(original.time_range.start.value + timedelta(hours=3)),
        ),
    )
    entry = store.remember(original)
    next_at = original.time_range.start.value + timedelta(hours=1)
    store.advance(entry, start_at=next_at)
    after = PostgresStreamRecoveryStore(store.dsn).pending()[0]
    assert after.task.time_range.start.value == next_at
    assert after.task_key == entry.task_key and after.token == entry.token
    assert store.remember(original) == after
    store.complete(after)
    assert not store.pending()
