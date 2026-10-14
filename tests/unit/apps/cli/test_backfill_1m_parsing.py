from __future__ import annotations

from datetime import datetime, timezone

import pytest

from apps.cli.commands.backfill_1m import IsoUtcTimestampParser, OptionalIntBatchSizeParser
from apps.cli.wiring.db.postgres import close_market_gateways, market_gateway


def test_parse_utc_iso_z_accepts_z_and_is_tz_aware() -> None:
    parser = IsoUtcTimestampParser()
    ts = parser.parse("2026-02-01T00:00:00Z")
    assert ts.value == datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)


def test_parse_utc_iso_z_rejects_naive() -> None:
    parser = IsoUtcTimestampParser()
    with pytest.raises(ValueError):
        parser.parse("2026-02-01T00:00:00")


def test_parse_batch_size_default_none() -> None:
    parser = OptionalIntBatchSizeParser()
    assert parser.parse(None) is None
    assert parser.parse("None") is None
    assert parser.parse(" none ") is None


def test_parse_batch_size_positive_int() -> None:
    parser = OptionalIntBatchSizeParser()
    assert parser.parse("10") == 10


def test_parse_batch_size_rejects_non_positive() -> None:
    parser = OptionalIntBatchSizeParser()
    with pytest.raises(ValueError):
        parser.parse("0")
    with pytest.raises(ValueError):
        parser.parse("-1")


def test_postgres_gateway_requires_explicit_dsn():
    with pytest.raises(ValueError, match="STRATEGY_PG_DSN"):
        market_gateway({})


def test_postgres_gateway_reuses_workload_and_rejects_unknown_class():
    env = {"STRATEGY_PG_DSN": "host=127.0.0.1 port=1 dbname=fixture user=fixture"}
    try:
        assert market_gateway(env, workload="stream") is market_gateway(env, workload="ingestion")
        assert market_gateway(env, workload="bulk") is not market_gateway(env, workload="stream")
        with pytest.raises(ValueError, match="workload"):
            market_gateway(env, workload="unknown")
    finally:
        close_market_gateways()
