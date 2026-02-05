from __future__ import annotations

from datetime import datetime, timezone

import pytest

from apps.cli.commands.backfill_1m import IsoUtcTimestampParser, OptionalIntBatchSizeParser
from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader


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


def test_clickhouse_env_loader_defaults() -> None:
    env = {}
    s = ClickHouseSettingsLoader(env).load()
    assert s.host == "localhost"
    assert s.port == 8123
    assert s.user == "default"
    assert s.password == ""
    assert s.database == "market_data"
    assert s.secure is False
    assert s.verify is True


def test_clickhouse_env_loader_parses_values() -> None:
    env = {
        "CH_HOST": "ch",
        "CH_PORT": "8443",
        "CH_USER": "u",
        "CH_PASSWORD": "p",
        "CH_DATABASE": "market_data",
        "CH_SECURE": "1",
        "CH_VERIFY": "0",
    }
    s = ClickHouseSettingsLoader(env).load()
    assert s.host == "ch"
    assert s.port == 8443
    assert s.user == "u"
    assert s.password == "p"
    assert s.secure is True
    assert s.verify is False
