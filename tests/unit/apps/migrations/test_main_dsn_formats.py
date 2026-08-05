from __future__ import annotations

from typing import Any

from sqlalchemy.engine import URL

import apps.migrations.main as migrations_main
from alembic.config import Config


def test_main_accepts_conninfo_dsn_with_raw_special_password(
    monkeypatch: Any,
) -> None:
    """
    Verify migration entrypoint accepts conninfo DSN with raw special password characters.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Upgrade flow can be stubbed to avoid real database connections in unit tests.
    Raises:
        AssertionError: If runner rejects conninfo DSN or parses password incorrectly.
    Side Effects:
        Monkeypatches upgrade function for test isolation.
    """
    captured: dict[str, object] = {}
    dsn = "host=postgres port=5432 dbname=roehub user=roehub password=M@ngala:100%"

    def _fake_upgrade_head_under_lock(
        *,
        config: Config,
        sqlalchemy_url: URL,
        lock_key: int,
    ) -> None:
        captured["config"] = config
        captured["sqlalchemy_url"] = sqlalchemy_url
        captured["lock_key"] = lock_key

    monkeypatch.setattr(migrations_main, "_upgrade_head_under_lock", _fake_upgrade_head_under_lock)

    exit_code = migrations_main.main(["--dsn", dsn])

    assert exit_code == 0
    parsed_url = captured["sqlalchemy_url"]
    assert isinstance(parsed_url, URL)
    assert parsed_url.username == "roehub"
    assert parsed_url.password == "M@ngala:100%"
    assert parsed_url.host == "postgres"
    assert parsed_url.port == 5432
    assert parsed_url.database == "roehub"


def test_main_accepts_percent_encoded_url_password_without_interpolation_failure(
    monkeypatch: Any,
) -> None:
    """
    Verify URL DSN with `%40` password encoding works without ConfigParser interpolation errors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Alembic config can use injected connection path without setting runtime sqlalchemy.url.
    Raises:
        AssertionError: If runner fails or stores percent-encoded runtime DSN in config.
    Side Effects:
        Monkeypatches upgrade function for test isolation.
    """
    captured: dict[str, object] = {}
    dsn = "postgresql://roehub:M%40ngala1906915@postgres:5432/roehub"

    def _fake_upgrade_head_under_lock(
        *,
        config: Config,
        sqlalchemy_url: URL,
        lock_key: int,
    ) -> None:
        captured["config"] = config
        captured["sqlalchemy_url"] = sqlalchemy_url
        captured["lock_key"] = lock_key

    monkeypatch.setattr(migrations_main, "_upgrade_head_under_lock", _fake_upgrade_head_under_lock)

    exit_code = migrations_main.main(["--dsn", dsn])

    assert exit_code == 0
    parsed_url = captured["sqlalchemy_url"]
    assert isinstance(parsed_url, URL)
    assert parsed_url.password == "M@ngala1906915"
    captured_config = captured["config"]
    assert isinstance(captured_config, Config)
    configured_url = captured_config.get_main_option("sqlalchemy.url") or ""
    assert "%40" not in configured_url
