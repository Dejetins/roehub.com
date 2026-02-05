from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseConnectGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.raw_kline_writer import (
    ClickHouseRawKlineWriter,
)


class RawKlineWriter(Protocol):
    def write_1m(self, rows) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ClickHouseSettings:
    host: str
    port: int
    user: str
    password: str
    database: str
    secure: bool
    verify: bool

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("CH_HOST must be non-empty")
        if self.port <= 0:
            raise ValueError("CH_PORT must be > 0")
        if not self.user.strip():
            raise ValueError("CH_USER must be non-empty")
        if not self.database.strip():
            raise ValueError("CH_DATABASE must be non-empty")


class ClickHouseSettingsLoader:
    """
    Env — источник правды.
    """

    def __init__(self, environ: Mapping[str, str]) -> None:
        self._env = environ

    def load(self) -> ClickHouseSettings:
        host = self._env.get("CH_HOST", "localhost")
        port = int(self._env.get("CH_PORT", "8123"))
        user = self._env.get("CH_USER", "default")
        password = self._env.get("CH_PASSWORD", "")
        database = self._env.get("CH_DATABASE", "market_data")
        secure = _parse_bool01(self._env.get("CH_SECURE", "0"))
        verify = _parse_bool01(self._env.get("CH_VERIFY", "1"))

        return ClickHouseSettings(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            secure=secure,
            verify=verify,
        )


@dataclass(frozen=True, slots=True)
class ClickHouseWriterFactory:
    environ: Mapping[str, str]

    def writer(self) -> RawKlineWriter:
        settings = ClickHouseSettingsLoader(self.environ).load()
        client = _clickhouse_client(settings)
        gateway = ClickHouseConnectGateway(client)
        return ClickHouseRawKlineWriter(gateway=gateway, database=settings.database)


def _clickhouse_client(settings: ClickHouseSettings):
    try:
        import clickhouse_connect  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("clickhouse-connect is required to run CLI backfill-1m") from e

    return clickhouse_connect.get_client(
        host=settings.host,
        port=settings.port,
        username=settings.user,
        password=settings.password,
        secure=settings.secure,
        verify=settings.verify,
    )


def _parse_bool01(value: str) -> bool:
    text = value.strip()
    if text == "1":
        return True
    if text == "0":
        return False
    raise ValueError("Expected '0' or '1' for boolean env var")
