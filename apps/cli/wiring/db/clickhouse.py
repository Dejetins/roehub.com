from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    Но для server-side запуска (и ноутбуков) поддерживаем fallback из env-file:
    - /etc/roehub/roehub.env (как в docker-compose)
    Плюс поддерживаем алиасы CLICKHOUSE_* -> CH_* (user/password/db).

    Приоритет значений:
    1) os.environ (или переданный mapping)
    2) env-file (/etc/roehub/roehub.env), если существует
    3) дефолты
    """

    def __init__(self, environ: Mapping[str, str]) -> None:
        self._env = environ

    def load(self) -> ClickHouseSettings:
        file_env = _read_env_file(Path("/etc/roehub/roehub.env"))

        def pick(*keys: str, default: str) -> str:
            for k in keys:
                v = self._env.get(k)
                if v is not None and str(v).strip() != "":
                    return str(v)
            for k in keys:
                v = file_env.get(k)
                if v is not None and str(v).strip() != "":
                    return str(v)
            return default

        host = pick("CH_HOST", default="localhost")
        port = int(pick("CH_PORT", default="8123"))

        # алиасы: CH_* приоритетнее, потом CLICKHOUSE_*
        user = pick("CH_USER", "CLICKHOUSE_USER", default="default")
        password = pick("CH_PASSWORD", "CLICKHOUSE_PASSWORD", default="")

        # ВАЖНО: по умолчанию используем market_data (ваш DDL),
        # не берём CLICKHOUSE_DB автоматически, чтобы не уехать в 'roehub'.
        database = pick("CH_DATABASE", default="market_data")

        secure = _parse_bool01(pick("CH_SECURE", default="0"))
        verify = _parse_bool01(pick("CH_VERIFY", default="1"))

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
        raise RuntimeError("clickhouse-connect is required to run CLI commands") from e

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


def _read_env_file(path: Path) -> dict[str, str]:
    """
    Мини-парсер env-файла формата KEY=VALUE.
    - игнорирует пустые строки и строки-комментарии (# ...)
    - поддерживает простые кавычки вокруг значения
    """
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        val = v.strip()

        # снять обрамляющие кавычки, если есть
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]

        if key:
            out[key] = val
    return out
