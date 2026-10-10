from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Protocol

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseConnectGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.raw_kline_writer import (
    ClickHouseRawKlineWriter,
)

ClickHouseWorkload = Literal["stream", "interactive", "backfill", "artifacts"]


class RawKlineWriter(Protocol):
    def write_1m(self, rows) -> None: ...


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
    - path из ROEHUB_ENV_FILE (если задан)
    - /etc/roehub/roehub.env (как в docker-compose)
    Плюс поддерживаем алиасы CLICKHOUSE_* -> CH_* (user/password/db).

    Приоритет значений:
    1) os.environ (или переданный mapping)
    2) env-file (ROEHUB_ENV_FILE, затем /etc/roehub/roehub.env), если читается
    3) дефолты
    """

    def __init__(self, environ: Mapping[str, str]) -> None:
        self._env = environ

    def load(self, *, workload: ClickHouseWorkload | None = None) -> ClickHouseSettings:
        file_env: dict[str, str] = {}
        env_file = str(self._env.get("ROEHUB_ENV_FILE", "")).strip()
        if env_file:
            file_env.update(_read_env_file(Path(env_file)))
        file_env.update(_read_env_file(Path("/etc/roehub/roehub.env")))

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

        # The composition root selects the class; no HTTP/request input controls it.
        # Legacy mode supports ordered migration. Enabled installations mount only the
        # selected secret and must never fall back to the old shared/admin identity.
        required = pick("ROEHUB_CH_WORKLOADS_REQUIRED", default="0")
        if required not in {"0", "1"}:
            raise ValueError("invalid ClickHouse workload activation")
        if required == "1":
            if workload not in {"stream", "interactive", "backfill", "artifacts"}:
                raise ValueError("ClickHouse workload is required")
            user = "roehub_" + workload
            credential = pick(f"ROEHUB_CH_{workload.upper()}_PASSWORD_FILE", default="")
            if not credential:
                raise ValueError("ClickHouse workload credential is required")
            password = _read_optional_credential_file(credential)

        else:
            # алиасы: CH_* приоритетнее, потом CLICKHOUSE_*
            user = pick("CH_USER", "CLICKHOUSE_USER", default="default")
            password = (
                pick("CH_PASSWORD", "CLICKHOUSE_PASSWORD", default="")
                or _read_optional_credential_file(
                    pick(
                        "ROEHUB_CLICKHOUSE_PASSWORD_FILE",
                        "CH_PASSWORD_FILE",
                        "CLICKHOUSE_PASSWORD_FILE",
                        default="",
                    )
                )
            )

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
        settings = ClickHouseSettingsLoader(self.environ).load(workload="backfill")
        client = _clickhouse_client(settings)
        gateway = ClickHouseConnectGateway(client)
        return ClickHouseRawKlineWriter(gateway=gateway, database=settings.database)


def _clickhouse_client(settings: ClickHouseSettings):
    try:
        import clickhouse_connect  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("clickhouse-connect is required to run CLI commands") from e

    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.work_errors import (
        transient_clickhouse_error,
    )

    # Driver initialization performs read-only metadata queries. Retry only classified
    # admission pressure here, never replay an application INSERT/command implicitly.
    for attempt in range(6):
        try:
            return clickhouse_connect.get_client(
                host=settings.host,
                port=settings.port,
                username=settings.user,
                password=settings.password,
                secure=settings.secure,
                verify=settings.verify,
            )
        except Exception as error:
            if attempt == 5 or transient_clickhouse_error(error) not in {
                "storage_unavailable", "storage_memory_pressure",
            }:
                raise
            time.sleep(min(1.0, 0.1 * 2 ** attempt) + random.uniform(0, 0.05))
    raise AssertionError("unreachable client initialization")


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
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    for line in lines:
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


def _read_optional_credential_file(path_value: str) -> str:
    """Read a configured credential file without copying it into the environment."""
    if not path_value:
        return ""
    path = Path(path_value)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"ClickHouse credential file is unreadable: {path}") from error
    if not value:
        raise ValueError(f"ClickHouse credential file is empty: {path}")
    return value
