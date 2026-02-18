from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .scalar_env_overrides import resolve_bool_override, resolve_positive_int_override

_ENV_NAME_KEY = "ROEHUB_ENV"
_STRATEGY_CONFIG_PATH_KEY = "ROEHUB_STRATEGY_CONFIG"
_ALLOWED_ENVS = ("dev", "prod", "test")

_STRATEGY_API_ENABLED_ENV_KEY = "ROEHUB_STRATEGY_API_ENABLED"
_STRATEGY_LIVE_WORKER_ENABLED_ENV_KEY = "ROEHUB_STRATEGY_LIVE_WORKER_ENABLED"
_STRATEGY_REALTIME_OUTPUT_ENABLED_ENV_KEY = (
    "ROEHUB_STRATEGY_REALTIME_OUTPUT_REDIS_STREAMS_ENABLED"
)
_STRATEGY_TELEGRAM_ENABLED_ENV_KEY = "ROEHUB_STRATEGY_TELEGRAM_ENABLED"
_STRATEGY_METRICS_PORT_ENV_KEY = "ROEHUB_STRATEGY_METRICS_PORT"


@dataclass(frozen=True, slots=True)
class StrategyApiRuntimeConfig:
    """
    StrategyApiRuntimeConfig — Strategy API runtime toggle from `strategy.yaml`.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - apps/api/main/app.py
      - apps/api/wiring/modules/strategy.py
      - configs/dev/strategy.yaml
    """

    enabled: bool


@dataclass(frozen=True, slots=True)
class StrategyRepairRuntimeConfig:
    """
    StrategyRepairRuntimeConfig — repair(read) retry policy for canonical lag handling.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy.yaml
    """

    retry_attempts: int
    retry_backoff_seconds: float

    def __post_init__(self) -> None:
        """
        Validate repair retry configuration invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Retry attempts and retry backoff are bounded non-negative values.
        Raises:
            ValueError: If one of repair settings is negative.
        Side Effects:
            None.
        """
        if self.retry_attempts < 0:
            raise ValueError("strategy.live_worker.repair.retry_attempts must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError(
                "strategy.live_worker.repair.retry_backoff_seconds must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class StrategyRedisStreamsRuntimeConfig:
    """
    StrategyRedisStreamsRuntimeConfig — Redis Streams consumer config for Strategy live worker.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/messaging/redis/
        redis_streams_live_candle_stream.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy.yaml
    """

    enabled: bool
    host: str
    port: int
    db: int
    password_env: str | None
    socket_timeout_s: float
    connect_timeout_s: float
    stream_prefix: str
    consumer_group: str
    read_count: int
    block_ms: int

    def __post_init__(self) -> None:
        """
        Validate Redis Streams consumer configuration invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Stream prefix stays aligned with market-data Redis Streams feed contract.
        Raises:
            ValueError: If one of config values is invalid.
        Side Effects:
            None.
        """
        if not self.host.strip():
            raise ValueError("strategy.live_worker.redis_streams.host must be non-empty")
        if self.port <= 0:
            raise ValueError("strategy.live_worker.redis_streams.port must be > 0")
        if self.db < 0:
            raise ValueError("strategy.live_worker.redis_streams.db must be >= 0")
        if self.socket_timeout_s <= 0:
            raise ValueError(
                "strategy.live_worker.redis_streams.socket_timeout_s must be > 0"
            )
        if self.connect_timeout_s <= 0:
            raise ValueError(
                "strategy.live_worker.redis_streams.connect_timeout_s must be > 0"
            )
        if not self.stream_prefix.strip():
            raise ValueError(
                "strategy.live_worker.redis_streams.stream_prefix must be non-empty"
            )
        if not self.consumer_group.strip():
            raise ValueError(
                "strategy.live_worker.redis_streams.consumer_group must be non-empty"
            )
        if self.read_count <= 0:
            raise ValueError("strategy.live_worker.redis_streams.read_count must be > 0")
        if self.block_ms < 0:
            raise ValueError("strategy.live_worker.redis_streams.block_ms must be >= 0")


@dataclass(frozen=True, slots=True)
class StrategyLiveWorkerRuntimeConfig:
    """
    StrategyLiveWorkerRuntimeConfig — Strategy live worker runtime section in `strategy.yaml`.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - apps/worker/strategy_live_runner/main/main.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy.yaml
    """

    enabled: bool
    poll_interval_seconds: int
    redis_streams: StrategyRedisStreamsRuntimeConfig
    repair: StrategyRepairRuntimeConfig

    def __post_init__(self) -> None:
        """
        Validate live-worker runtime invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Poll interval is positive for deterministic active-run polling cadence.
        Raises:
            ValueError: If poll interval is not positive.
        Side Effects:
            None.
        """
        if self.poll_interval_seconds <= 0:
            raise ValueError("strategy.live_worker.poll_interval_seconds must be > 0")


@dataclass(frozen=True, slots=True)
class StrategyRealtimeOutputRedisStreamsRuntimeConfig:
    """
    StrategyRealtimeOutputRedisStreamsRuntimeConfig — Redis publish config for Strategy output.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/messaging/redis/
        redis_streams_realtime_output_publisher.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy.yaml
    """

    enabled: bool
    host: str
    port: int
    db: int
    password_env: str | None
    socket_timeout_s: float
    connect_timeout_s: float
    metrics_stream_prefix: str
    events_stream_prefix: str

    def __post_init__(self) -> None:
        """
        Validate realtime output Redis configuration when feature is enabled.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Disabled realtime output does not require Redis connection settings.
        Raises:
            ValueError: If enabled config values are invalid.
        Side Effects:
            None.
        """
        if not self.enabled:
            return
        if not self.host.strip():
            raise ValueError(
                "strategy.realtime_output.redis_streams.host must be non-empty"
            )
        if self.port <= 0:
            raise ValueError("strategy.realtime_output.redis_streams.port must be > 0")
        if self.db < 0:
            raise ValueError("strategy.realtime_output.redis_streams.db must be >= 0")
        if self.socket_timeout_s <= 0:
            raise ValueError(
                "strategy.realtime_output.redis_streams.socket_timeout_s must be > 0"
            )
        if self.connect_timeout_s <= 0:
            raise ValueError(
                "strategy.realtime_output.redis_streams.connect_timeout_s must be > 0"
            )
        if not self.metrics_stream_prefix.strip():
            raise ValueError(
                "strategy.realtime_output.redis_streams.metrics_stream_prefix must be "
                "non-empty"
            )
        if not self.events_stream_prefix.strip():
            raise ValueError(
                "strategy.realtime_output.redis_streams.events_stream_prefix must be "
                "non-empty"
            )


@dataclass(frozen=True, slots=True)
class StrategyRealtimeOutputRuntimeConfig:
    """
    StrategyRealtimeOutputRuntimeConfig — Strategy realtime output runtime section container.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy.yaml
    """

    redis_streams: StrategyRealtimeOutputRedisStreamsRuntimeConfig


@dataclass(frozen=True, slots=True)
class StrategyTelegramRuntimeConfig:
    """
    StrategyTelegramRuntimeConfig — Strategy Telegram notifier runtime section.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/telegram_notification_policy.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/prod/strategy.yaml
    """

    enabled: bool
    mode: str
    bot_token_env: str | None
    api_base_url: str
    send_timeout_s: float
    debounce_failed_seconds: int

    def __post_init__(self) -> None:
        """
        Validate Telegram notifier runtime configuration invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Supported notifier modes are fixed for Strategy runtime v1.
        Raises:
            ValueError: If one of Telegram settings is invalid.
        Side Effects:
            Normalizes mode/base URL/token-env values in place.
        """
        normalized_mode = self.mode.strip()
        if normalized_mode not in {"log_only", "telegram"}:
            raise ValueError(
                "strategy.telegram.mode must be one of: log_only, telegram"
            )
        if self.send_timeout_s <= 0:
            raise ValueError("strategy.telegram.send_timeout_s must be > 0")
        if self.debounce_failed_seconds < 0:
            raise ValueError("strategy.telegram.debounce_failed_seconds must be >= 0")

        normalized_api_base = self.api_base_url.strip()
        if not normalized_api_base:
            raise ValueError("strategy.telegram.api_base_url must be non-empty")

        normalized_bot_token_env = None
        if self.bot_token_env is not None:
            normalized_bot_token_env = self.bot_token_env.strip() or None

        if self.enabled and normalized_mode == "telegram" and normalized_bot_token_env is None:
            raise ValueError(
                "strategy.telegram.bot_token_env is required for mode=telegram"
            )

        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "api_base_url", normalized_api_base.rstrip("/"))
        object.__setattr__(self, "bot_token_env", normalized_bot_token_env)


@dataclass(frozen=True, slots=True)
class StrategyMetricsRuntimeConfig:
    """
    StrategyMetricsRuntimeConfig — Strategy metrics runtime settings.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - apps/worker/strategy_live_runner/main/main.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy.yaml
    """

    port: int

    def __post_init__(self) -> None:
        """
        Validate metrics runtime settings invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Metrics endpoint must bind to positive TCP port.
        Raises:
            ValueError: If metrics port is invalid.
        Side Effects:
            None.
        """
        if self.port <= 0:
            raise ValueError("strategy.metrics.port must be > 0")


@dataclass(frozen=True, slots=True)
class StrategyRuntimeConfig:
    """
    StrategyRuntimeConfig — source-of-truth Strategy runtime config (`strategy.yaml`).

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - configs/dev/strategy.yaml
      - apps/api/main/app.py
      - apps/worker/strategy_live_runner/main/main.py
    """

    version: int
    api: StrategyApiRuntimeConfig
    live_worker: StrategyLiveWorkerRuntimeConfig
    realtime_output: StrategyRealtimeOutputRuntimeConfig
    telegram: StrategyTelegramRuntimeConfig
    metrics: StrategyMetricsRuntimeConfig

    def __post_init__(self) -> None:
        """
        Validate top-level Strategy runtime config invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Runtime schema version is fixed to `1` in v1 contract.
        Raises:
            ValueError: If schema version is unsupported.
        Side Effects:
            None.
        """
        if self.version != 1:
            raise ValueError(f"strategy config version must be 1, got {self.version}")


def resolve_strategy_config_path(
    *,
    environ: Mapping[str, str],
    cli_config_path: str | Path | None = None,
) -> Path:
    """
    Resolve Strategy runtime config path using CLI/env/fallback precedence.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - apps/worker/strategy_live_runner/main/main.py
      - apps/api/main/app.py
      - src/trading/platform/config/indicators_compute_numba.py

    Args:
        environ: Runtime environment mapping.
        cli_config_path: Optional explicit CLI override path.
    Returns:
        Path: Resolved path to runtime config.
    Assumptions:
        Precedence is CLI `--config` > `ROEHUB_STRATEGY_CONFIG` > `configs/<env>/strategy.yaml`.
    Raises:
        ValueError: If `ROEHUB_ENV` value is invalid.
    Side Effects:
        None.
    """
    if cli_config_path is not None:
        raw_cli_path = str(cli_config_path).strip()
        if raw_cli_path:
            return Path(raw_cli_path)

    override_path = environ.get(_STRATEGY_CONFIG_PATH_KEY, "").strip()
    if override_path:
        return Path(override_path)

    env_name = _resolve_env_name(environ=environ)
    return Path("configs") / env_name / "strategy.yaml"


def load_strategy_runtime_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> StrategyRuntimeConfig:
    """
    Load and validate Strategy source-of-truth runtime YAML config.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - configs/dev/strategy.yaml
      - apps/api/main/app.py
      - apps/worker/strategy_live_runner/main/main.py

    Args:
        path: Path to `strategy.yaml`.
        environ: Optional runtime environment mapping used for scalar overrides.
    Returns:
        StrategyRuntimeConfig: Parsed and validated runtime config.
    Assumptions:
        YAML payload contains top-level `version` and `strategy` mapping.
    Raises:
        FileNotFoundError: If config path does not exist.
        ValueError: If YAML structure, values, or scalar env overrides are invalid.
    Side Effects:
        Reads one UTF-8 YAML file from filesystem.
    """
    effective_environ = os.environ if environ is None else environ
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"strategy config not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("strategy config must be mapping at top-level")

    version = _get_int(payload, "version", required=True)
    strategy_map = _get_mapping(payload, "strategy", required=True)

    api_map = _get_mapping(strategy_map, "api", required=False)
    live_worker_map = _get_mapping(strategy_map, "live_worker", required=False)
    live_worker_redis_map = _get_mapping(live_worker_map, "redis_streams", required=False)
    live_worker_repair_map = _get_mapping(live_worker_map, "repair", required=False)
    realtime_output_map = _get_mapping(strategy_map, "realtime_output", required=False)
    realtime_output_redis_map = _get_mapping(
        realtime_output_map,
        "redis_streams",
        required=False,
    )
    telegram_map = _get_mapping(strategy_map, "telegram", required=False)
    metrics_map = _get_mapping(strategy_map, "metrics", required=False)

    api_enabled = resolve_bool_override(
        environ=effective_environ,
        key=_STRATEGY_API_ENABLED_ENV_KEY,
        default=_get_bool_with_default(api_map, "enabled", default=True),
    )
    live_worker_enabled = resolve_bool_override(
        environ=effective_environ,
        key=_STRATEGY_LIVE_WORKER_ENABLED_ENV_KEY,
        default=_get_bool_with_default(live_worker_map, "enabled", default=True),
    )
    realtime_output_enabled = resolve_bool_override(
        environ=effective_environ,
        key=_STRATEGY_REALTIME_OUTPUT_ENABLED_ENV_KEY,
        default=_get_bool_with_default(realtime_output_redis_map, "enabled", default=False),
    )
    telegram_enabled = resolve_bool_override(
        environ=effective_environ,
        key=_STRATEGY_TELEGRAM_ENABLED_ENV_KEY,
        default=_get_bool_with_default(telegram_map, "enabled", default=False),
    )
    metrics_port = resolve_positive_int_override(
        environ=effective_environ,
        key=_STRATEGY_METRICS_PORT_ENV_KEY,
        default=_get_int_with_default(metrics_map, "port", default=9203),
    )

    return StrategyRuntimeConfig(
        version=version,
        api=StrategyApiRuntimeConfig(enabled=api_enabled),
        live_worker=StrategyLiveWorkerRuntimeConfig(
            enabled=live_worker_enabled,
            poll_interval_seconds=_get_int_with_default(
                live_worker_map,
                "poll_interval_seconds",
                default=5,
            ),
            redis_streams=StrategyRedisStreamsRuntimeConfig(
                enabled=_get_bool_with_default(live_worker_redis_map, "enabled", default=True),
                host=_get_str_with_default(live_worker_redis_map, "host", default="redis"),
                port=_get_int_with_default(live_worker_redis_map, "port", default=6379),
                db=_get_int_with_default(live_worker_redis_map, "db", default=0),
                password_env=_get_optional_str_with_default(
                    live_worker_redis_map,
                    "password_env",
                    default="ROEHUB_REDIS_PASSWORD",
                ),
                socket_timeout_s=_get_float_with_default(
                    live_worker_redis_map,
                    "socket_timeout_s",
                    default=2.0,
                ),
                connect_timeout_s=_get_float_with_default(
                    live_worker_redis_map,
                    "connect_timeout_s",
                    default=2.0,
                ),
                stream_prefix=_get_str_with_default(
                    live_worker_redis_map,
                    "stream_prefix",
                    default="md.candles.1m",
                ),
                consumer_group=_get_str_with_default(
                    live_worker_redis_map,
                    "consumer_group",
                    default="strategy.live_runner.v1",
                ),
                read_count=_get_int_with_default(live_worker_redis_map, "read_count", default=200),
                block_ms=_get_int_with_default(live_worker_redis_map, "block_ms", default=100),
            ),
            repair=StrategyRepairRuntimeConfig(
                retry_attempts=_get_int_with_default(
                    live_worker_repair_map,
                    "retry_attempts",
                    default=3,
                ),
                retry_backoff_seconds=_get_float_with_default(
                    live_worker_repair_map,
                    "retry_backoff_seconds",
                    default=1.0,
                ),
            ),
        ),
        realtime_output=StrategyRealtimeOutputRuntimeConfig(
            redis_streams=StrategyRealtimeOutputRedisStreamsRuntimeConfig(
                enabled=realtime_output_enabled,
                host=_get_str_with_default(realtime_output_redis_map, "host", default="redis"),
                port=_get_int_with_default(realtime_output_redis_map, "port", default=6379),
                db=_get_int_with_default(realtime_output_redis_map, "db", default=0),
                password_env=_get_optional_str_with_default(
                    realtime_output_redis_map,
                    "password_env",
                    default="ROEHUB_REDIS_PASSWORD",
                ),
                socket_timeout_s=_get_float_with_default(
                    realtime_output_redis_map,
                    "socket_timeout_s",
                    default=2.0,
                ),
                connect_timeout_s=_get_float_with_default(
                    realtime_output_redis_map,
                    "connect_timeout_s",
                    default=2.0,
                ),
                metrics_stream_prefix=_get_str_with_default(
                    realtime_output_redis_map,
                    "metrics_stream_prefix",
                    default="strategy.metrics.v1.user",
                ),
                events_stream_prefix=_get_str_with_default(
                    realtime_output_redis_map,
                    "events_stream_prefix",
                    default="strategy.events.v1.user",
                ),
            )
        ),
        telegram=StrategyTelegramRuntimeConfig(
            enabled=telegram_enabled,
            mode=_get_str_with_default(telegram_map, "mode", default="log_only"),
            bot_token_env=_get_optional_str_with_default(
                telegram_map,
                "bot_token_env",
                default="TELEGRAM_BOT_TOKEN",
            ),
            api_base_url=_get_str_with_default(
                telegram_map,
                "api_base_url",
                default="https://api.telegram.org",
            ),
            send_timeout_s=_get_float_with_default(
                telegram_map,
                "send_timeout_s",
                default=3.0,
            ),
            debounce_failed_seconds=_get_int_with_default(
                telegram_map,
                "debounce_failed_seconds",
                default=600,
            ),
        ),
        metrics=StrategyMetricsRuntimeConfig(port=metrics_port),
    )


def _resolve_env_name(*, environ: Mapping[str, str]) -> str:
    """
    Resolve normalized runtime environment name for strategy config fallback path.

    Args:
        environ: Runtime environment mapping.
    Returns:
        str: One of `dev`, `prod`, or `test`.
    Assumptions:
        Missing `ROEHUB_ENV` defaults to `dev`.
    Raises:
        ValueError: If value is outside allowed environment literals.
    Side Effects:
        None.
    """
    raw_env_name = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if raw_env_name not in _ALLOWED_ENVS:
        raise ValueError(
            f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {raw_env_name!r}"
        )
    return raw_env_name


def _get_mapping(data: Mapping[str, Any], key: str, *, required: bool) -> Mapping[str, Any]:
    """
    Read nested mapping value from config payload.

    Args:
        data: Source mapping.
        key: Nested mapping key name.
        required: Whether key must be present.
    Returns:
        Mapping[str, Any]: Nested mapping value or empty mapping.
    Assumptions:
        Optional missing nested sections are represented as empty mapping.
    Raises:
        ValueError: If required key is missing or value is not mapping.
    Side Effects:
        None.
    """
    value = data.get(key)
    if value is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"expected mapping at key '{key}', got {type(value).__name__}")
    return value


def _get_int(data: Mapping[str, Any], key: str, *, required: bool) -> int:
    """
    Read integer config value with bool rejection.

    Args:
        data: Source mapping.
        key: Integer key name.
        required: Whether key is required.
    Returns:
        int: Parsed integer value.
    Assumptions:
        Boolean values are rejected even though bool is an int subclass.
    Raises:
        ValueError: If required key missing or value type is invalid.
    Side Effects:
        None.
    """
    value = data.get(key)
    if value is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected int at key '{key}', got {type(value).__name__}")
    return value


def _get_int_with_default(data: Mapping[str, Any], key: str, *, default: int) -> int:
    """
    Read optional integer config value with explicit default.

    Args:
        data: Source mapping.
        key: Integer key name.
        default: Value used when key is absent.
    Returns:
        int: Parsed integer value.
    Assumptions:
        Bool values are rejected by `_get_int`.
    Raises:
        ValueError: If present value is not an integer.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    return _get_int(data, key, required=True)


def _get_float_with_default(data: Mapping[str, Any], key: str, *, default: float) -> float:
    """
    Read optional float config value with explicit default.

    Args:
        data: Source mapping.
        key: Float key name.
        default: Value used when key is absent.
    Returns:
        float: Parsed float value.
    Assumptions:
        Integer values are accepted and converted to float.
    Raises:
        ValueError: If present value is not numeric.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected float at key '{key}', got {type(value).__name__}")
    return float(value)


def _get_str_with_default(data: Mapping[str, Any], key: str, *, default: str) -> str:
    """
    Read optional non-empty string config value with explicit default.

    Args:
        data: Source mapping.
        key: String key name.
        default: Value used when key is absent.
    Returns:
        str: Parsed non-empty string value.
    Assumptions:
        Empty strings are invalid.
    Raises:
        ValueError: If present value is not a non-empty string.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"expected string at key '{key}', got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"key '{key}' must be non-empty")
    return normalized


def _get_optional_str_with_default(
    data: Mapping[str, Any],
    key: str,
    *,
    default: str | None,
) -> str | None:
    """
    Read optional nullable string config value with explicit default.

    Args:
        data: Source mapping.
        key: Nullable string key name.
        default: Value used when key is absent.
    Returns:
        str | None: Parsed value or None.
    Assumptions:
        Empty strings are normalized to None.
    Raises:
        ValueError: If present value is neither string nor null.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    value = data[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected string or null at key '{key}', got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _get_bool_with_default(data: Mapping[str, Any], key: str, *, default: bool) -> bool:
    """
    Read optional boolean config value with explicit default.

    Args:
        data: Source mapping.
        key: Boolean key name.
        default: Value used when key is absent.
    Returns:
        bool: Parsed boolean value.
    Assumptions:
        Only actual bool values are accepted.
    Raises:
        ValueError: If present value is not boolean.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"expected bool at key '{key}', got {type(value).__name__}")
    return value


__all__ = [
    "StrategyApiRuntimeConfig",
    "StrategyLiveWorkerRuntimeConfig",
    "StrategyMetricsRuntimeConfig",
    "StrategyRealtimeOutputRedisStreamsRuntimeConfig",
    "StrategyRealtimeOutputRuntimeConfig",
    "StrategyRedisStreamsRuntimeConfig",
    "StrategyRepairRuntimeConfig",
    "StrategyRuntimeConfig",
    "StrategyTelegramRuntimeConfig",
    "load_strategy_runtime_config",
    "resolve_strategy_config_path",
]
