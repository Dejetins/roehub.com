from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .scalar_env_overrides import resolve_bool_override, resolve_positive_int_override
from .strategy_runtime_config import StrategyRuntimeConfig, load_strategy_runtime_config

_STRATEGY_LIVE_WORKER_ENABLED_ENV_KEY = "ROEHUB_STRATEGY_LIVE_WORKER_ENABLED"
_STRATEGY_REALTIME_OUTPUT_ENABLED_ENV_KEY = (
    "ROEHUB_STRATEGY_REALTIME_OUTPUT_REDIS_STREAMS_ENABLED"
)
_STRATEGY_TELEGRAM_ENABLED_ENV_KEY = "ROEHUB_STRATEGY_TELEGRAM_ENABLED"
_STRATEGY_METRICS_PORT_ENV_KEY = "ROEHUB_STRATEGY_METRICS_PORT"


@dataclass(frozen=True, slots=True)
class StrategyLiveRunnerRepairConfig:
    """
    StrategyLiveRunnerRepairConfig — repair(read) retry policy for canonical ClickHouse lag.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
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
            Retry attempts and backoff are bounded positive values in Strategy live-runner v1.
        Raises:
            ValueError: If config values are out of allowed range.
        Side Effects:
            None.
        """
        if self.retry_attempts < 0:
            raise ValueError("strategy_live_runner.repair.retry_attempts must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError(
                "strategy_live_runner.repair.retry_backoff_seconds must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class StrategyLiveRunnerRedisConfig:
    """
    StrategyLiveRunnerRedisConfig — Redis Streams consumer config for Strategy live-runner.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md
      - src/trading/contexts/strategy/adapters/outbound/messaging/redis/
        redis_streams_live_candle_stream.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
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
        Validate Redis consumer configuration invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Stream prefix must remain `md.candles.1m` for v1 feed contract compatibility.
        Raises:
            ValueError: If one of values is invalid.
        Side Effects:
            None.
        """
        if not self.host.strip():
            raise ValueError("strategy_live_runner.redis_streams.host must be non-empty")
        if self.port <= 0:
            raise ValueError("strategy_live_runner.redis_streams.port must be > 0")
        if self.db < 0:
            raise ValueError("strategy_live_runner.redis_streams.db must be >= 0")
        if self.socket_timeout_s <= 0:
            raise ValueError("strategy_live_runner.redis_streams.socket_timeout_s must be > 0")
        if self.connect_timeout_s <= 0:
            raise ValueError("strategy_live_runner.redis_streams.connect_timeout_s must be > 0")
        if not self.stream_prefix.strip():
            raise ValueError(
                "strategy_live_runner.redis_streams.stream_prefix must be non-empty"
            )
        if not self.consumer_group.strip():
            raise ValueError(
                "strategy_live_runner.redis_streams.consumer_group must be non-empty"
            )
        if self.read_count <= 0:
            raise ValueError("strategy_live_runner.redis_streams.read_count must be > 0")
        if self.block_ms < 0:
            raise ValueError("strategy_live_runner.redis_streams.block_ms must be >= 0")


@dataclass(frozen=True, slots=True)
class StrategyLiveRunnerRealtimeOutputConfig:
    """
    StrategyLiveRunnerRealtimeOutputConfig — Redis Streams publish config for realtime output v1.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/messaging/redis/
        redis_streams_realtime_output_publisher.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy_live_runner.yaml
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
        Validate realtime output config invariants when feature is enabled.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Feature can be disabled without requiring Redis settings.
        Raises:
            ValueError: If enabled config values are invalid.
        Side Effects:
            None.
        """
        if not self.enabled:
            return
        if not self.host.strip():
            raise ValueError("strategy_live_runner.realtime_output.host must be non-empty")
        if self.port <= 0:
            raise ValueError("strategy_live_runner.realtime_output.port must be > 0")
        if self.db < 0:
            raise ValueError("strategy_live_runner.realtime_output.db must be >= 0")
        if self.socket_timeout_s <= 0:
            raise ValueError(
                "strategy_live_runner.realtime_output.socket_timeout_s must be > 0"
            )
        if self.connect_timeout_s <= 0:
            raise ValueError(
                "strategy_live_runner.realtime_output.connect_timeout_s must be > 0"
            )
        if not self.metrics_stream_prefix.strip():
            raise ValueError(
                "strategy_live_runner.realtime_output.metrics_stream_prefix must be non-empty"
            )
        if not self.events_stream_prefix.strip():
            raise ValueError(
                "strategy_live_runner.realtime_output.events_stream_prefix must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class StrategyLiveRunnerTelegramConfig:
    """
    StrategyLiveRunnerTelegramConfig — runtime config for Strategy Telegram notifier v1.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/telegram_notification_policy.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/prod/strategy_live_runner.yaml
    """

    enabled: bool
    mode: str
    bot_token_env: str | None
    api_base_url: str
    send_timeout_s: float
    debounce_failed_seconds: int

    def __post_init__(self) -> None:
        """
        Validate Telegram notifier runtime config invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `mode` is fixed to `log_only` or `telegram` in Strategy notifier v1.
        Raises:
            ValueError: If one of config values is invalid.
        Side Effects:
            None.
        """
        normalized_mode = self.mode.strip()
        if normalized_mode not in {"log_only", "telegram"}:
            raise ValueError(
                "strategy_live_runner.telegram.mode must be one of: log_only, telegram"
            )
        if self.send_timeout_s <= 0:
            raise ValueError("strategy_live_runner.telegram.send_timeout_s must be > 0")
        if self.debounce_failed_seconds < 0:
            raise ValueError(
                "strategy_live_runner.telegram.debounce_failed_seconds must be >= 0"
            )

        normalized_api_base = self.api_base_url.strip()
        if not normalized_api_base:
            raise ValueError("strategy_live_runner.telegram.api_base_url must be non-empty")

        normalized_bot_token_env = None
        if self.bot_token_env is not None:
            normalized_bot_token_env = self.bot_token_env.strip() or None

        if self.enabled and normalized_mode == "telegram" and normalized_bot_token_env is None:
            raise ValueError(
                "strategy_live_runner.telegram.bot_token_env is required for mode=telegram"
            )

        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "api_base_url", normalized_api_base.rstrip("/"))
        object.__setattr__(self, "bot_token_env", normalized_bot_token_env)


@dataclass(frozen=True, slots=True)
class StrategyLiveRunnerRuntimeConfig:
    """
    StrategyLiveRunnerRuntimeConfig — top-level runtime config for Strategy live-runner worker.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - apps/worker/strategy_live_runner/main/main.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy.yaml
      - configs/dev/strategy_live_runner.yaml
    """

    version: int
    live_worker_enabled: bool
    poll_interval_seconds: int
    redis_streams: StrategyLiveRunnerRedisConfig
    realtime_output: StrategyLiveRunnerRealtimeOutputConfig
    telegram: StrategyLiveRunnerTelegramConfig
    repair: StrategyLiveRunnerRepairConfig
    metrics_port: int

    def __post_init__(self) -> None:
        """
        Validate top-level runtime config invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Polling interval is positive and controls Postgres active-run refresh cadence.
        Raises:
            ValueError: If polling interval is invalid.
        Side Effects:
            None.
        """
        if self.poll_interval_seconds <= 0:
            raise ValueError("strategy_live_runner.poll_interval_seconds must be > 0")
        if self.metrics_port <= 0:
            raise ValueError("strategy.metrics.port must be > 0")


def load_strategy_live_runner_runtime_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> StrategyLiveRunnerRuntimeConfig:
    """
    Load Strategy live-runner runtime config from legacy payload, shim, or `strategy.yaml`.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - configs/dev/strategy_live_runner.yaml
      - configs/dev/strategy.yaml
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py

    Args:
        path: Path to `strategy_live_runner.yaml` or `strategy.yaml`.
        environ: Optional runtime environment mapping used by scalar overrides.
    Returns:
        StrategyLiveRunnerRuntimeConfig: Parsed runtime config for live worker wiring.
    Assumptions:
        Loader supports three formats in v1:
        - direct `strategy.yaml`,
        - shim `strategy_live_runner.config_ref.path`,
        - legacy full `strategy_live_runner` payload.
    Raises:
        FileNotFoundError: If config path (or referenced shim path) does not exist.
        ValueError: If YAML shape/values are invalid.
    Side Effects:
        Reads one or two config files from disk depending on shim mode.
    """
    effective_environ = os.environ if environ is None else environ
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"strategy live-runner config not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("strategy live-runner config must be mapping at top-level")

    if "strategy" in payload:
        return _build_live_runner_runtime_from_strategy_config(
            load_strategy_runtime_config(config_path, environ=effective_environ)
        )

    version = _get_int(payload, "version", required=True)
    runner_map = _get_mapping(payload, "strategy_live_runner", required=True)

    config_ref_map = _get_mapping(runner_map, "config_ref", required=False)
    config_ref_path = _get_optional_str_with_default(config_ref_map, "path", default=None)
    if config_ref_path is not None:
        resolved_ref_path = _resolve_config_ref_path(
            source_config_path=config_path,
            raw_ref_path=config_ref_path,
        )
        return _build_live_runner_runtime_from_strategy_config(
            load_strategy_runtime_config(resolved_ref_path, environ=effective_environ)
        )

    redis_map = _get_mapping(runner_map, "redis_streams", required=False)
    realtime_output_map = _get_mapping(runner_map, "realtime_output", required=False)
    telegram_map = _get_mapping(runner_map, "telegram", required=False)
    repair_map = _get_mapping(runner_map, "repair", required=False)

    realtime_output_enabled = resolve_bool_override(
        environ=effective_environ,
        key=_STRATEGY_REALTIME_OUTPUT_ENABLED_ENV_KEY,
        default=_get_bool_with_default(realtime_output_map, "enabled", default=False),
    )
    telegram_enabled = resolve_bool_override(
        environ=effective_environ,
        key=_STRATEGY_TELEGRAM_ENABLED_ENV_KEY,
        default=_get_bool_with_default(telegram_map, "enabled", default=False),
    )
    return StrategyLiveRunnerRuntimeConfig(
        version=version,
        live_worker_enabled=resolve_bool_override(
            environ=effective_environ,
            key=_STRATEGY_LIVE_WORKER_ENABLED_ENV_KEY,
            default=True,
        ),
        poll_interval_seconds=_get_int_with_default(
            runner_map,
            "poll_interval_seconds",
            default=5,
        ),
        redis_streams=StrategyLiveRunnerRedisConfig(
            enabled=_get_bool_with_default(redis_map, "enabled", default=True),
            host=_get_str_with_default(redis_map, "host", default="redis"),
            port=_get_int_with_default(redis_map, "port", default=6379),
            db=_get_int_with_default(redis_map, "db", default=0),
            password_env=_get_optional_str_with_default(
                redis_map,
                "password_env",
                default="ROEHUB_REDIS_PASSWORD",
            ),
            socket_timeout_s=_get_float_with_default(redis_map, "socket_timeout_s", default=2.0),
            connect_timeout_s=_get_float_with_default(
                redis_map,
                "connect_timeout_s",
                default=2.0,
            ),
            stream_prefix=_get_str_with_default(
                redis_map,
                "stream_prefix",
                default="md.candles.1m",
            ),
            consumer_group=_get_str_with_default(
                redis_map,
                "consumer_group",
                default="strategy.live_runner.v1",
            ),
            read_count=_get_int_with_default(redis_map, "read_count", default=200),
            block_ms=_get_int_with_default(redis_map, "block_ms", default=100),
        ),
        realtime_output=StrategyLiveRunnerRealtimeOutputConfig(
            enabled=realtime_output_enabled,
            host=_get_str_with_default(realtime_output_map, "host", default="redis"),
            port=_get_int_with_default(realtime_output_map, "port", default=6379),
            db=_get_int_with_default(realtime_output_map, "db", default=0),
            password_env=_get_optional_str_with_default(
                realtime_output_map,
                "password_env",
                default="ROEHUB_REDIS_PASSWORD",
            ),
            socket_timeout_s=_get_float_with_default(
                realtime_output_map,
                "socket_timeout_s",
                default=2.0,
            ),
            connect_timeout_s=_get_float_with_default(
                realtime_output_map,
                "connect_timeout_s",
                default=2.0,
            ),
            metrics_stream_prefix=_get_str_with_default(
                realtime_output_map,
                "metrics_stream_prefix",
                default="strategy.metrics.v1.user",
            ),
            events_stream_prefix=_get_str_with_default(
                realtime_output_map,
                "events_stream_prefix",
                default="strategy.events.v1.user",
            ),
        ),
        telegram=StrategyLiveRunnerTelegramConfig(
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
        repair=StrategyLiveRunnerRepairConfig(
            retry_attempts=_get_int_with_default(repair_map, "retry_attempts", default=3),
            retry_backoff_seconds=_get_float_with_default(
                repair_map,
                "retry_backoff_seconds",
                default=1.0,
            ),
        ),
        metrics_port=resolve_positive_int_override(
            environ=effective_environ,
            key=_STRATEGY_METRICS_PORT_ENV_KEY,
            default=9203,
        ),
    )


def _build_live_runner_runtime_from_strategy_config(
    strategy_config: StrategyRuntimeConfig,
) -> StrategyLiveRunnerRuntimeConfig:
    """
    Convert source-of-truth Strategy runtime config into live-runner legacy config shape.

    Args:
        strategy_config: Parsed `strategy.yaml` runtime config.
    Returns:
        StrategyLiveRunnerRuntimeConfig: Live worker runtime config consumed by worker wiring.
    Assumptions:
        Mapping is deterministic and field-by-field without lossy transformations.
    Raises:
        ValueError: Propagated from destination dataclass invariant checks.
    Side Effects:
        None.
    """
    live_worker_redis = strategy_config.live_worker.redis_streams
    realtime_output_redis = strategy_config.realtime_output.redis_streams
    telegram = strategy_config.telegram
    return StrategyLiveRunnerRuntimeConfig(
        version=strategy_config.version,
        live_worker_enabled=strategy_config.live_worker.enabled,
        poll_interval_seconds=strategy_config.live_worker.poll_interval_seconds,
        redis_streams=StrategyLiveRunnerRedisConfig(
            enabled=live_worker_redis.enabled,
            host=live_worker_redis.host,
            port=live_worker_redis.port,
            db=live_worker_redis.db,
            password_env=live_worker_redis.password_env,
            socket_timeout_s=live_worker_redis.socket_timeout_s,
            connect_timeout_s=live_worker_redis.connect_timeout_s,
            stream_prefix=live_worker_redis.stream_prefix,
            consumer_group=live_worker_redis.consumer_group,
            read_count=live_worker_redis.read_count,
            block_ms=live_worker_redis.block_ms,
        ),
        realtime_output=StrategyLiveRunnerRealtimeOutputConfig(
            enabled=realtime_output_redis.enabled,
            host=realtime_output_redis.host,
            port=realtime_output_redis.port,
            db=realtime_output_redis.db,
            password_env=realtime_output_redis.password_env,
            socket_timeout_s=realtime_output_redis.socket_timeout_s,
            connect_timeout_s=realtime_output_redis.connect_timeout_s,
            metrics_stream_prefix=realtime_output_redis.metrics_stream_prefix,
            events_stream_prefix=realtime_output_redis.events_stream_prefix,
        ),
        telegram=StrategyLiveRunnerTelegramConfig(
            enabled=telegram.enabled,
            mode=telegram.mode,
            bot_token_env=telegram.bot_token_env,
            api_base_url=telegram.api_base_url,
            send_timeout_s=telegram.send_timeout_s,
            debounce_failed_seconds=telegram.debounce_failed_seconds,
        ),
        repair=StrategyLiveRunnerRepairConfig(
            retry_attempts=strategy_config.live_worker.repair.retry_attempts,
            retry_backoff_seconds=strategy_config.live_worker.repair.retry_backoff_seconds,
        ),
        metrics_port=strategy_config.metrics.port,
    )


def _resolve_config_ref_path(*, source_config_path: Path, raw_ref_path: str) -> Path:
    """
    Resolve `strategy_live_runner.config_ref.path` from shim config.

    Args:
        source_config_path: Path to the shim file containing config reference.
        raw_ref_path: Raw `config_ref.path` value from YAML.
    Returns:
        Path: Resolved path to referenced `strategy.yaml`.
    Assumptions:
        Repo-relative references usually start with `configs/` in v1 shim files.
    Raises:
        ValueError: If reference path is blank after normalization.
    Side Effects:
        None.
    """
    normalized_ref_path = raw_ref_path.strip()
    if not normalized_ref_path:
        raise ValueError("strategy_live_runner.config_ref.path must be non-empty")

    candidate = Path(normalized_ref_path)
    if candidate.is_absolute():
        return candidate

    if candidate.parts and candidate.parts[0] == "configs":
        return candidate
    return source_config_path.parent / candidate


def _get_mapping(data: Mapping[str, Any], key: str, *, required: bool) -> Mapping[str, Any]:
    """
    Read nested mapping value from config payload.

    Args:
        data: Source mapping.
        key: Mapping key name.
        required: Whether key is required.
    Returns:
        Mapping[str, Any]: Nested mapping or empty mapping for optional missing key.
    Assumptions:
        Optional missing sections are represented as empty mapping.
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
        Bool values are rejected even though bool subclasses int.
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
        Bool values are rejected.
    Raises:
        ValueError: If present value is not integer.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    return _get_int(data, key, required=True)


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
        Empty strings are invalid for runtime config fields.
    Raises:
        ValueError: If present value is not non-empty string.
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
        str | None: Parsed string value or None.
    Assumptions:
        Empty strings are normalized to None.
    Raises:
        ValueError: If present value is not string or null.
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
