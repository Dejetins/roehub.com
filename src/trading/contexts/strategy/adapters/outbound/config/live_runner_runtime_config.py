from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


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
class StrategyLiveRunnerRuntimeConfig:
    """
    StrategyLiveRunnerRuntimeConfig — top-level runtime config for Strategy live-runner worker.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - apps/worker/strategy_live_runner/main/main.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy_live_runner.yaml
    """

    version: int
    poll_interval_seconds: int
    redis_streams: StrategyLiveRunnerRedisConfig
    repair: StrategyLiveRunnerRepairConfig

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


def load_strategy_live_runner_runtime_config(path: str | Path) -> StrategyLiveRunnerRuntimeConfig:
    """
    Load and validate Strategy live-runner runtime YAML config.

    Args:
        path: Path to `strategy_live_runner.yaml`.
    Returns:
        StrategyLiveRunnerRuntimeConfig: Parsed runtime config.
    Assumptions:
        YAML has top-level `version` and `strategy_live_runner` mapping.
    Raises:
        FileNotFoundError: If config path does not exist.
        ValueError: If YAML shape/values are invalid.
    Side Effects:
        Reads one config file from disk.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"strategy live-runner config not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("strategy live-runner config must be mapping at top-level")

    version = _get_int(payload, "version", required=True)
    runner_map = _get_mapping(payload, "strategy_live_runner", required=True)
    redis_map = _get_mapping(runner_map, "redis_streams", required=False)
    repair_map = _get_mapping(runner_map, "repair", required=False)

    return StrategyLiveRunnerRuntimeConfig(
        version=version,
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
        repair=StrategyLiveRunnerRepairConfig(
            retry_attempts=_get_int_with_default(repair_map, "retry_attempts", default=3),
            retry_backoff_seconds=_get_float_with_default(
                repair_map,
                "retry_backoff_seconds",
                default=1.0,
            ),
        ),
    )


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
