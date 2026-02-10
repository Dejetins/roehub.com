from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from trading.shared_kernel.primitives.market_id import MarketId
from trading.shared_kernel.primitives.utc_timestamp import UtcTimestamp

_ALLOWED_EXCHANGES = {"binance", "bybit"}
_ALLOWED_MARKET_TYPES = {"spot", "futures"}


@dataclass(frozen=True, slots=True)
class BackoffConfig:
    base_s: float
    max_s: float
    jitter_s: float

    def __post_init__(self) -> None:
        _require_positive("backoff.base_s", self.base_s)
        _require_positive("backoff.max_s", self.max_s)
        _require_non_negative("backoff.jitter_s", self.jitter_s)
        if self.base_s > self.max_s:
            raise ValueError(f"backoff.base_s must be <= backoff.max_s, got {self.base_s} > {self.max_s}") # noqa: E501


@dataclass(frozen=True, slots=True)
class RateLimiterConfig:
    mode: str
    safety_factor: float
    max_concurrency: int

    def __post_init__(self) -> None:
        if self.mode != "autodetect":
            raise ValueError(f"rest.limiter.mode must be 'autodetect', got {self.mode!r}")
        if not (0.0 < self.safety_factor <= 1.0):
            raise ValueError(f"rest.limiter.safety_factor must be in (0, 1], got {self.safety_factor}") # noqa: E501
        _require_positive_int("rest.limiter.max_concurrency", self.max_concurrency)


@dataclass(frozen=True, slots=True)
class RestConfig:
    base_url: str
    timeout_s: float
    retries: int
    earliest_available_ts_utc: UtcTimestamp
    backoff: BackoffConfig
    limiter: RateLimiterConfig

    def __post_init__(self) -> None:
        _require_non_empty("rest.base_url", self.base_url)
        _require_positive("rest.timeout_s", self.timeout_s)
        _require_non_negative_int("rest.retries", self.retries)


@dataclass(frozen=True, slots=True)
class WsReconnectConfig:
    min_delay_s: float
    max_delay_s: float
    factor: float
    jitter_s: float

    def __post_init__(self) -> None:
        _require_positive("ws.reconnect.min_delay_s", self.min_delay_s)
        _require_positive("ws.reconnect.max_delay_s", self.max_delay_s)
        _require_positive("ws.reconnect.factor", self.factor)
        _require_non_negative("ws.reconnect.jitter_s", self.jitter_s)
        if self.min_delay_s > self.max_delay_s:
            raise ValueError(
                f"ws.reconnect.min_delay_s must be <= ws.reconnect.max_delay_s, got {self.min_delay_s} > {self.max_delay_s}" # noqa: E501
            )


@dataclass(frozen=True, slots=True)
class WsConfig:
    url: str
    ping_interval_s: float
    pong_timeout_s: float
    reconnect: WsReconnectConfig
    max_symbols_per_connection: int

    def __post_init__(self) -> None:
        _require_non_empty("ws.url", self.url)
        _require_positive("ws.ping_interval_s", self.ping_interval_s)
        _require_positive("ws.pong_timeout_s", self.pong_timeout_s)
        _require_positive_int("ws.max_symbols_per_connection", self.max_symbols_per_connection)


@dataclass(frozen=True, slots=True)
class MarketConfig:
    market_id: MarketId
    exchange: str
    market_type: str
    market_code: str
    rest: RestConfig
    ws: WsConfig

    def __post_init__(self) -> None:
        if self.exchange not in _ALLOWED_EXCHANGES:
            raise ValueError(f"exchange must be one of {_ALLOWED_EXCHANGES}, got {self.exchange!r}")
        if self.market_type not in _ALLOWED_MARKET_TYPES:
            raise ValueError(f"market_type must be one of {_ALLOWED_MARKET_TYPES}, got {self.market_type!r}") # noqa: E501
        _require_non_empty("market_code", self.market_code)


@dataclass(frozen=True, slots=True)
class RawWriteConfig:
    flush_interval_ms: int
    max_buffer_rows: int

    def __post_init__(self) -> None:
        """
        Validate raw-write buffering policy.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - `flush_interval_ms` must stay at or below 500ms to satisfy WS-to-raw SLO target.
        - `max_buffer_rows` must be a positive integer.

        Errors/Exceptions:
        - Raises `ValueError` when constraints are violated.

        Side effects:
        - None.
        """
        _require_positive_int("ingestion.raw_write.flush_interval_ms", self.flush_interval_ms)
        _require_positive_int("ingestion.raw_write.max_buffer_rows", self.max_buffer_rows)
        if self.flush_interval_ms > 500:
            raise ValueError(
                "ingestion.raw_write.flush_interval_ms must be <= 500, "
                f"got {self.flush_interval_ms}"
            )


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    raw_write: RawWriteConfig
    rest_concurrency_instruments: int
    tail_lookback_minutes: int

    def __post_init__(self) -> None:
        """
        Validate ingestion runtime controls used by worker and scheduler.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - REST instrument-level concurrency must be positive.
        - Tail lookback in minutes must be positive.

        Errors/Exceptions:
        - Raises `ValueError` when constraints are violated.

        Side effects:
        - None.
        """
        _require_positive_int(
            "ingestion.rest_concurrency_instruments",
            self.rest_concurrency_instruments,
        )
        _require_positive_int("ingestion.tail_lookback_minutes", self.tail_lookback_minutes)


@dataclass(frozen=True, slots=True)
class SchedulerJobConfig:
    interval_seconds: int

    def __post_init__(self) -> None:
        """
        Validate one scheduler job interval.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Interval is always positive.

        Errors/Exceptions:
        - Raises `ValueError` when interval is not positive.

        Side effects:
        - None.
        """
        _require_positive_int("scheduler.jobs.*.interval_seconds", self.interval_seconds)


@dataclass(frozen=True, slots=True)
class SchedulerJobsConfig:
    sync_whitelist: SchedulerJobConfig
    enrich: SchedulerJobConfig
    rest_insurance_catchup: SchedulerJobConfig


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    jobs: SchedulerJobsConfig


@dataclass(frozen=True, slots=True)
class BackfillConfig:
    max_days_per_insert: int
    chunk_align: str

    def __post_init__(self) -> None:
        _require_positive_int("backfill.max_days_per_insert", self.max_days_per_insert)
        if self.max_days_per_insert > 7:
            raise ValueError(f"backfill.max_days_per_insert must be <= 7, got {self.max_days_per_insert}")  # noqa: E501
        if self.chunk_align != "utc_day":
            raise ValueError(f"backfill.chunk_align must be 'utc_day', got {self.chunk_align!r}")


@dataclass(frozen=True, slots=True)
class RedisStreamsConfig:
    enabled: bool
    host: str
    port: int
    db: int
    password_env: str | None
    socket_timeout_s: float
    connect_timeout_s: float
    stream_mode: str
    stream_prefix: str
    retention_days: int
    maxlen_approx: int | None

    def __post_init__(self) -> None:
        """
        Validate runtime Redis Streams configuration for WS live feed.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - `stream_mode` is fixed to `per_instrument`.
        - `maxlen_approx` defaults to `retention_days * 1440` when omitted.

        Errors/Exceptions:
        - Raises `ValueError` on invalid host/port/timeouts/mode/retention values.

        Side effects:
        - May normalize `maxlen_approx` in frozen dataclass via `object.__setattr__`.
        """
        _require_non_empty("live_feed.redis_streams.host", self.host)
        _require_positive_int("live_feed.redis_streams.port", self.port)
        _require_non_negative_int("live_feed.redis_streams.db", self.db)
        _require_positive(
            "live_feed.redis_streams.socket_timeout_s",
            self.socket_timeout_s,
        )
        _require_positive(
            "live_feed.redis_streams.connect_timeout_s",
            self.connect_timeout_s,
        )
        if self.stream_mode != "per_instrument":
            raise ValueError(
                "live_feed.redis_streams.stream_mode must be 'per_instrument', "
                f"got {self.stream_mode!r}"
            )
        _require_non_empty("live_feed.redis_streams.stream_prefix", self.stream_prefix)
        _require_positive_int("live_feed.redis_streams.retention_days", self.retention_days)
        if self.maxlen_approx is None:
            object.__setattr__(self, "maxlen_approx", self.retention_days * 1440)
            return
        _require_positive_int("live_feed.redis_streams.maxlen_approx", self.maxlen_approx)


@dataclass(frozen=True, slots=True)
class LiveFeedConfig:
    redis_streams: RedisStreamsConfig

    def __post_init__(self) -> None:
        """
        Validate live feed config container.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - `redis_streams` is always present to avoid optional checks in wiring.

        Errors/Exceptions:
        - Raises `ValueError` when redis_streams config is not provided.

        Side effects:
        - None.
        """
        if self.redis_streams is None:  # type: ignore[truthy-bool]
            raise ValueError("live_feed.redis_streams must be provided")


@dataclass(frozen=True, slots=True)
class MarketDataRuntimeConfig:
    version: int
    markets: tuple[MarketConfig, ...]
    ingestion: IngestionConfig
    scheduler: SchedulerConfig
    backfill: BackfillConfig
    live_feed: LiveFeedConfig

    def market_by_id(self, market_id: MarketId) -> MarketConfig:
        for m in self.markets:
            if m.market_id.value == market_id.value:
                return m
        raise KeyError(f"market_id not found in config: {market_id.value}")

    def market_ids(self) -> tuple[int, ...]:
        return tuple(m.market_id.value for m in self.markets)

    @property
    def raw_write(self) -> RawWriteConfig:
        """
        Provide backward-compatible access to raw write settings.

        Parameters:
        - None.

        Returns:
        - `RawWriteConfig` from ingestion section.

        Assumptions/Invariants:
        - `self.ingestion` is validated during config load.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        return self.ingestion.raw_write


def load_market_data_runtime_config(path: str | Path) -> MarketDataRuntimeConfig:
    """
    Load and validate market-data runtime YAML.

    Parameters:
    - path: filesystem path to `market_data.yaml`.

    Returns:
    - Parsed and validated `MarketDataRuntimeConfig`.

    Assumptions/Invariants:
    - YAML top-level is a mapping with keys `version` and `market_data`.
    - Market ids are unique within `market_data.markets`.

    Errors/Exceptions:
    - Raises `FileNotFoundError` if config path does not exist.
    - Raises `ValueError` on schema/validation failures.

    Side effects:
    - Reads one file from disk.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"market_data config not found: {p}")

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("market_data config must be a YAML mapping at top-level")

    version = _get_int(data, "version", required=True)
    md = _get_mapping(data, "market_data", required=True)

    markets_raw = _get_list(md, "markets", required=True)
    markets = tuple(_parse_market(m) for m in markets_raw)

    # uniqueness of market_id
    ids = [m.market_id.value for m in markets]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate market_id in config: {ids}")

    ingestion = _get_mapping(md, "ingestion", required=True)
    raw_write_map = _get_mapping(ingestion, "raw_write", required=False)
    if raw_write_map:
        raw_write = RawWriteConfig(
            flush_interval_ms=_get_int(raw_write_map, "flush_interval_ms", required=True),
            max_buffer_rows=_get_int(raw_write_map, "max_buffer_rows", required=True),
        )
    else:
        raw_write = RawWriteConfig(
            flush_interval_ms=_get_int(ingestion, "flush_interval_ms", required=True),
            max_buffer_rows=_get_int(ingestion, "max_buffer_rows", required=True),
        )
    ingestion_cfg = IngestionConfig(
        raw_write=raw_write,
        rest_concurrency_instruments=_get_int_with_default(
            ingestion,
            "rest_concurrency_instruments",
            default=4,
        ),
        tail_lookback_minutes=_get_int_with_default(
            ingestion,
            "tail_lookback_minutes",
            default=180,
        ),
    )

    scheduler_map = _get_mapping(md, "scheduler", required=False)
    scheduler_jobs = _get_mapping(scheduler_map, "jobs", required=False)
    scheduler_cfg = SchedulerConfig(
        jobs=SchedulerJobsConfig(
            sync_whitelist=SchedulerJobConfig(
                interval_seconds=_get_int_nested_with_default(
                    scheduler_jobs,
                    section="sync_whitelist",
                    key="interval_seconds",
                    default=3600,
                )
            ),
            enrich=SchedulerJobConfig(
                interval_seconds=_get_int_nested_with_default(
                    scheduler_jobs,
                    section="enrich",
                    key="interval_seconds",
                    default=21600,
                )
            ),
            rest_insurance_catchup=SchedulerJobConfig(
                interval_seconds=_get_int_nested_with_default(
                    scheduler_jobs,
                    section="rest_insurance_catchup",
                    key="interval_seconds",
                    default=3600,
                )
            ),
        )
    )

    backfill_map = _get_mapping(md, "backfill", required=True)
    backfill = BackfillConfig(
        max_days_per_insert=_get_int(backfill_map, "max_days_per_insert", required=True),
        chunk_align=_get_str(backfill_map, "chunk_align", required=True),
    )
    live_feed_map = _get_mapping(md, "live_feed", required=False)
    redis_streams_map = _get_mapping(live_feed_map, "redis_streams", required=False)
    live_feed = LiveFeedConfig(
        redis_streams=RedisStreamsConfig(
            enabled=_get_bool_with_default(redis_streams_map, "enabled", default=False),
            host=_get_str_with_default(redis_streams_map, "host", default="redis"),
            port=_get_int_with_default(redis_streams_map, "port", default=6379),
            db=_get_int_with_default(redis_streams_map, "db", default=0),
            password_env=_get_optional_str_with_default(
                redis_streams_map,
                "password_env",
                default="ROEHUB_REDIS_PASSWORD",
            ),
            socket_timeout_s=_get_float_with_default(
                redis_streams_map,
                "socket_timeout_s",
                default=2.0,
            ),
            connect_timeout_s=_get_float_with_default(
                redis_streams_map,
                "connect_timeout_s",
                default=2.0,
            ),
            stream_mode=_get_str_with_default(
                redis_streams_map,
                "stream_mode",
                default="per_instrument",
            ),
            stream_prefix=_get_str_with_default(
                redis_streams_map,
                "stream_prefix",
                default="md.candles.1m",
            ),
            retention_days=_get_int_with_default(redis_streams_map, "retention_days", default=7),
            maxlen_approx=_get_optional_int(redis_streams_map, "maxlen_approx"),
        )
    )

    return MarketDataRuntimeConfig(
        version=version,
        markets=markets,
        ingestion=ingestion_cfg,
        scheduler=scheduler_cfg,
        backfill=backfill,
        live_feed=live_feed,
    )


def _parse_market(m: Any) -> MarketConfig:
    """
    Parse and validate one market mapping from runtime YAML.

    Parameters:
    - m: raw market mapping loaded from YAML.

    Returns:
    - Fully validated `MarketConfig`.

    Assumptions/Invariants:
    - Required nested sections `rest` and `ws` are present.
    - `rest.earliest_available_ts_utc` is ISO-8601 UTC timestamp not in the future.

    Errors/Exceptions:
    - Raises `ValueError` when shape or value constraints are violated.

    Side effects:
    - None.
    """
    if not isinstance(m, dict):
        raise ValueError("each market entry must be a mapping")

    market_id = MarketId(_get_int(m, "market_id", required=True))
    exchange = _get_str(m, "exchange", required=True)
    market_type = _get_str(m, "market_type", required=True)
    market_code = _get_str(m, "market_code", required=True)

    rest_map = _get_mapping(m, "rest", required=True)
    backoff_map = _get_mapping(rest_map, "backoff", required=True)
    limiter_map = _get_mapping(rest_map, "limiter", required=True)

    rest = RestConfig(
        base_url=_get_str(rest_map, "base_url", required=True),
        timeout_s=_get_float(rest_map, "timeout_s", required=True),
        retries=_get_int(rest_map, "retries", required=True),
        earliest_available_ts_utc=_get_utc_timestamp(
            rest_map,
            "earliest_available_ts_utc",
        ),
        backoff=BackoffConfig(
            base_s=_get_float(backoff_map, "base_s", required=True),
            max_s=_get_float(backoff_map, "max_s", required=True),
            jitter_s=_get_float(backoff_map, "jitter_s", required=True),
        ),
        limiter=RateLimiterConfig(
            mode=_get_str(limiter_map, "mode", required=True),
            safety_factor=_get_float(limiter_map, "safety_factor", required=True),
            max_concurrency=_get_int(limiter_map, "max_concurrency", required=True),
        ),
    )

    ws_map = _get_mapping(m, "ws", required=True)
    reconnect_map = _get_mapping(ws_map, "reconnect", required=True)
    ws = WsConfig(
        url=_get_str(ws_map, "url", required=True),
        ping_interval_s=_get_float(ws_map, "ping_interval_s", required=True),
        pong_timeout_s=_get_float(ws_map, "pong_timeout_s", required=True),
        reconnect=WsReconnectConfig(
            min_delay_s=_get_float(reconnect_map, "min_delay_s", required=True),
            max_delay_s=_get_float(reconnect_map, "max_delay_s", required=True),
            factor=_get_float(reconnect_map, "factor", required=True),
            jitter_s=_get_float(reconnect_map, "jitter_s", required=True),
        ),
        max_symbols_per_connection=_get_int(ws_map, "max_symbols_per_connection", required=True),
    )

    return MarketConfig(
        market_id=market_id,
        exchange=exchange,
        market_type=market_type,
        market_code=market_code,
        rest=rest,
        ws=ws,
    )


def _get_mapping(d: Mapping[str, Any], key: str, *, required: bool) -> Mapping[str, Any]:
    v = d.get(key)
    if v is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"expected mapping at key '{key}', got {type(v).__name__}")
    return v


def _get_list(d: Mapping[str, Any], key: str, *, required: bool) -> list[Any]:
    v = d.get(key)
    if v is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return []
    if not isinstance(v, list):
        raise ValueError(f"expected list at key '{key}', got {type(v).__name__}")
    return v


def _get_str(d: Mapping[str, Any], key: str, *, required: bool) -> str:
    v = d.get(key)
    if v is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return ""
    if not isinstance(v, str):
        raise ValueError(f"expected string at key '{key}', got {type(v).__name__}")
    if not v.strip():
        raise ValueError(f"key '{key}' must be non-empty")
    return v


def _get_int(d: Mapping[str, Any], key: str, *, required: bool) -> int:
    v = d.get(key)
    if v is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return 0
    if isinstance(v, bool):
        raise ValueError(f"expected int at key '{key}', got bool")
    if not isinstance(v, int):
        raise ValueError(f"expected int at key '{key}', got {type(v).__name__}")
    return v


def _get_float(d: Mapping[str, Any], key: str, *, required: bool) -> float:
    v = d.get(key)
    if v is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return 0.0
    if isinstance(v, bool):
        raise ValueError(f"expected float at key '{key}', got bool")
    if isinstance(v, (int, float)):
        return float(v)
    raise ValueError(f"expected float at key '{key}', got {type(v).__name__}")


def _get_utc_timestamp(d: Mapping[str, Any], key: str) -> UtcTimestamp:
    """
    Read one ISO-8601 UTC timestamp from mapping and validate it is not in future.

    Parameters:
    - d: source mapping.
    - key: timestamp field name.
    Returns:
    - Parsed `UtcTimestamp`.

    Assumptions/Invariants:
    - Accepts ISO-8601 with explicit timezone or `Z` suffix.
    - Result must not be later than current UTC wall clock.

    Errors/Exceptions:
    - Raises `ValueError` when value is missing, malformed, timezone-naive, or in future.

    Side effects:
    - None.
    """
    raw = _get_str(d, key, required=True)

    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp at key '{key}': {raw!r}") from exc

    ts = UtcTimestamp(parsed)
    now_utc = datetime.now(tz=timezone.utc)
    if ts.value > now_utc:
        raise ValueError(f"key '{key}' must not be in the future: {raw!r}")
    return ts


def _get_int_with_default(d: Mapping[str, Any], key: str, *, default: int) -> int:
    """
    Read optional integer key from mapping with explicit default.

    Parameters:
    - d: source mapping.
    - key: integer field name.
    - default: value returned when key is absent.

    Returns:
    - Parsed integer from mapping or `default`.

    Assumptions/Invariants:
    - Boolean values are rejected even though `bool` is an `int` subclass.

    Errors/Exceptions:
    - Raises `ValueError` when present value is not a valid int.

    Side effects:
    - None.
    """
    if key not in d:
        return default
    return _get_int(d, key, required=True)


def _get_str_with_default(d: Mapping[str, Any], key: str, *, default: str) -> str:
    """
    Read optional string key from mapping with explicit default.

    Parameters:
    - d: source mapping.
    - key: string field name.
    - default: value returned when key is absent.

    Returns:
    - Non-empty string value.

    Assumptions/Invariants:
    - Empty strings are not allowed.

    Errors/Exceptions:
    - Raises `ValueError` when present value is not a valid non-empty string.

    Side effects:
    - None.
    """
    if key not in d:
        return default
    return _get_str(d, key, required=True)


def _get_float_with_default(d: Mapping[str, Any], key: str, *, default: float) -> float:
    """
    Read optional float key from mapping with explicit default.

    Parameters:
    - d: source mapping.
    - key: float field name.
    - default: value returned when key is absent.

    Returns:
    - Parsed float from mapping or `default`.

    Assumptions/Invariants:
    - Integer values are accepted and converted to float.

    Errors/Exceptions:
    - Raises `ValueError` when present value is not numeric.

    Side effects:
    - None.
    """
    if key not in d:
        return default
    return _get_float(d, key, required=True)


def _get_bool_with_default(d: Mapping[str, Any], key: str, *, default: bool) -> bool:
    """
    Read optional boolean key from mapping with explicit default.

    Parameters:
    - d: source mapping.
    - key: boolean field name.
    - default: value returned when key is absent.

    Returns:
    - Parsed bool from mapping or `default`.

    Assumptions/Invariants:
    - Only explicit YAML booleans are accepted.

    Errors/Exceptions:
    - Raises `ValueError` when present value is not bool.

    Side effects:
    - None.
    """
    if key not in d:
        return default
    value = d[key]
    if not isinstance(value, bool):
        raise ValueError(f"expected bool at key '{key}', got {type(value).__name__}")
    return value


def _get_optional_int(d: Mapping[str, Any], key: str) -> int | None:
    """
    Read optional integer key where `null`/missing maps to None.

    Parameters:
    - d: source mapping.
    - key: integer field name.

    Returns:
    - Parsed positive integer, or `None` if key is missing or null.

    Assumptions/Invariants:
    - Boolean values are rejected.

    Errors/Exceptions:
    - Raises `ValueError` when present value is not a valid int.

    Side effects:
    - None.
    """
    if key not in d or d[key] is None:
        return None
    return _get_int(d, key, required=True)


def _get_optional_str_with_default(
    d: Mapping[str, Any],
    key: str,
    *,
    default: str | None,
) -> str | None:
    """
    Read optional string key with nullable support and explicit default.

    Parameters:
    - d: source mapping.
    - key: string field name.
    - default: returned when key is absent.

    Returns:
    - Non-empty stripped string, `None`, or default.

    Assumptions/Invariants:
    - Empty string is treated as invalid to avoid ambiguous env variable names.

    Errors/Exceptions:
    - Raises `ValueError` when present value is not string or is empty.

    Side effects:
    - None.
    """
    if key not in d:
        return default
    value = d[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected string at key '{key}', got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"key '{key}' must be non-empty when provided")
    return stripped


def _get_int_nested_with_default(
    d: Mapping[str, Any],
    *,
    section: str,
    key: str,
    default: int,
) -> int:
    """
    Read optional integer from nested mapping section with default.

    Parameters:
    - d: parent mapping containing optional subsection.
    - section: subsection name.
    - key: integer key inside subsection.
    - default: returned when subsection or key is absent.

    Returns:
    - Parsed integer value or default.

    Assumptions/Invariants:
    - Subsection, when present, must be a mapping.

    Errors/Exceptions:
    - Raises `ValueError` for invalid subsection shape or invalid integer value.

    Side effects:
    - None.
    """
    section_map = _get_mapping(d, section, required=False)
    return _get_int_with_default(section_map, key, default=default)


def _require_non_empty(name: str, s: str) -> None:
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive(name: str, x: float) -> None:
    if x <= 0:
        raise ValueError(f"{name} must be > 0, got {x}")


def _require_non_negative(name: str, x: float) -> None:
    if x < 0:
        raise ValueError(f"{name} must be >= 0, got {x}")


def _require_positive_int(name: str, x: int) -> None:
    if x <= 0:
        raise ValueError(f"{name} must be > 0, got {x}")


def _require_non_negative_int(name: str, x: int) -> None:
    if x < 0:
        raise ValueError(f"{name} must be >= 0, got {x}")
