from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from trading.shared_kernel.primitives.market_id import MarketId

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
        _require_positive_int("ingestion.raw_write.flush_interval_ms", self.flush_interval_ms)
        _require_positive_int("ingestion.raw_write.max_buffer_rows", self.max_buffer_rows)


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
class MarketDataRuntimeConfig:
    version: int
    markets: tuple[MarketConfig, ...]
    raw_write: RawWriteConfig
    backfill: BackfillConfig

    def market_by_id(self, market_id: MarketId) -> MarketConfig:
        for m in self.markets:
            if m.market_id.value == market_id.value:
                return m
        raise KeyError(f"market_id not found in config: {market_id.value}")

    def market_ids(self) -> tuple[int, ...]:
        return tuple(m.market_id.value for m in self.markets)


def load_market_data_runtime_config(path: str | Path) -> MarketDataRuntimeConfig:
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
    raw_write_map = _get_mapping(ingestion, "raw_write", required=True)
    raw_write = RawWriteConfig(
        flush_interval_ms=_get_int(raw_write_map, "flush_interval_ms", required=True),
        max_buffer_rows=_get_int(raw_write_map, "max_buffer_rows", required=True),
    )

    backfill_map = _get_mapping(md, "backfill", required=True)
    backfill = BackfillConfig(
        max_days_per_insert=_get_int(backfill_map, "max_days_per_insert", required=True),
        chunk_align=_get_str(backfill_map, "chunk_align", required=True),
    )

    return MarketDataRuntimeConfig(
        version=version,
        markets=markets,
        raw_write=raw_write,
        backfill=backfill,
    )


def _parse_market(m: Any) -> MarketConfig:
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
