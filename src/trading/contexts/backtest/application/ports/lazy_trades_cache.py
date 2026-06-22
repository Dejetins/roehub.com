from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Protocol

BacktestLazyTradesCacheStatus = Literal[
    "miss",
    "hit",
    "expired",
    "read_failed",
    "write_failed",
]


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesCacheKey:
    job_id: str
    variant_key: str
    variant_hash: str
    request_hash: str
    engine_params_hash: str
    artifact_manifest_hash: str
    funding_manifest_hash: str | None = None

    def as_mapping(self) -> dict[str, str | None]:
        payload: dict[str, str | None] = {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "request_hash": self.request_hash,
            "engine_params_hash": self.engine_params_hash,
            "artifact_manifest_hash": self.artifact_manifest_hash,
            "funding_manifest_hash": self.funding_manifest_hash,
        }
        return payload

    def identity_mapping(self) -> dict[str, str]:
        payload = {key: value for key, value in self.as_mapping().items() if value is not None}
        return {key: str(value) for key, value in payload.items()}

    @property
    def digest(self) -> str:
        return canonical_json_sha256(self.identity_mapping())


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesCacheReadResult:
    status: BacktestLazyTradesCacheStatus
    payload: Mapping[str, Any] | None = None
    warning: str | None = None

    @property
    def is_hit(self) -> bool:
        return self.status == "hit" and self.payload is not None


class BacktestLazyTradesCache(Protocol):
    def read(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
    ) -> BacktestLazyTradesCacheReadResult: ...

    def write(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        payload: Mapping[str, Any],
        now: datetime,
        ttl_seconds: int,
    ) -> None: ...

    def read_page(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
        page: int,
        page_size: int,
    ) -> BacktestLazyTradesCacheReadResult: ...

    def read_series(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
        kind: str,
        points: int,
    ) -> BacktestLazyTradesCacheReadResult: ...

    def read_monthly_stats(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
    ) -> BacktestLazyTradesCacheReadResult: ...

    def read_symbol_stats(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
        symbol: str | None,
    ) -> BacktestLazyTradesCacheReadResult: ...

    def read_csv(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
        max_rows: int,
    ) -> BacktestLazyTradesCacheReadResult: ...


def build_lazy_trades_cache_key(
    *,
    job_id: str,
    variant_key: str,
    variant_hash: str,
    request_hash: str,
    engine_params_hash: str,
    artifact_manifest_hash: str,
    funding_manifest_hash: str | None = None,
) -> BacktestLazyTradesCacheKey:
    return BacktestLazyTradesCacheKey(
        job_id=job_id,
        variant_key=variant_key,
        variant_hash=variant_hash,
        request_hash=request_hash,
        engine_params_hash=engine_params_hash,
        artifact_manifest_hash=artifact_manifest_hash,
        funding_manifest_hash=funding_manifest_hash,
    )


def canonical_json_sha256(payload: Any) -> str:
    rendered = json.dumps(
        normalize_json_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def normalize_json_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): normalize_json_payload(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [normalize_json_payload(item) for item in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return normalize_json_payload(item())
    return str(value)


__all__ = [
    "BacktestLazyTradesCache",
    "BacktestLazyTradesCacheKey",
    "BacktestLazyTradesCacheReadResult",
    "BacktestLazyTradesCacheStatus",
    "build_lazy_trades_cache_key",
    "canonical_json_sha256",
    "normalize_json_payload",
]
