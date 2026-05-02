from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from trading.contexts.backtest.application.ports.lazy_trades_cache import (
    BacktestLazyTradesCache,
    BacktestLazyTradesCacheKey,
    BacktestLazyTradesCacheReadResult,
    normalize_json_payload,
)

DEFAULT_LAZY_TRADES_CACHE_ROOT = Path("/opt/roehub/state/backtest/trades_cache")


@dataclass(frozen=True, slots=True)
class LocalFileBacktestLazyTradesCache(BacktestLazyTradesCache):
    root: Path = DEFAULT_LAZY_TRADES_CACHE_ROOT

    def read(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
    ) -> BacktestLazyTradesCacheReadResult:
        path = self._path(cache_key=cache_key)
        if not path.exists():
            return BacktestLazyTradesCacheReadResult(status="miss")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                return BacktestLazyTradesCacheReadResult(
                    status="read_failed",
                    warning="cache envelope is not a JSON object",
                )
            if raw.get("cache_key_digest") != cache_key.digest:
                return BacktestLazyTradesCacheReadResult(
                    status="read_failed",
                    warning="cache key digest mismatch",
                )
            written_at = _parse_datetime(raw.get("written_at"))
            if written_at is None:
                return BacktestLazyTradesCacheReadResult(
                    status="read_failed",
                    warning="cache written_at is missing or invalid",
                )
            if now.astimezone(UTC) - written_at > timedelta(seconds=ttl_seconds):
                return BacktestLazyTradesCacheReadResult(status="expired")
            payload = raw.get("payload")
            if not isinstance(payload, Mapping):
                return BacktestLazyTradesCacheReadResult(
                    status="read_failed",
                    warning="cache payload is not a JSON object",
                )
            return BacktestLazyTradesCacheReadResult(status="hit", payload=payload)
        except Exception as error:  # noqa: BLE001
            return BacktestLazyTradesCacheReadResult(status="read_failed", warning=str(error))

    def write(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        payload: Mapping[str, Any],
        now: datetime,
        ttl_seconds: int,
    ) -> None:
        path = self._path(cache_key=cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        written_at = now.astimezone(UTC)
        envelope = {
            "schema": "backtest_lazy_trades_cache_v1",
            "cache_key_digest": cache_key.digest,
            "cache_key": cache_key.as_mapping(),
            "written_at": written_at.isoformat().replace("+00:00", "Z"),
            "expires_at": (written_at + timedelta(seconds=ttl_seconds))
            .isoformat()
            .replace("+00:00", "Z"),
            "ttl_seconds": ttl_seconds,
            "payload": normalize_json_payload(payload),
        }
        rendered = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(rendered)
            tmp.write("\n")
            tmp_path = Path(tmp.name)
        try:
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _path(self, *, cache_key: BacktestLazyTradesCacheKey) -> Path:
        digest = cache_key.digest
        return self.root / digest[:2] / f"{digest}.json"


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


__all__ = [
    "DEFAULT_LAZY_TRADES_CACHE_ROOT",
    "LocalFileBacktestLazyTradesCache",
]
