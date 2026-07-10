from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from trading.contexts.rl_trading.domain.stage08k_monitor_runtime import (
    Stage08kPendingVirtualTrade,
)


class FileStage08kMonitorStateStore:
    """Durable local state for one-minute monitor-only virtual positions."""

    def __init__(self, *, path: Path) -> None:
        if not path.is_absolute():
            raise ValueError("monitor state path must be absolute")
        self._path = path
        self._lock = threading.Lock()
        self._pending, self._last_processed = self._load()

    def get(self, *, instrument_key: str) -> Stage08kPendingVirtualTrade | None:
        with self._lock:
            return self._pending.get(instrument_key)

    def all_pending(self) -> tuple[Stage08kPendingVirtualTrade, ...]:
        with self._lock:
            return tuple(self._pending[key] for key in sorted(self._pending))

    def last_processed_close_utc(self, *, instrument_key: str) -> datetime | None:
        with self._lock:
            return self._last_processed.get(instrument_key)

    def commit_processed(
        self,
        *,
        instrument_key: str,
        candle_close_utc: datetime,
        pending_trade: Stage08kPendingVirtualTrade | None,
    ) -> None:
        if candle_close_utc.tzinfo is None:
            raise ValueError("monitor candle close must be timezone-aware")
        normalized_close = candle_close_utc.astimezone(UTC)
        if pending_trade is not None and pending_trade.instrument_key != instrument_key:
            raise ValueError("pending trade instrument does not match commit instrument")
        with self._lock:
            previous = self._last_processed.get(instrument_key)
            if previous is not None and normalized_close < previous:
                raise ValueError("monitor state time regression")
            if pending_trade is None:
                self._pending.pop(instrument_key, None)
            else:
                self._pending[instrument_key] = pending_trade
            self._last_processed[instrument_key] = normalized_close
            self._persist()

    def upsert(self, *, trade: Stage08kPendingVirtualTrade) -> None:
        with self._lock:
            current = self._pending.get(trade.instrument_key)
            if current is not None and current != trade:
                raise ValueError("pending virtual trade already exists")
            self._pending[trade.instrument_key] = trade
            self._persist()

    def remove(self, *, instrument_key: str) -> None:
        with self._lock:
            self._pending.pop(instrument_key, None)
            self._persist()

    def _load(
        self,
    ) -> tuple[dict[str, Stage08kPendingVirtualTrade], dict[str, datetime]]:
        if not self._path.exists():
            return {}, {}
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("invalid stage08k monitor state payload")
        rows = payload.get("pending", [])
        if not isinstance(rows, list):
            raise ValueError("invalid stage08k monitor pending payload")
        pending: dict[str, Stage08kPendingVirtualTrade] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid stage08k monitor pending row")
            trade = Stage08kPendingVirtualTrade.from_payload(row)
            if trade.instrument_key in pending:
                raise ValueError("duplicate stage08k monitor instrument state")
            pending[trade.instrument_key] = trade
        raw_last_processed = payload.get("last_processed_close_utc", {})
        if not isinstance(raw_last_processed, dict):
            raise ValueError("invalid stage08k monitor cursor payload")
        last_processed: dict[str, datetime] = {}
        for instrument_key, rendered in raw_last_processed.items():
            parsed = datetime.fromisoformat(str(rendered).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("monitor cursor timestamp must be timezone-aware")
            last_processed[str(instrument_key)] = parsed.astimezone(UTC)
        return pending, last_processed

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_processed_close_utc": {
                key: self._last_processed[key]
                .astimezone(UTC)
                .isoformat()
                .replace("+00:00", "Z")
                for key in sorted(self._last_processed)
            },
            "pending": [self._pending[key].as_payload() for key in sorted(self._pending)],
            "schema_version": 1,
        }
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self._path)
