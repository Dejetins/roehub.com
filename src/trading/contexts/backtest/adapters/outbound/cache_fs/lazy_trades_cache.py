from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

from trading.contexts.backtest.application.ports.lazy_trades_cache import (
    BacktestLazyTradesCache,
    BacktestLazyTradesCacheKey,
    BacktestLazyTradesCacheReadResult,
    normalize_json_payload,
)
from trading.contexts.backtest.application.services.v2.result_series import (
    MAX_BACKTEST_MONTHLY_STATS_ITEMS,
    MAX_BACKTEST_RESULT_POINTS,
    MAX_BACKTEST_SYMBOL_STATS_ITEMS,
    MAX_BACKTEST_TRADES_PAGE_SIZE,
    normalize_chart_points,
)

DEFAULT_LAZY_TRADES_CACHE_ROOT = Path("/opt/roehub/state/backtest/trades_cache")
_CACHE_SCHEMA = "backtest_lazy_trades_cache_bundle_v2"
_TRADES_FIELDS = (
    "trade_index",
    "entry_timestamp",
    "exit_timestamp",
    "side",
    "entry_price",
    "exit_price",
    "quantity",
    "notional_quote",
    "gross_pnl_quote",
    "net_pnl_quote",
    "return_pct",
    "fee_quote",
    "slippage_quote",
    "exit_reason",
    "equity_after",
    "safe_quote_after",
    "timeframe",
)


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
        metadata = self._read_metadata(cache_key=cache_key, now=now, ttl_seconds=ttl_seconds)
        if not metadata.is_hit or metadata.payload is None:
            return metadata
        payload = dict(metadata.payload)
        payload["trades"] = ()
        payload["chart_overlay"] = {
            "schema": "backtest_chart_overlay_v1",
            "markers": [],
            "segments": [],
        }
        return BacktestLazyTradesCacheReadResult(status="hit", payload=payload)

    def write(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        payload: Mapping[str, Any],
        now: datetime,
        ttl_seconds: int,
    ) -> None:
        bundle_dir = self._bundle_dir(cache_key=cache_key)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        written_at = now.astimezone(UTC)
        trades = _ordered_trades(
            tuple(item for item in payload.get("trades", ()) if isinstance(item, Mapping))
        )
        metadata_payload = {
            key: value for key, value in payload.items() if key not in {"trades", "chart_overlay"}
        }
        cache_payload = dict(_mapping(metadata_payload.get("cache")))
        cache_payload["cache_path"] = str(bundle_dir)
        metadata_payload["cache"] = cache_payload
        metadata_payload["chart_overlay"] = {
            "schema": "backtest_chart_overlay_v1",
            "markers": [],
            "segments": [],
        }
        envelope = {
            "schema": _CACHE_SCHEMA,
            "cache_key_digest": cache_key.digest,
            "cache_key": cache_key.as_mapping(),
            "written_at": written_at.isoformat().replace("+00:00", "Z"),
            "expires_at": (written_at + timedelta(seconds=ttl_seconds))
            .isoformat()
            .replace("+00:00", "Z"),
            "ttl_seconds": ttl_seconds,
            "trade_count": len(trades),
            "payload": normalize_json_payload(metadata_payload),
        }
        self._write_json_atomic(self._metadata_path(cache_key=cache_key), envelope)
        self._write_trades_atomic(self._trades_path(cache_key=cache_key), trades)

    def read_page(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
        page: int,
        page_size: int,
    ) -> BacktestLazyTradesCacheReadResult:
        metadata = self._read_metadata(cache_key=cache_key, now=now, ttl_seconds=ttl_seconds)
        if not metadata.is_hit or metadata.payload is None:
            return metadata
        effective_page = max(1, page)
        effective_page_size = min(max(1, page_size), MAX_BACKTEST_TRADES_PAGE_SIZE)
        total = _trade_count(metadata.payload)
        offset = (effective_page - 1) * effective_page_size
        items = tuple(
            _iter_trade_slice(
                path=self._trades_path(cache_key=cache_key),
                offset=offset,
                limit=effective_page_size,
            )
        )
        has_next = offset + effective_page_size < total
        payload = _view_base(metadata.payload)
        payload.update(
            {
                "items": items,
                "pagination": {
                    "mode": "page",
                    "page": effective_page,
                    "page_size": effective_page_size,
                    "max_page_size": MAX_BACKTEST_TRADES_PAGE_SIZE,
                    "total": total,
                    "has_next": has_next,
                    "has_previous": effective_page > 1,
                    "next_page": effective_page + 1 if has_next else None,
                    "previous_page": effective_page - 1 if effective_page > 1 else None,
                    "sort": "trade_index_asc",
                },
            }
        )
        return BacktestLazyTradesCacheReadResult(status="hit", payload=payload)

    def read_series(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
        kind: str,
        points: int,
    ) -> BacktestLazyTradesCacheReadResult:
        metadata = self._read_metadata(cache_key=cache_key, now=now, ttl_seconds=ttl_seconds)
        if not metadata.is_hit or metadata.payload is None:
            return metadata
        limit = normalize_chart_points(points)
        source_points = _count_equity_points(self._trades_path(cache_key=cache_key))
        selected = _selected_indices(source_points=source_points, limit=limit)
        sampled = tuple(
            _iter_sampled_series_points(
                path=self._trades_path(cache_key=cache_key),
                selected=selected,
                kind=kind,
            )
        )
        payload = _view_base(metadata.payload)
        payload.update(
            {
                "kind": kind,
                "points": sampled,
                "requested_points": points,
                "max_points": limit,
                "returned_points": len(sampled),
                "source_points": source_points,
                "downsampled": len(sampled) < source_points,
            }
        )
        return BacktestLazyTradesCacheReadResult(status="hit", payload=payload)

    def read_monthly_stats(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
    ) -> BacktestLazyTradesCacheReadResult:
        metadata = self._read_metadata(cache_key=cache_key, now=now, ttl_seconds=ttl_seconds)
        if not metadata.is_hit or metadata.payload is None:
            return metadata
        buckets: dict[str, dict[str, Any]] = {}
        for trade in _iter_trades(path=self._trades_path(cache_key=cache_key)):
            month = _month_key(trade.get("exit_timestamp"))
            bucket = buckets.setdefault(
                month,
                {
                    "month": month,
                    "trades_count": 0,
                    "net_pnl_quote": 0.0,
                    "return_pct": 0.0,
                    "wins": 0,
                    "losses": 0,
                },
            )
            pnl = _float(trade.get("net_pnl_quote"))
            bucket["trades_count"] += 1
            bucket["net_pnl_quote"] += pnl
            bucket["return_pct"] += _float(trade.get("return_pct"))
            if pnl > 0:
                bucket["wins"] += 1
            elif pnl < 0:
                bucket["losses"] += 1
        source_items = tuple(_with_win_rate(item) for _, item in sorted(buckets.items()))
        items = source_items[:MAX_BACKTEST_MONTHLY_STATS_ITEMS]
        payload = _view_base(metadata.payload)
        payload.update(
            {
                "kind": "monthly",
                "items": items,
                "bounds": {
                    "max_items": MAX_BACKTEST_MONTHLY_STATS_ITEMS,
                    "returned_items": len(items),
                    "source_items": len(source_items),
                    "truncated": len(items) < len(source_items),
                    "sort": "month_asc",
                },
            }
        )
        return BacktestLazyTradesCacheReadResult(status="hit", payload=payload)

    def read_symbol_stats(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
        symbol: str | None,
    ) -> BacktestLazyTradesCacheReadResult:
        metadata = self._read_metadata(cache_key=cache_key, now=now, ttl_seconds=ttl_seconds)
        if not metadata.is_hit or metadata.payload is None:
            return metadata
        item: dict[str, Any] = {
            "symbol": symbol or "unknown",
            "trades_count": 0,
            "net_pnl_quote": 0.0,
            "return_pct": 0.0,
            "wins": 0,
            "losses": 0,
        }
        for trade in _iter_trades(path=self._trades_path(cache_key=cache_key)):
            pnl = _float(trade.get("net_pnl_quote"))
            item["trades_count"] += 1
            item["net_pnl_quote"] += pnl
            item["return_pct"] += _float(trade.get("return_pct"))
            if pnl > 0:
                item["wins"] += 1
            elif pnl < 0:
                item["losses"] += 1
        payload = _view_base(metadata.payload)
        payload.update(
            {
                "kind": "symbol",
                "items": (_with_win_rate(item),),
                "bounds": {
                    "max_items": MAX_BACKTEST_SYMBOL_STATS_ITEMS,
                    "returned_items": 1,
                    "source_items": 1,
                    "truncated": False,
                    "sort": "symbol_asc",
                },
            }
        )
        return BacktestLazyTradesCacheReadResult(status="hit", payload=payload)

    def read_csv(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
        max_rows: int,
    ) -> BacktestLazyTradesCacheReadResult:
        metadata = self._read_metadata(cache_key=cache_key, now=now, ttl_seconds=ttl_seconds)
        if not metadata.is_hit or metadata.payload is None:
            return metadata
        total = _trade_count(metadata.payload)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=_TRADES_FIELDS, extrasaction="ignore")
        writer.writeheader()
        row_count = 0
        for trade in _iter_trade_slice(
            path=self._trades_path(cache_key=cache_key),
            offset=0,
            limit=max_rows,
        ):
            writer.writerow({field: trade.get(field) for field in _TRADES_FIELDS})
            row_count += 1
        payload = dict(metadata.payload)
        payload.update(
            {
                "content": output.getvalue(),
                "row_count": row_count,
                "total_rows": total,
                "max_rows": max_rows,
                "truncated": total > max_rows,
                "sort": "trade_index_asc",
            }
        )
        return BacktestLazyTradesCacheReadResult(status="hit", payload=payload)

    def _read_metadata(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
    ) -> BacktestLazyTradesCacheReadResult:
        path = self._metadata_path(cache_key=cache_key)
        if not path.exists():
            if self._legacy_path(cache_key=cache_key).exists():
                return BacktestLazyTradesCacheReadResult(
                    status="miss",
                    warning="legacy monolithic cache ignored",
                )
            return BacktestLazyTradesCacheReadResult(status="miss")
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, Mapping):
                return BacktestLazyTradesCacheReadResult(
                    status="read_failed",
                    warning="cache metadata is not a JSON object",
                )
            if raw.get("schema") != _CACHE_SCHEMA:
                return BacktestLazyTradesCacheReadResult(
                    status="read_failed",
                    warning="unsupported lazy trades cache schema",
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
            normalized = dict(payload)
            normalized["trade_count"] = int(raw.get("trade_count") or 0)
            return BacktestLazyTradesCacheReadResult(status="hit", payload=normalized)
        except Exception as error:  # noqa: BLE001
            return BacktestLazyTradesCacheReadResult(status="read_failed", warning=str(error))

    def _write_json_atomic(self, path: Path, payload: Mapping[str, Any]) -> None:
        rendered = json.dumps(
            normalize_json_payload(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        _write_text_atomic(path=path, content=f"{rendered}\n")

    def _write_trades_atomic(
        self,
        path: Path,
        trades: Sequence[Mapping[str, Any]],
    ) -> None:
        lines = "".join(
            json.dumps(
                normalize_json_payload(trade),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
            for trade in trades
        )
        _write_text_atomic(path=path, content=lines)

    def _bundle_dir(self, *, cache_key: BacktestLazyTradesCacheKey) -> Path:
        digest = cache_key.digest
        return self.root / digest[:2] / digest

    def _metadata_path(self, *, cache_key: BacktestLazyTradesCacheKey) -> Path:
        return self._bundle_dir(cache_key=cache_key) / "metadata.json"

    def _trades_path(self, *, cache_key: BacktestLazyTradesCacheKey) -> Path:
        return self._bundle_dir(cache_key=cache_key) / "trades.jsonl"

    def _legacy_path(self, *, cache_key: BacktestLazyTradesCacheKey) -> Path:
        digest = cache_key.digest
        return self.root / digest[:2] / f"{digest}.json"


def _write_text_atomic(*, path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _iter_trades(*, path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if isinstance(raw, Mapping):
                yield dict(raw)


def _iter_trade_slice(*, path: Path, offset: int, limit: int) -> Iterable[dict[str, Any]]:
    if limit <= 0:
        return
    for index, trade in enumerate(_iter_trades(path=path)):
        if index < offset:
            continue
        if index >= offset + limit:
            break
        yield trade


def _count_equity_points(path: Path) -> int:
    count = 0
    for trade in _iter_trades(path=path):
        if _float_or_none(trade.get("equity_after")) is not None:
            count += 1
    return count


def _selected_indices(*, source_points: int, limit: int) -> frozenset[int]:
    if source_points <= 0:
        return frozenset()
    if source_points <= limit:
        return frozenset(range(source_points))
    effective_limit = min(max(1, limit), MAX_BACKTEST_RESULT_POINTS)
    if effective_limit <= 1:
        return frozenset({source_points - 1})
    step = (source_points - 1) / (effective_limit - 1)
    return frozenset(int(round(item_index * step)) for item_index in range(effective_limit))


def _iter_sampled_series_points(
    *,
    path: Path,
    selected: frozenset[int],
    kind: str,
) -> Iterable[dict[str, Any]]:
    peak: float | None = None
    point_index = 0
    for trade in _iter_trades(path=path):
        equity = _float_or_none(trade.get("equity_after"))
        if equity is None:
            continue
        value = equity
        payload = {
            "x": trade.get("exit_timestamp") or trade.get("trade_index"),
            "trade_index": trade.get("trade_index"),
            "value": value,
            "net_pnl_quote": _float(trade.get("net_pnl_quote")),
        }
        if kind == "drawdown":
            peak = value if peak is None else max(peak, value)
            drawdown = 0.0 if peak <= 0 else ((value - peak) / peak) * 100.0
            payload = {
                "x": trade.get("exit_timestamp") or trade.get("trade_index"),
                "trade_index": trade.get("trade_index"),
                "value": drawdown,
                "equity": value,
            }
        if point_index in selected:
            yield payload
        point_index += 1


def _view_base(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "job_id": str(payload["job_id"]),
            "variant_key": str(payload["variant_key"]),
            "variant_hash": str(payload["variant_hash"]),
            "summary_metrics": _mapping(payload.get("summary_metrics")),
            "cache": _cache_hit_payload(payload=payload),
            "timing": _mapping(payload.get("timing")),
        }


def _cache_hit_payload(*, payload: Mapping[str, Any]) -> dict[str, Any]:
    cache = _mapping(payload.get("cache"))
    cache["status"] = "hit"
    return cache


def _trade_count(payload: Mapping[str, Any]) -> int:
    return int(payload.get("trade_count") or 0)


def _ordered_trades(trades: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        trade
        for _, trade in sorted(
            enumerate(trades),
            key=lambda item: _trade_sort_key(index=item[0], trade=item[1]),
        )
    )


def _trade_sort_key(*, index: int, trade: Mapping[str, Any]) -> tuple[int, float, str, str, int]:
    trade_index = _float_or_none(trade.get("trade_index"))
    trade_index_missing = 1 if trade_index is None else 0
    return (
        trade_index_missing,
        trade_index if trade_index is not None else float(index),
        str(trade.get("exit_timestamp") or ""),
        str(trade.get("entry_timestamp") or ""),
        index,
    )


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _month_key(value: Any) -> str:
    if isinstance(value, str) and len(value) >= 7:
        return value[:7]
    return "unknown"


def _with_win_rate(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    trades = int(payload.get("trades_count") or 0)
    wins = int(payload.get("wins") or 0)
    payload["win_rate_pct"] = (wins / trades) * 100.0 if trades > 0 else None
    return payload


def _float(value: Any) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else 0.0


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


__all__ = [
    "DEFAULT_LAZY_TRADES_CACHE_ROOT",
    "LocalFileBacktestLazyTradesCache",
]
