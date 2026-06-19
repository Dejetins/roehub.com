from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apps.cli.wiring.db.clickhouse import (  # noqa: E402
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.market_data.adapters.outbound.clients.common_http import (  # noqa: E402
    RequestsHttpClient,
)
from trading.contexts.market_data.adapters.outbound.clients.rest_candle_ingest_source import (  # noqa: E402,E501
    RestCandleIngestSource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (  # noqa: E402
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.canonical_candle_index_reader import (  # noqa: E402,E501
    ClickHouseCanonicalCandleIndexReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (  # noqa: E402,E501
    ClickHouseConnectGateway,
    ClickHouseGateway,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.raw_kline_writer import (  # noqa: E402,E501
    ClickHouseRawKlineWriter,
)
from trading.contexts.market_data.application.dto import RestFillTask  # noqa: E402
from trading.contexts.market_data.application.use_cases import RestFillRange1mUseCase  # noqa: E402
from trading.platform.time.system_clock import SystemClock  # noqa: E402
from trading.shared_kernel.primitives import (  # noqa: E402
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)

BINANCE_FUTURES_MARKET_CODE = "binance:futures"
DATASET_HF_PERIOD_REBUILD = "hf_period_rebuild_current_trading"
DATASET_POST_HF_EXTENSION = "post_hf_extension_current_trading"
STALE_REASON = "excluded_stale_not_currently_trading_usdt_perpetual"
DEFAULT_STAGE04A_MANIFEST = Path(
    "/opt/roehub/state/rl_trading/stage04a_binance_futures_universe/"
    "stage04a_universe_manifest.json"
)
DEFAULT_STAGE04B_ROOT = Path(
    "/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill"
)
DEFAULT_PLAN_JSON = DEFAULT_STAGE04B_ROOT / "stage04b_backfill_resume_manifest.json"
DEFAULT_COVERAGE_JSON = DEFAULT_STAGE04B_ROOT / "stage04b_coverage_report.json"
DEFAULT_LOG_JSONL = DEFAULT_STAGE04B_ROOT / "stage04b_backfill_run.jsonl"


@dataclass(frozen=True, slots=True)
class SourceWindow:
    dataset_version: str
    source_start: datetime
    source_end: datetime


@dataclass(frozen=True, slots=True)
class ChunkRunReport:
    chunk_id: str
    rows_read: int
    rows_written: int
    batches_written: int
    started_at_utc: str
    finished_at_utc: str


def build_plan_manifest(
    *,
    stage04a_manifest: Mapping[str, Any],
    exchange_info: Mapping[str, Any],
    latest_candle_utc: datetime,
    generated_at_utc: datetime,
    chunk_days: int,
) -> dict[str, Any]:
    if chunk_days <= 0 or chunk_days > 31:
        raise ValueError("chunk_days must be in [1..31]")

    market_id = int(stage04a_manifest["market_id"])
    accepted_windows = _source_lower_bounds_by_symbol(stage04a_manifest)
    current_symbols = _current_trading_usdt_perpetual_symbols(exchange_info)
    accepted_symbols = sorted(accepted_windows)
    stale_symbols = [symbol for symbol in accepted_symbols if symbol not in current_symbols]
    active_symbols = [symbol for symbol in accepted_symbols if symbol in current_symbols]
    source_windows = _dataset_source_windows(latest_candle_utc)

    symbol_plans: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for symbol in active_symbols:
        source_lower_bound = accepted_windows[symbol]
        window_plans: list[dict[str, Any]] = []
        for source_window in source_windows:
            safe_start = max(source_window.source_start, source_lower_bound)
            if safe_start >= source_window.source_end:
                window_plans.append(
                    {
                        "dataset_version": source_window.dataset_version,
                        "required_source_start_utc": _format_utc(source_window.source_start),
                        "required_source_end_utc": _format_utc(source_window.source_end),
                        "safe_source_start_utc": _format_utc(safe_start),
                        "safe_source_end_utc": _format_utc(source_window.source_end),
                        "expected_minutes": 0,
                        "chunk_count": 0,
                        "status": "empty_after_history_start",
                    }
                )
                continue

            planned_chunks = [
                _chunk_record(
                    market_id=market_id,
                    symbol=symbol,
                    dataset_version=source_window.dataset_version,
                    start=chunk_start,
                    end=chunk_end,
                )
                for chunk_start, chunk_end in _iter_chunks(
                    start=safe_start,
                    end=source_window.source_end,
                    chunk_days=chunk_days,
                )
            ]
            chunks.extend(planned_chunks)
            window_plans.append(
                {
                    "dataset_version": source_window.dataset_version,
                    "required_source_start_utc": _format_utc(source_window.source_start),
                    "required_source_end_utc": _format_utc(source_window.source_end),
                    "safe_source_start_utc": _format_utc(safe_start),
                    "safe_source_end_utc": _format_utc(source_window.source_end),
                    "expected_minutes": _minute_count(safe_start, source_window.source_end),
                    "chunk_count": len(planned_chunks),
                    "status": "planned",
                }
            )

        symbol_plans.append(
            {
                "symbol": symbol,
                "market_id": market_id,
                "source_lower_bound_utc": _format_utc(source_lower_bound),
                "windows": window_plans,
            }
        )

    chunks = sorted(
        chunks,
        key=lambda item: (
            str(item["symbol"]),
            str(item["dataset_version"]),
            str(item["start_utc"]),
            str(item["end_utc"]),
        ),
    )
    for chunk in chunks:
        chunk["status"] = "pending"

    return {
        "schema_version": 1,
        "stage": "04B",
        "market": BINANCE_FUTURES_MARKET_CODE,
        "market_id": market_id,
        "generated_at_utc": _format_utc(generated_at_utc),
        "latest_binance_futures_candle_utc": _format_utc(latest_candle_utc),
        "post_hf_extension_source_end_utc": _format_utc(
            _floor_minute(latest_candle_utc) + timedelta(minutes=1)
        ),
        "stage04a_manifest": {
            "accepted_count": int(stage04a_manifest.get("accepted_count", len(accepted_symbols))),
            "accepted_symbols_sha256": _hash_lines(accepted_symbols),
            "source_manifest_stage": stage04a_manifest.get("stage"),
            "source_manifest_market": stage04a_manifest.get("market"),
        },
        "current_metadata": {
            "current_trading_usdt_perpetual_count": len(current_symbols),
            "active_stage04a_symbol_count": len(active_symbols),
            "stale_stage04a_symbol_count": len(stale_symbols),
            "stale_symbols": [
                {"symbol": symbol, "reason": STALE_REASON} for symbol in stale_symbols
            ],
            "stale_symbols_sha256": _hash_lines(stale_symbols),
        },
        "dataset_versions": [
            {
                "dataset_version": window.dataset_version,
                "source_start_utc": _format_utc(window.source_start),
                "source_end_utc": _format_utc(window.source_end),
            }
            for window in source_windows
        ],
        "chunk_policy": {
            "chunk_days": chunk_days,
            "range_semantics": "half-open UTC [start, end)",
            "reason": "stage04b_backfill",
        },
        "symbols": symbol_plans,
        "chunks": chunks,
        "summary": {
            "symbols_total_from_stage04a": len(accepted_symbols),
            "symbols_planned": len(active_symbols),
            "chunks_total": len(chunks),
            "expected_minutes_total": sum(int(chunk["expected_minutes"]) for chunk in chunks),
            "chunks_sha256": _hash_json(chunks),
        },
        "execution": {
            "status": "planned",
            "started_at_utc": None,
            "updated_at_utc": _format_utc(generated_at_utc),
            "finished_at_utc": None,
            "chunks_completed": 0,
            "chunks_failed": 0,
            "rows_read": 0,
            "rows_written": 0,
            "batches_written": 0,
        },
    }


def build_coverage_report(
    *,
    plan_manifest: Mapping[str, Any],
    gateway: ClickHouseGateway,
    database: str,
    collected_at_utc: datetime,
) -> dict[str, Any]:
    market_id = int(plan_manifest["market_id"])
    entries: list[dict[str, Any]] = []
    for symbol_plan in cast(Sequence[Mapping[str, Any]], plan_manifest["symbols"]):
        symbol = str(symbol_plan["symbol"])
        for window in cast(Sequence[Mapping[str, Any]], symbol_plan["windows"]):
            if int(window["expected_minutes"]) <= 0:
                continue
            start = _parse_utc(str(window["safe_source_start_utc"]))
            end = _parse_utc(str(window["safe_source_end_utc"]))
            row = _query_coverage_row(
                gateway=gateway,
                database=database,
                market_id=market_id,
                symbol=symbol,
                start=start,
                end=end,
            )
            entries.append(
                coverage_entry_from_row(
                    symbol=symbol,
                    dataset_version=str(window["dataset_version"]),
                    start=start,
                    end=end,
                    row=row,
                )
            )

    blocked_entries = [
        entry
        for entry in entries
        if entry["missing_minutes"] != 0
        or entry["duplicate_rows"] != 0
        or entry["volume_quote_rows"] != entry["physical_rows"]
        or entry["trades_count_rows"] != entry["physical_rows"]
        or entry["vwap_computable_rows"] + entry["zero_volume_rows"] != entry["physical_rows"]
    ]
    return {
        "schema_version": 1,
        "stage": "04B",
        "market": BINANCE_FUTURES_MARKET_CODE,
        "collected_at_utc": _format_utc(collected_at_utc),
        "latest_binance_futures_candle_utc": plan_manifest.get(
            "latest_binance_futures_candle_utc"
        ),
        "coverage_status": "accepted_coverage" if not blocked_entries else "residual_gaps",
        "summary": {
            "windows_total": len(entries),
            "windows_blocked": len(blocked_entries),
            "expected_minutes_total": sum(int(entry["expected_minutes"]) for entry in entries),
            "distinct_minutes_total": sum(int(entry["distinct_minutes"]) for entry in entries),
            "missing_minutes_total": sum(int(entry["missing_minutes"]) for entry in entries),
            "duplicate_rows_total": sum(int(entry["duplicate_rows"]) for entry in entries),
            "zero_volume_rows_total": sum(int(entry["zero_volume_rows"]) for entry in entries),
        },
        "entries": entries,
        "entries_sha256": _hash_json(entries),
    }


def coverage_entry_from_row(
    *,
    symbol: str,
    dataset_version: str,
    start: datetime,
    end: datetime,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    expected_minutes = _minute_count(start, end)
    physical_rows = int(row.get("physical_rows") or 0)
    distinct_minutes = int(row.get("distinct_minutes") or 0)
    duplicate_rows = max(0, physical_rows - distinct_minutes)
    first_minute_key = row.get("first_minute_key")
    last_minute_key = row.get("last_minute_key")
    return {
        "symbol": symbol,
        "dataset_version": dataset_version,
        "start_utc": _format_utc(start),
        "end_utc": _format_utc(end),
        "expected_minutes": expected_minutes,
        "physical_rows": physical_rows,
        "distinct_minutes": distinct_minutes,
        "missing_minutes": max(0, expected_minutes - distinct_minutes),
        "duplicate_rows": duplicate_rows,
        "first_candle_utc": (
            None if first_minute_key is None else _format_utc(_minute_key_to_dt(first_minute_key))
        ),
        "last_candle_utc": (
            None if last_minute_key is None else _format_utc(_minute_key_to_dt(last_minute_key))
        ),
        "volume_quote_rows": int(row.get("volume_quote_rows") or 0),
        "trades_count_rows": int(row.get("trades_count_rows") or 0),
        "zero_volume_rows": int(row.get("zero_volume_rows") or 0),
        "vwap_computable_rows": int(row.get("vwap_computable_rows") or 0),
    }


def execute_manifest(
    *,
    manifest_path: Path,
    market_config_path: Path,
    batch_size: int,
    max_chunks: int | None,
    max_runtime_seconds: float | None,
    delay_s: float,
    log_jsonl: Path,
    workers: int,
    max_chunk_attempts: int,
    failure_backoff_s: float,
    skip_covered_chunks: bool,
) -> int:
    if workers <= 0:
        raise ValueError("workers must be > 0")
    if max_chunk_attempts <= 0:
        raise ValueError("max_chunk_attempts must be > 0")
    if failure_backoff_s < 0:
        raise ValueError("failure_backoff_s must be >= 0")

    manifest = _read_json(manifest_path)
    cfg = load_market_data_runtime_config(market_config_path)
    settings = ClickHouseSettingsLoader(os.environ).load()
    gateway: ClickHouseGateway
    if workers == 1:
        gateway = ClickHouseConnectGateway(_clickhouse_client(settings))
    else:
        gateway = ThreadLocalClickHouseConnectGateway(
            client_factory=lambda: _clickhouse_client(settings),
        )
    ingest_id = uuid4()
    thread_state = threading.local()

    def use_case_factory() -> RestFillRange1mUseCase:
        use_case = getattr(thread_state, "use_case", None)
        if use_case is None:
            use_case = _make_rest_fill_use_case(
                cfg=cfg,
                gateway=gateway,
                database=settings.database,
                batch_size=batch_size,
                ingest_id=ingest_id,
            )
            thread_state.use_case = use_case
        return cast(RestFillRange1mUseCase, use_case)

    execution = cast(dict[str, Any], manifest["execution"])
    now = _now_utc()
    if execution.get("started_at_utc") is None:
        execution["started_at_utc"] = _format_utc(now)
    execution["status"] = "running"
    execution["updated_at_utc"] = _format_utc(now)
    execution["ingest_id"] = str(ingest_id)
    execution["coordinator"] = "sharded_parallel" if workers > 1 else "sequential"
    execution["workers"] = workers
    execution["max_chunk_attempts"] = max_chunk_attempts
    execution["skip_covered_chunks"] = bool(skip_covered_chunks)
    _atomic_write_json(manifest_path, manifest)

    started_monotonic = time.monotonic()
    started_this_run = 0
    failed_this_run = False
    retry_after_monotonic: float | None = None
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)

    chunks = cast(list[dict[str, Any]], manifest["chunks"])
    _reset_retryable_failed_chunks(chunks=chunks, max_chunk_attempts=max_chunk_attempts)
    inflight: dict[Future[ChunkRunReport], dict[str, Any]] = {}
    active_ranges_by_symbol: dict[str, list[tuple[datetime, datetime]]] = {}

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage04b") as executor:
        while True:
            if retry_after_monotonic is not None:
                now_monotonic = time.monotonic()
                if now_monotonic < retry_after_monotonic and not inflight:
                    time.sleep(retry_after_monotonic - now_monotonic)
                if time.monotonic() >= retry_after_monotonic:
                    retry_after_monotonic = None

            while (
                not failed_this_run
                and retry_after_monotonic is None
                and len(inflight) < workers
                and not _runtime_limit_reached(
                    started_monotonic=started_monotonic,
                    max_runtime_seconds=max_runtime_seconds,
                )
                and (max_chunks is None or started_this_run < max_chunks)
            ):
                chunk = _next_schedulable_chunk(
                    chunks=chunks,
                    active_ranges_by_symbol=active_ranges_by_symbol,
                )
                if chunk is None:
                    break

                skip_check_failed = False
                if skip_covered_chunks:
                    try:
                        covered_chunk = _chunk_has_accepted_canonical_coverage(
                            chunk=chunk,
                            gateway=gateway,
                            database=settings.database,
                        )
                    except Exception as exc:  # noqa: BLE001
                        skip_check_failed = True
                        execution["coverage_skip_check_failures"] = (
                            int(execution.get("coverage_skip_check_failures") or 0) + 1
                        )
                        _append_log(
                            log_jsonl,
                            {
                                "event": "coverage_skip_check_failed",
                                "chunk_id": chunk["chunk_id"],
                                "symbol": chunk["symbol"],
                                "dataset_version": chunk["dataset_version"],
                                "start_utc": chunk["start_utc"],
                                "end_utc": chunk["end_utc"],
                                "error_type": type(exc).__name__,
                                "error_summary": _safe_error_summary(exc),
                                "finished_at_utc": _format_utc(_now_utc()),
                                "workers": workers,
                            },
                        )
                    else:
                        skip_check_failed = not covered_chunk

                if skip_covered_chunks and not skip_check_failed:
                    finished_at = _now_utc()
                    chunk["status"] = "completed"
                    chunk["finished_at_utc"] = _format_utc(finished_at)
                    chunk["rows_read"] = 0
                    chunk["rows_written"] = 0
                    chunk["batches_written"] = 0
                    chunk["skip_reason"] = "accepted_canonical_coverage_exists"
                    execution["chunks_completed"] = (
                        int(execution.get("chunks_completed") or 0) + 1
                    )
                    execution["chunks_skipped_covered"] = (
                        int(execution.get("chunks_skipped_covered") or 0) + 1
                    )
                    execution["updated_at_utc"] = _format_utc(finished_at)
                    started_this_run += 1
                    _append_log(
                        log_jsonl,
                        {
                            "event": "chunk_skipped_covered",
                            "chunk_id": chunk["chunk_id"],
                            "symbol": chunk["symbol"],
                            "dataset_version": chunk["dataset_version"],
                            "start_utc": chunk["start_utc"],
                            "end_utc": chunk["end_utc"],
                            "finished_at_utc": _format_utc(finished_at),
                            "workers": workers,
                        },
                    )
                    _atomic_write_json(manifest_path, manifest)
                    continue

                started_at = _now_utc()
                shard = str(chunk["symbol"])
                chunk_range = _chunk_time_range(chunk)
                chunk["status"] = "running"
                chunk["started_at_utc"] = _format_utc(started_at)
                chunk["coordinator"] = execution["coordinator"]
                chunk["worker_count"] = workers
                chunk["shard_key"] = shard
                chunk["attempts"] = int(chunk.get("attempts") or 0) + 1
                chunk.pop("finished_at_utc", None)
                chunk.pop("error_type", None)
                chunk.pop("error_summary", None)
                execution["updated_at_utc"] = _format_utc(started_at)
                _atomic_write_json(manifest_path, manifest)

                future = executor.submit(
                    _run_chunk,
                    chunk=dict(chunk),
                    use_case_factory=use_case_factory,
                )
                inflight[future] = chunk
                active_ranges_by_symbol.setdefault(shard, []).append(chunk_range)
                started_this_run += 1

            if not inflight:
                break

            done, _pending = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                chunk = inflight.pop(future)
                _remove_active_range(
                    active_ranges_by_symbol=active_ranges_by_symbol,
                    chunk=chunk,
                )
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    failed_at = _now_utc()
                    error_summary = _safe_error_summary(exc)
                    chunk["finished_at_utc"] = _format_utc(failed_at)
                    chunk["error_type"] = type(exc).__name__
                    chunk["error_summary"] = error_summary
                    execution["updated_at_utc"] = _format_utc(failed_at)
                    attempts = int(chunk.get("attempts") or 1)
                    if attempts < max_chunk_attempts:
                        chunk["status"] = "pending"
                        chunk["last_retry_at_utc"] = _format_utc(failed_at)
                        execution["chunks_retried"] = int(execution.get("chunks_retried") or 0) + 1
                        if failure_backoff_s > 0:
                            retry_after_monotonic = max(
                                retry_after_monotonic or 0.0,
                                time.monotonic() + failure_backoff_s,
                            )
                        _append_log(
                            log_jsonl,
                            {
                                "event": "chunk_retry_scheduled",
                                "chunk_id": chunk["chunk_id"],
                                "symbol": chunk["symbol"],
                                "dataset_version": chunk["dataset_version"],
                                "attempts": attempts,
                                "max_chunk_attempts": max_chunk_attempts,
                                "error_type": type(exc).__name__,
                                "error_summary": error_summary,
                                "finished_at_utc": _format_utc(failed_at),
                                "failure_backoff_s": failure_backoff_s,
                                "workers": workers,
                            },
                        )
                        _atomic_write_json(manifest_path, manifest)
                        continue

                    failed_this_run = True
                    chunk["status"] = "failed"
                    execution["status"] = "failed"
                    execution["chunks_failed"] = int(execution.get("chunks_failed") or 0) + 1
                    _append_log(
                        log_jsonl,
                        {
                            "event": "chunk_failed",
                            "chunk_id": chunk["chunk_id"],
                            "symbol": chunk["symbol"],
                            "dataset_version": chunk["dataset_version"],
                            "error_type": type(exc).__name__,
                            "error_summary": error_summary,
                            "finished_at_utc": _format_utc(failed_at),
                            "workers": workers,
                        },
                    )
                    _atomic_write_json(manifest_path, manifest)
                    continue

                chunk["status"] = "completed"
                chunk["finished_at_utc"] = result.finished_at_utc
                chunk["rows_read"] = result.rows_read
                chunk["rows_written"] = result.rows_written
                chunk["batches_written"] = result.batches_written
                execution["chunks_completed"] = int(execution.get("chunks_completed") or 0) + 1
                execution["rows_read"] = int(execution.get("rows_read") or 0) + result.rows_read
                execution["rows_written"] = (
                    int(execution.get("rows_written") or 0) + result.rows_written
                )
                execution["batches_written"] = (
                    int(execution.get("batches_written") or 0) + result.batches_written
                )
                execution["updated_at_utc"] = result.finished_at_utc
                _append_log(
                    log_jsonl,
                    {
                        "event": "chunk_completed",
                        "chunk_id": chunk["chunk_id"],
                        "symbol": chunk["symbol"],
                        "dataset_version": chunk["dataset_version"],
                        "start_utc": chunk["start_utc"],
                        "end_utc": chunk["end_utc"],
                        "rows_read": result.rows_read,
                        "rows_written": result.rows_written,
                        "batches_written": result.batches_written,
                        "started_at_utc": result.started_at_utc,
                        "finished_at_utc": result.finished_at_utc,
                        "workers": workers,
                    },
                )
                _atomic_write_json(manifest_path, manifest)

            if delay_s > 0 and inflight and not failed_this_run:
                time.sleep(delay_s)

    if failed_this_run:
        failed_at = _now_utc()
        execution["status"] = "failed"
        execution["updated_at_utc"] = _format_utc(failed_at)
        execution["pending_chunks"] = len(
            [
                chunk
                for chunk in cast(Sequence[Mapping[str, Any]], manifest["chunks"])
                if chunk.get("status") != "completed"
            ]
        )
        _atomic_write_json(manifest_path, manifest)
        return 1

    remaining = [
        chunk
        for chunk in cast(Sequence[Mapping[str, Any]], manifest["chunks"])
        if chunk.get("status") != "completed"
    ]
    finished = _now_utc()
    execution["updated_at_utc"] = _format_utc(finished)
    if not remaining:
        execution["status"] = "completed"
        execution["finished_at_utc"] = _format_utc(finished)
    else:
        execution["status"] = "paused_with_pending_chunks"
        execution["pending_chunks"] = len(remaining)
    _atomic_write_json(manifest_path, manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "execute":
        return _cmd_execute(args)
    if args.command == "coverage":
        return _cmd_coverage(args)
    raise SystemExit(f"unsupported command: {args.command}")


def _cmd_plan(args: argparse.Namespace) -> int:
    cfg = load_market_data_runtime_config(args.market_config)
    market = next(item for item in cfg.markets if item.market_code == BINANCE_FUTURES_MARKET_CODE)
    stage04a_manifest = _read_json(args.stage04a_manifest)
    exchange_info = (
        _read_json(args.exchange_info_json)
        if args.exchange_info_json is not None
        else _load_exchange_info(base_url=market.rest.base_url, timeout_s=market.rest.timeout_s)
    )
    latest_candle_utc = (
        _parse_utc(args.latest_candle_utc)
        if args.latest_candle_utc is not None
        else _query_latest_candle_utc(market_id=int(market.market_id.value))
    )
    manifest = build_plan_manifest(
        stage04a_manifest=stage04a_manifest,
        exchange_info=exchange_info,
        latest_candle_utc=latest_candle_utc,
        generated_at_utc=_now_utc(),
        chunk_days=int(args.chunk_days),
    )
    _atomic_write_json(args.output_json, manifest)
    print(
        _render_json(
            {
                "plan_manifest": str(args.output_json),
                "chunks_total": manifest["summary"]["chunks_total"],
                "chunks_sha256": manifest["summary"]["chunks_sha256"],
                "active_stage04a_symbol_count": manifest["current_metadata"][
                    "active_stage04a_symbol_count"
                ],
                "stale_stage04a_symbol_count": manifest["current_metadata"][
                    "stale_stage04a_symbol_count"
                ],
            }
        )
    )
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    return execute_manifest(
        manifest_path=args.manifest,
        market_config_path=args.market_config,
        batch_size=int(args.batch_size),
        max_chunks=args.max_chunks,
        max_runtime_seconds=args.max_runtime_seconds,
        delay_s=float(args.delay_s),
        log_jsonl=args.log_jsonl,
        workers=int(args.workers),
        max_chunk_attempts=int(args.max_chunk_attempts),
        failure_backoff_s=float(args.failure_backoff_s),
        skip_covered_chunks=bool(args.skip_covered_chunks),
    )


def _cmd_coverage(args: argparse.Namespace) -> int:
    settings = ClickHouseSettingsLoader(os.environ).load()
    gateway = ClickHouseConnectGateway(_clickhouse_client(settings))
    report = build_coverage_report(
        plan_manifest=_read_json(args.manifest),
        gateway=gateway,
        database=settings.database,
        collected_at_utc=_now_utc(),
    )
    _atomic_write_json(args.output_json, report)
    print(
        _render_json(
            {
                "coverage_report": str(args.output_json),
                "coverage_status": report["coverage_status"],
                "entries_sha256": report["entries_sha256"],
                **cast(Mapping[str, Any], report["summary"]),
            }
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 04B Binance Futures historical backfill planner/executor."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Create a dry-run/resume manifest.")
    plan.add_argument("--stage04a-manifest", type=Path, default=DEFAULT_STAGE04A_MANIFEST)
    plan.add_argument("--market-config", type=Path, default=Path("configs/prod/market_data.yaml"))
    plan.add_argument("--exchange-info-json", type=Path, default=None)
    plan.add_argument("--latest-candle-utc", type=str, default=None)
    plan.add_argument("--chunk-days", type=int, default=7)
    plan.add_argument("--output-json", type=Path, default=DEFAULT_PLAN_JSON)

    execute = sub.add_parser("execute", help="Execute pending chunks from a manifest.")
    execute.add_argument("--manifest", type=Path, default=DEFAULT_PLAN_JSON)
    execute.add_argument(
        "--market-config",
        type=Path,
        default=Path("configs/prod/market_data.yaml"),
    )
    execute.add_argument("--batch-size", type=int, default=10_000)
    execute.add_argument("--max-chunks", type=int, default=None)
    execute.add_argument("--max-runtime-seconds", type=float, default=None)
    execute.add_argument("--delay-s", type=float, default=0.2)
    execute.add_argument("--workers", type=int, default=1)
    execute.add_argument("--max-chunk-attempts", type=int, default=1)
    execute.add_argument("--failure-backoff-s", type=float, default=30.0)
    execute.add_argument("--skip-covered-chunks", action="store_true")
    execute.add_argument("--log-jsonl", type=Path, default=DEFAULT_LOG_JSONL)

    coverage = sub.add_parser("coverage", help="Compute per-symbol/window coverage.")
    coverage.add_argument("--manifest", type=Path, default=DEFAULT_PLAN_JSON)
    coverage.add_argument("--output-json", type=Path, default=DEFAULT_COVERAGE_JSON)
    return parser


def _dataset_source_windows(latest_candle_utc: datetime) -> list[SourceWindow]:
    latest = _floor_minute(latest_candle_utc)
    if latest <= _parse_utc("2025-06-01T00:00:00Z"):
        raise ValueError("latest_candle_utc must be after 2025-06-01T00:00:00Z")
    return [
        SourceWindow(
            dataset_version=DATASET_HF_PERIOD_REBUILD,
            source_start=_parse_utc("2020-01-13T22:30:00Z"),
            source_end=_parse_utc("2025-06-01T01:00:00Z"),
        ),
        SourceWindow(
            dataset_version=DATASET_POST_HF_EXTENSION,
            source_start=_parse_utc("2025-05-31T22:30:00Z"),
            source_end=latest + timedelta(minutes=1),
        ),
    ]


def _source_lower_bounds_by_symbol(stage04a_manifest: Mapping[str, Any]) -> dict[str, datetime]:
    rows = stage04a_manifest.get("accepted_symbol_source_windows")
    if not isinstance(rows, Sequence):
        raise ValueError("Stage 04A manifest missing accepted_symbol_source_windows")
    out: dict[str, datetime] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        source_lower_bound = row.get("source_lower_bound_utc")
        if not symbol or not isinstance(source_lower_bound, str):
            continue
        out[symbol] = _parse_utc(source_lower_bound)
    accepted_symbols = stage04a_manifest.get("accepted_symbols")
    if isinstance(accepted_symbols, Sequence):
        missing = sorted(
            str(symbol).strip().upper()
            for symbol in accepted_symbols
            if str(symbol).strip().upper() not in out
        )
        if missing:
            raise ValueError(f"Stage 04A manifest missing source lower bounds: {missing[:5]}")
    return out


def _current_trading_usdt_perpetual_symbols(exchange_info: Mapping[str, Any]) -> set[str]:
    rows = exchange_info.get("symbols")
    if not isinstance(rows, list):
        raise ValueError("Binance exchangeInfo payload is missing symbols list")
    out: set[str] = set()
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        if item.get("status") != "TRADING":
            continue
        if item.get("contractType") != "PERPETUAL":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if symbol:
            out.add(symbol)
    return out


def _chunk_record(
    *,
    market_id: int,
    symbol: str,
    dataset_version: str,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    chunk_id = _chunk_id(
        market_id=market_id,
        symbol=symbol,
        dataset_version=dataset_version,
        start=start,
        end=end,
    )
    return {
        "chunk_id": chunk_id,
        "market_id": market_id,
        "symbol": symbol,
        "dataset_version": dataset_version,
        "start_utc": _format_utc(start),
        "end_utc": _format_utc(end),
        "expected_minutes": _minute_count(start, end),
    }


def _chunk_id(
    *,
    market_id: int,
    symbol: str,
    dataset_version: str,
    start: datetime,
    end: datetime,
) -> str:
    raw = f"{market_id}|{symbol}|{dataset_version}|{_format_utc(start)}|{_format_utc(end)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _iter_chunks(
    *,
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> Iterable[tuple[datetime, datetime]]:
    cursor = _floor_minute(start)
    end = _floor_minute(end)
    step = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(end, cursor + step)
        yield cursor, chunk_end
        cursor = chunk_end


def _task_from_chunk(chunk: Mapping[str, Any]) -> RestFillTask:
    return RestFillTask(
        instrument_id=InstrumentId(
            MarketId(int(chunk["market_id"])),
            Symbol(str(chunk["symbol"])),
        ),
        time_range=TimeRange(
            UtcTimestamp(_parse_utc(str(chunk["start_utc"]))),
            UtcTimestamp(_parse_utc(str(chunk["end_utc"]))),
        ),
        reason="stage04b_backfill",
    )


def _make_rest_fill_use_case(
    *,
    cfg: Any,
    gateway: ClickHouseGateway,
    database: str,
    batch_size: int,
    ingest_id: Any,
) -> RestFillRange1mUseCase:
    clock = SystemClock()
    source = RestCandleIngestSource(
        cfg=cfg,
        clock=clock,
        http=RequestsHttpClient(),
        ingest_id=ingest_id,
    )
    return RestFillRange1mUseCase(
        source=source,
        writer=ClickHouseRawKlineWriter(gateway=gateway, database=database),
        clock=clock,
        max_days_per_insert=cfg.backfill.max_days_per_insert,
        batch_size=batch_size,
        index_reader=ClickHouseCanonicalCandleIndexReader(
            gateway=gateway,
            database=database,
        ),
    )


def _run_chunk(
    *,
    chunk: Mapping[str, Any],
    use_case_factory: Callable[[], RestFillRange1mUseCase],
) -> ChunkRunReport:
    started_at = _now_utc()
    result = use_case_factory().run(_task_from_chunk(chunk))
    finished_at = _now_utc()
    return ChunkRunReport(
        chunk_id=str(chunk["chunk_id"]),
        rows_read=result.rows_read,
        rows_written=result.rows_written,
        batches_written=result.batches_written,
        started_at_utc=_format_utc(started_at),
        finished_at_utc=_format_utc(finished_at),
    )


def _chunk_has_accepted_canonical_coverage(
    *,
    chunk: Mapping[str, Any],
    gateway: ClickHouseGateway,
    database: str,
) -> bool:
    start = _parse_utc(str(chunk["start_utc"]))
    end = _parse_utc(str(chunk["end_utc"]))
    entry = coverage_entry_from_row(
        symbol=str(chunk["symbol"]),
        dataset_version=str(chunk["dataset_version"]),
        start=start,
        end=end,
        row=_query_coverage_row(
            gateway=gateway,
            database=database,
            market_id=int(chunk["market_id"]),
            symbol=str(chunk["symbol"]),
            start=start,
            end=end,
        ),
    )
    return (
        entry["missing_minutes"] == 0
        and entry["duplicate_rows"] == 0
        and entry["volume_quote_rows"] == entry["physical_rows"]
        and entry["trades_count_rows"] == entry["physical_rows"]
        and entry["vwap_computable_rows"] + entry["zero_volume_rows"]
        == entry["physical_rows"]
    )


def _next_schedulable_chunk(
    *,
    chunks: Sequence[dict[str, Any]],
    active_ranges_by_symbol: Mapping[str, Sequence[tuple[datetime, datetime]]],
) -> dict[str, Any] | None:
    for chunk in chunks:
        if chunk.get("status") == "completed":
            continue
        if chunk.get("status") == "failed":
            continue
        if _overlaps_active_range(
            chunk=chunk,
            active_ranges_by_symbol=active_ranges_by_symbol,
        ):
            continue
        return chunk
    return None


def _chunk_time_range(chunk: Mapping[str, Any]) -> tuple[datetime, datetime]:
    return _parse_utc(str(chunk["start_utc"])), _parse_utc(str(chunk["end_utc"]))


def _overlaps_active_range(
    *,
    chunk: Mapping[str, Any],
    active_ranges_by_symbol: Mapping[str, Sequence[tuple[datetime, datetime]]],
) -> bool:
    symbol = str(chunk["symbol"])
    active_ranges = active_ranges_by_symbol.get(symbol, ())
    if not active_ranges:
        return False
    start, end = _chunk_time_range(chunk)
    return any(
        start < active_end and end > active_start
        for active_start, active_end in active_ranges
    )


def _remove_active_range(
    *,
    active_ranges_by_symbol: dict[str, list[tuple[datetime, datetime]]],
    chunk: Mapping[str, Any],
) -> None:
    symbol = str(chunk["symbol"])
    active_ranges = active_ranges_by_symbol.get(symbol)
    if not active_ranges:
        return
    chunk_range = _chunk_time_range(chunk)
    try:
        active_ranges.remove(chunk_range)
    except ValueError:
        return
    if not active_ranges:
        active_ranges_by_symbol.pop(symbol, None)


def _reset_retryable_failed_chunks(
    *,
    chunks: Sequence[dict[str, Any]],
    max_chunk_attempts: int,
) -> None:
    for chunk in chunks:
        if chunk.get("status") != "failed":
            continue
        if int(chunk.get("attempts") or 0) >= max_chunk_attempts:
            continue
        chunk["status"] = "pending"
        chunk["retry_reason"] = "retryable_previous_failure"


def _runtime_limit_reached(
    *,
    started_monotonic: float,
    max_runtime_seconds: float | None,
) -> bool:
    return (
        max_runtime_seconds is not None
        and time.monotonic() - started_monotonic >= max_runtime_seconds
    )


def _query_latest_candle_utc(*, market_id: int) -> datetime:
    settings = ClickHouseSettingsLoader(os.environ).load()
    gateway = ClickHouseConnectGateway(_clickhouse_client(settings))
    query = f"""
    SELECT max(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS last_minute_key
    FROM {settings.database}.canonical_candles_1m
    WHERE market_id = %(market_id)s
    """
    rows = gateway.select(query, {"market_id": market_id})
    if not rows or rows[0].get("last_minute_key") is None:
        raise RuntimeError(f"No canonical candles found for market_id={market_id}")
    return _minute_key_to_dt(rows[0]["last_minute_key"])


def _safe_error_summary(exc: Exception) -> str:
    text = str(exc).splitlines()[0].strip()
    if not text:
        return type(exc).__name__
    if text.startswith("Invalid Binance kline item"):
        return "Invalid Binance kline item"
    if text.startswith("Invalid Bybit kline item"):
        return "Invalid Bybit kline item"
    for marker in (" body=", " params=", " item:", " body:"):
        index = text.find(marker)
        if index >= 0:
            text = text[:index].strip()
    return text[:240]


def _query_coverage_row(
    *,
    gateway: ClickHouseGateway,
    database: str,
    market_id: int,
    symbol: str,
    start: datetime,
    end: datetime,
) -> Mapping[str, Any]:
    query = f"""
    SELECT
        min(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS first_minute_key,
        max(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS last_minute_key,
        count() AS physical_rows,
        uniqExact(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS distinct_minutes,
        countIf(volume_quote IS NOT NULL) AS volume_quote_rows,
        countIf(trades_count IS NOT NULL) AS trades_count_rows,
        countIf(volume_base = 0) AS zero_volume_rows,
        countIf(volume_base > 0 AND volume_quote IS NOT NULL) AS vwap_computable_rows
    FROM {database}.canonical_candles_1m
    WHERE market_id = %(market_id)s
      AND symbol = %(symbol)s
      AND ts_open >= fromUnixTimestamp64Milli(%(start_ms)s, 'UTC')
      AND ts_open < fromUnixTimestamp64Milli(%(end_ms)s, 'UTC')
    SETTINGS max_threads = 1
    """
    rows = gateway.select(
        query,
        {
            "market_id": market_id,
            "symbol": symbol,
            "start_ms": _dt_to_epoch_ms(start),
            "end_ms": _dt_to_epoch_ms(end),
        },
    )
    return rows[0] if rows else {}


def _load_exchange_info(*, base_url: str, timeout_s: float) -> Mapping[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/fapi/v1/exchangeInfo",
        headers={"User-Agent": "roehub-stage04b-backfill/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return cast(Mapping[str, Any], json.load(response))


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_render_json(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_log(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _render_json(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _hash_lines(items: Iterable[str]) -> str:
    text = "\n".join(items)
    if text:
        text += "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(UTC).replace(microsecond=0)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(second=value.second, microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _floor_minute(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(second=0, microsecond=0)


def _minute_count(start: datetime, end: datetime) -> int:
    return int((_floor_minute(end) - _floor_minute(start)).total_seconds() // 60)


def _dt_to_epoch_ms(value: datetime) -> int:
    return int(value.astimezone(UTC).timestamp() * 1000)


def _minute_key_to_dt(value: Any) -> datetime:
    return datetime.fromtimestamp(int(value) * 60, tz=UTC)


def _now_utc() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


if __name__ == "__main__":
    raise SystemExit(main())
