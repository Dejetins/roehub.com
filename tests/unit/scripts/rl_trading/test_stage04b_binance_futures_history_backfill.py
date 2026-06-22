from __future__ import annotations

from datetime import UTC, datetime

from scripts.rl_trading.stage04b_binance_futures_history_backfill import (
    DATASET_HF_PERIOD_REBUILD,
    DATASET_POST_HF_EXTENSION,
    STALE_REASON,
    _build_parser,
    _next_schedulable_chunk,
    _reset_interrupted_running_chunks,
    _reset_retryable_failed_chunks,
    _safe_error_summary,
    build_plan_manifest,
    coverage_entry_from_row,
)


def test_stage04b_plan_uses_source_lower_bound_and_dynamic_extension_end() -> None:
    manifest = build_plan_manifest(
        stage04a_manifest={
            "stage": "04A",
            "market": "binance:futures",
            "market_id": 2,
            "accepted_count": 2,
            "accepted_symbols": ["BTCUSDT", "NEWUSDT"],
            "accepted_symbol_source_windows": [
                {
                    "symbol": "BTCUSDT",
                    "source_lower_bound_utc": "2020-01-13T22:30:00Z",
                },
                {
                    "symbol": "NEWUSDT",
                    "source_lower_bound_utc": "2025-06-02T00:00:00Z",
                },
            ],
        },
        exchange_info={
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "NEWUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                },
            ]
        },
        latest_candle_utc=datetime(2026, 6, 19, 12, 34, 45, tzinfo=UTC),
        generated_at_utc=datetime(2026, 6, 19, 12, 35, tzinfo=UTC),
        chunk_days=7,
    )

    assert manifest["post_hf_extension_source_end_utc"] == "2026-06-19T12:35:00Z"
    assert {item["dataset_version"] for item in manifest["dataset_versions"]} == {
        DATASET_HF_PERIOD_REBUILD,
        DATASET_POST_HF_EXTENSION,
    }
    new_symbol = next(item for item in manifest["symbols"] if item["symbol"] == "NEWUSDT")
    hf_window = next(
        item
        for item in new_symbol["windows"]
        if item["dataset_version"] == DATASET_HF_PERIOD_REBUILD
    )
    extension_window = next(
        item
        for item in new_symbol["windows"]
        if item["dataset_version"] == DATASET_POST_HF_EXTENSION
    )

    assert hf_window["status"] == "empty_after_history_start"
    assert extension_window["safe_source_start_utc"] == "2025-06-02T00:00:00Z"
    assert extension_window["chunk_count"] > 0


def test_stage04b_plan_uses_confirmed_first_kline_after_stage04a_lower_bound() -> None:
    manifest = build_plan_manifest(
        stage04a_manifest={
            "stage": "04A",
            "market": "binance:futures",
            "market_id": 2,
            "accepted_symbols": ["ICPUSDT"],
            "accepted_symbol_source_windows": [
                {
                    "symbol": "ICPUSDT",
                    "source_lower_bound_utc": "2021-07-30T07:00:00Z",
                },
            ],
        },
        exchange_info={
            "symbols": [
                {
                    "symbol": "ICPUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "onboardDate": 1627628400000,
                },
            ]
        },
        history_start_overrides={
            "ICPUSDT": datetime(2022, 9, 27, 2, 30, tzinfo=UTC),
        },
        latest_candle_utc=datetime(2026, 6, 19, 12, 34, 45, tzinfo=UTC),
        generated_at_utc=datetime(2026, 6, 19, 12, 35, tzinfo=UTC),
        chunk_days=7,
    )

    symbol_plan = manifest["symbols"][0]
    assert symbol_plan["source_lower_bound_utc"] == "2022-09-27T02:30:00Z"
    hf_window = next(
        item
        for item in symbol_plan["windows"]
        if item["dataset_version"] == DATASET_HF_PERIOD_REBUILD
    )
    assert hf_window["safe_source_start_utc"] == "2022-09-27T02:30:00Z"
    assert manifest["history_start_probe"]["confirmed_symbol_count"] == 1


def test_stage04b_plan_excludes_stage04a_symbols_missing_from_current_metadata() -> None:
    manifest = build_plan_manifest(
        stage04a_manifest={
            "stage": "04A",
            "market": "binance:futures",
            "market_id": 2,
            "accepted_symbols": ["BTCUSDT", "OLDUSDT"],
            "accepted_symbol_source_windows": [
                {
                    "symbol": "BTCUSDT",
                    "source_lower_bound_utc": "2020-01-13T22:30:00Z",
                },
                {
                    "symbol": "OLDUSDT",
                    "source_lower_bound_utc": "2020-01-13T22:30:00Z",
                },
            ],
        },
        exchange_info={
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "SUPPLEMENTUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "onboardDate": 1767225600000,
                },
                {
                    "symbol": "OLDUSDT",
                    "status": "BREAK",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                },
            ]
        },
        latest_candle_utc=datetime(2026, 6, 19, 12, 34, tzinfo=UTC),
        generated_at_utc=datetime(2026, 6, 19, 12, 35, tzinfo=UTC),
        chunk_days=7,
    )

    assert manifest["current_metadata"]["active_stage04a_symbol_count"] == 1
    assert manifest["current_metadata"]["supplement_symbol_count"] == 1
    assert manifest["summary"]["symbols_planned"] == 2
    assert manifest["summary"]["symbols_supplement_planned"] == 1
    assert manifest["current_metadata"]["stale_symbols"] == [
        {"symbol": "OLDUSDT", "reason": STALE_REASON}
    ]
    assert [item["symbol"] for item in manifest["symbols"]] == [
        "BTCUSDT",
        "SUPPLEMENTUSDT",
    ]
    supplement = next(item for item in manifest["symbols"] if item["symbol"] == "SUPPLEMENTUSDT")
    assert supplement["source_lower_bound_utc"] == "2026-01-01T00:00:00Z"


def test_stage04b_plan_reuses_completed_chunks_from_previous_manifest() -> None:
    previous = build_plan_manifest(
        stage04a_manifest={
            "stage": "04A",
            "market": "binance:futures",
            "market_id": 2,
            "accepted_symbols": ["BTCUSDT"],
            "accepted_symbol_source_windows": [
                {
                    "symbol": "BTCUSDT",
                    "source_lower_bound_utc": "2020-01-13T22:30:00Z",
                },
            ],
        },
        exchange_info={
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                },
            ]
        },
        latest_candle_utc=datetime(2026, 6, 19, 12, 34, tzinfo=UTC),
        generated_at_utc=datetime(2026, 6, 19, 12, 35, tzinfo=UTC),
        chunk_days=31,
    )
    previous["execution"]["status"] = "completed"
    first_chunk = previous["chunks"][0]
    first_chunk.update(
        {
            "status": "completed",
            "rows_read": 44640,
            "rows_written": 44640,
            "batches_written": 5,
            "finished_at_utc": "2026-06-20T00:00:00Z",
        }
    )

    manifest = build_plan_manifest(
        stage04a_manifest={
            "stage": "04A",
            "market": "binance:futures",
            "market_id": 2,
            "accepted_symbols": ["BTCUSDT"],
            "accepted_symbol_source_windows": [
                {
                    "symbol": "BTCUSDT",
                    "source_lower_bound_utc": "2020-01-13T22:30:00Z",
                },
            ],
        },
        exchange_info={
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "NEWUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "onboardDate": 1767225600000,
                },
            ]
        },
        latest_candle_utc=datetime(2026, 6, 19, 12, 34, tzinfo=UTC),
        generated_at_utc=datetime(2026, 6, 19, 12, 35, tzinfo=UTC),
        chunk_days=31,
        previous_manifest=previous,
    )

    reused = [
        chunk
        for chunk in manifest["chunks"]
        if chunk.get("reuse_source") == "previous_stage04b_manifest"
    ]
    assert len(reused) == 1
    assert reused[0]["chunk_id"] == first_chunk["chunk_id"]
    assert reused[0]["rows_written"] == 44640
    assert manifest["summary"]["chunks_reused_completed_from_previous_manifest"] == 1
    assert manifest["execution"]["chunks_completed"] == 1


def test_stage04b_coverage_entry_reports_missing_duplicates_and_vwap_counts() -> None:
    entry = coverage_entry_from_row(
        symbol="BTCUSDT",
        dataset_version=DATASET_POST_HF_EXTENSION,
        start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
        row={
            "first_minute_key": 29452320,
            "last_minute_key": 29452328,
            "physical_rows": 10,
            "distinct_minutes": 9,
            "volume_quote_rows": 10,
            "trades_count_rows": 9,
            "zero_volume_rows": 1,
            "vwap_computable_rows": 8,
        },
    )

    assert entry["expected_minutes"] == 10
    assert entry["missing_minutes"] == 1
    assert entry["duplicate_rows"] == 1
    assert entry["trades_count_rows"] == 9
    assert entry["zero_volume_rows"] == 1
    assert entry["vwap_computable_rows"] == 8


def test_stage04b_error_summary_does_not_keep_raw_response_body() -> None:
    summary = _safe_error_summary(
        RuntimeError(
            "HTTP 418 for https://fapi.binance.com/fapi/v1/klines "
            "params={'symbol': 'BTCUSDT'} body={raw payload}"
        )
    )

    assert summary == "HTTP 418 for https://fapi.binance.com/fapi/v1/klines"


def test_stage04b_execute_workers_default_to_sequential() -> None:
    args = _build_parser().parse_args(["execute"])

    assert args.workers == 1
    assert args.max_chunk_attempts == 1
    assert args.skip_covered_chunks is False


def test_stage04b_execute_accepts_explicit_worker_count() -> None:
    args = _build_parser().parse_args(
        ["execute", "--workers", "4", "--max-chunk-attempts", "3", "--skip-covered-chunks"]
    )

    assert args.workers == 4
    assert args.max_chunk_attempts == 3
    assert args.skip_covered_chunks is True


def test_stage04b_parallel_scheduler_spreads_work_across_symbols() -> None:
    chunks = [
        {
            "chunk_id": "a",
            "symbol": "BTCUSDT",
            "status": "pending",
            "start_utc": "2024-01-08T00:00:00Z",
            "end_utc": "2024-01-15T00:00:00Z",
        },
        {
            "chunk_id": "b",
            "symbol": "ETHUSDT",
            "status": "pending",
            "start_utc": "2024-01-01T00:00:00Z",
            "end_utc": "2024-01-08T00:00:00Z",
        },
    ]

    selected = _next_schedulable_chunk(
        chunks=chunks,
        active_ranges_by_symbol={
            "BTCUSDT": [
                (
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 8, 0, 0, tzinfo=UTC),
                )
            ]
        },
        now_utc=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
    )

    assert selected == chunks[1]


def test_stage04b_parallel_scheduler_blocks_overlapping_same_symbol_chunks() -> None:
    chunks = [
        {
            "chunk_id": "a",
            "symbol": "BTCUSDT",
            "status": "pending",
            "start_utc": "2024-01-07T00:00:00Z",
            "end_utc": "2024-01-14T00:00:00Z",
        },
        {
            "chunk_id": "b",
            "symbol": "ETHUSDT",
            "status": "pending",
            "start_utc": "2024-01-01T00:00:00Z",
            "end_utc": "2024-01-08T00:00:00Z",
        },
    ]

    selected = _next_schedulable_chunk(
        chunks=chunks,
        active_ranges_by_symbol={
            "BTCUSDT": [
                (
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 8, 0, 0, tzinfo=UTC),
                )
            ]
        },
        now_utc=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
    )

    assert selected == chunks[1]


def test_stage04b_parallel_scheduler_skips_chunks_in_retry_cooldown() -> None:
    chunks = [
        {
            "chunk_id": "a",
            "symbol": "BTCUSDT",
            "status": "pending",
            "start_utc": "2024-01-01T00:00:00Z",
            "end_utc": "2024-01-08T00:00:00Z",
            "next_retry_after_utc": "2024-01-01T00:03:00Z",
        },
        {
            "chunk_id": "b",
            "symbol": "ETHUSDT",
            "status": "pending",
            "start_utc": "2024-01-01T00:00:00Z",
            "end_utc": "2024-01-08T00:00:00Z",
        },
    ]

    selected = _next_schedulable_chunk(
        chunks=chunks,
        active_ranges_by_symbol={},
        now_utc=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert selected == chunks[1]


def test_stage04b_parallel_scheduler_ignores_stale_running_chunks() -> None:
    chunks = [
        {
            "chunk_id": "a",
            "symbol": "BTCUSDT",
            "status": "running",
            "start_utc": "2024-01-01T00:00:00Z",
            "end_utc": "2024-01-08T00:00:00Z",
        },
        {
            "chunk_id": "b",
            "symbol": "ETHUSDT",
            "status": "pending",
            "start_utc": "2024-01-01T00:00:00Z",
            "end_utc": "2024-01-08T00:00:00Z",
        },
    ]

    selected = _next_schedulable_chunk(
        chunks=chunks,
        active_ranges_by_symbol={},
        now_utc=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert selected == chunks[1]


def test_stage04b_reset_interrupted_running_chunks_requeues_them() -> None:
    chunks = [
        {
            "chunk_id": "a",
            "symbol": "BTCUSDT",
            "status": "running",
            "started_at_utc": "2024-01-01T00:00:00Z",
            "shard_key": "BTCUSDT",
            "worker_count": 4,
        }
    ]

    _reset_interrupted_running_chunks(chunks=chunks)

    assert chunks[0]["status"] == "pending"
    assert chunks[0]["retry_reason"] == "interrupted_previous_run"
    assert "started_at_utc" not in chunks[0]
    assert "shard_key" not in chunks[0]
    assert "worker_count" not in chunks[0]


def test_stage04b_reset_retryable_failed_chunks_keeps_exhausted_failures_terminal() -> None:
    chunks = [
        {"chunk_id": "a", "symbol": "BTCUSDT", "status": "failed", "attempts": 1},
        {"chunk_id": "b", "symbol": "ETHUSDT", "status": "failed", "attempts": 3},
    ]

    _reset_retryable_failed_chunks(chunks=chunks, max_chunk_attempts=3)

    assert chunks[0]["status"] == "pending"
    assert chunks[0]["retry_reason"] == "retryable_previous_failure"
    assert chunks[1]["status"] == "failed"
