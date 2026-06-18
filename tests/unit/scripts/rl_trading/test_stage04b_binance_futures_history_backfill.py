from __future__ import annotations

from datetime import UTC, datetime

from scripts.rl_trading.stage04b_binance_futures_history_backfill import (
    DATASET_HF_PERIOD_REBUILD,
    DATASET_POST_HF_EXTENSION,
    STALE_REASON,
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
    assert manifest["current_metadata"]["stale_symbols"] == [
        {"symbol": "OLDUSDT", "reason": STALE_REASON}
    ]
    assert [item["symbol"] for item in manifest["symbols"]] == ["BTCUSDT"]


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
