from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from trading.contexts.rl_trading.domain.dataset_refresh_manifest import (
    DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1,
    DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1,
    Stage04BArtifactRef,
    build_dataset_refresh_manifest_v1,
    hash_dataset_refresh_payload_v1,
)
from trading.contexts.rl_trading.domain.feature_contract import FEATURE_CONTRACT_HASH_V1


def test_stage04c_manifest_accepts_clean_stage04b_coverage() -> None:
    manifest = build_dataset_refresh_manifest_v1(
        stage04b_source_window_manifest=_stage04b_plan(),
        stage04b_coverage_report=_coverage_report(),
        source_window_artifact=Stage04BArtifactRef(
            path="/opt/roehub/state/rl_trading/stage04b/plan.json",
            sha256="a" * 64,
        ),
        coverage_artifact=Stage04BArtifactRef(
            path="/opt/roehub/state/rl_trading/stage04b/coverage.json",
            sha256="b" * 64,
        ),
        runtime_manifest_path="/opt/roehub/state/rl_trading/stage04c/manifest.json",
        generated_at_utc=datetime(2026, 6, 22, 23, 45, tzinfo=UTC),
    )

    assert manifest["acceptance_status"] == "accepted"
    assert manifest["universe"]["symbols_count"] == 2
    assert manifest["universe"]["hf_membership_filter_applied"] is False
    assert manifest["feature_contract_dependency"]["feature_contract_hash"] == (
        FEATURE_CONTRACT_HASH_V1
    )
    assert manifest["stage05_handoff"]["status"] == "ready"

    by_version = {item["dataset_version"]: item for item in manifest["dataset_versions"]}
    hf_version = by_version[DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1]
    extension_version = by_version[DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1]

    assert hf_version["status"] == "accepted"
    assert hf_version["included_symbols"] == ["BTCUSDT"]
    assert hf_version["excluded_symbols"] == [
        {
            "reason": "listed_after_dataset_source_window",
            "required_source_end_utc": "2025-06-01T01:00:00Z",
            "safe_source_start_utc": "2025-09-17T15:45:00Z",
            "symbol": "NEWUSDT",
        }
    ]
    assert [item["split"] for item in hf_version["signal_windows"]] == [
        "train",
        "validation",
        "test",
        "backtest",
    ]

    assert extension_version["status"] == "accepted"
    assert extension_version["included_symbols"] == ["BTCUSDT", "NEWUSDT"]
    assert extension_version["excluded_symbols"] == []
    assert extension_version["signal_windows"] == [
        {
            "range_semantics": "half-open UTC [start, end)",
            "signal_end_utc": "2026-06-21T14:10:00Z",
            "signal_start_utc": "2025-06-01T00:00:00Z",
            "source_end_utc": "2026-06-21T15:10:00Z",
            "source_start_utc": "2025-05-31T22:30:00Z",
            "split": "post_hf_extension",
        }
    ]


def test_stage04c_manifest_blocks_version_when_coverage_predicate_fails() -> None:
    coverage = _coverage_report()
    coverage["coverage_status"] = "residual_gaps"
    coverage["entries"][0]["missing_minutes"] = 1
    coverage["entries"][0]["distinct_minutes"] = 9

    manifest = build_dataset_refresh_manifest_v1(
        stage04b_source_window_manifest=_stage04b_plan(),
        stage04b_coverage_report=coverage,
        source_window_artifact=Stage04BArtifactRef(path="plan.json", sha256="a" * 64),
        coverage_artifact=Stage04BArtifactRef(path="coverage.json", sha256="b" * 64),
        runtime_manifest_path="manifest.json",
        generated_at_utc=datetime(2026, 6, 22, 23, 45, tzinfo=UTC),
    )

    assert manifest["acceptance_status"] == "partial_rejected"
    by_version = {item["dataset_version"]: item for item in manifest["dataset_versions"]}
    assert by_version[DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1]["status"] == "blocked"
    assert by_version[DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1]["status"] == "accepted"
    assert manifest["stage05_handoff"]["status"] == "blocked"


def test_stage04c_manifest_hash_is_stable_for_same_payload() -> None:
    first = build_dataset_refresh_manifest_v1(
        stage04b_source_window_manifest=_stage04b_plan(),
        stage04b_coverage_report=_coverage_report(),
        source_window_artifact=Stage04BArtifactRef(path="plan.json", sha256="a" * 64),
        coverage_artifact=Stage04BArtifactRef(path="coverage.json", sha256="b" * 64),
        runtime_manifest_path="manifest.json",
        generated_at_utc=datetime(2026, 6, 22, 23, 45, tzinfo=UTC),
    )
    second = build_dataset_refresh_manifest_v1(
        stage04b_source_window_manifest=_stage04b_plan(),
        stage04b_coverage_report=_coverage_report(),
        source_window_artifact=Stage04BArtifactRef(path="plan.json", sha256="a" * 64),
        coverage_artifact=Stage04BArtifactRef(path="coverage.json", sha256="b" * 64),
        runtime_manifest_path="manifest.json",
        generated_at_utc=datetime(2026, 6, 22, 23, 45, tzinfo=UTC),
    )

    assert hash_dataset_refresh_payload_v1(first) == hash_dataset_refresh_payload_v1(second)


def _stage04b_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "04B",
        "market": "binance:futures",
        "market_id": 2,
        "latest_binance_futures_candle_utc": "2026-06-21T15:09:00Z",
        "current_metadata": {
            "active_stage04a_symbol_count": 1,
            "current_trading_usdt_perpetual_symbols_sha256": "symbols-hash",
            "supplement_symbol_count": 1,
        },
        "history_start_probe": {
            "confirmed_symbols_sha256": "confirmed-symbols-hash",
            "confirmed_starts_sha256": "confirmed-starts-hash",
        },
        "summary": {
            "chunks_sha256": "chunks-hash",
            "expected_minutes_total": 30,
        },
        "dataset_versions": [
            {
                "dataset_version": DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1,
                "source_start_utc": "2020-01-13T22:30:00Z",
                "source_end_utc": "2025-06-01T01:00:00Z",
            },
            {
                "dataset_version": DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1,
                "source_start_utc": "2025-05-31T22:30:00Z",
                "source_end_utc": "2026-06-21T15:10:00Z",
            },
        ],
        "symbols": [
            {
                "market_id": 2,
                "symbol": "NEWUSDT",
                "source_lower_bound_utc": "2025-09-17T15:45:00Z",
                "windows": [
                    {
                        "dataset_version": DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1,
                        "required_source_start_utc": "2020-01-13T22:30:00Z",
                        "required_source_end_utc": "2025-06-01T01:00:00Z",
                        "safe_source_start_utc": "2025-09-17T15:45:00Z",
                        "safe_source_end_utc": "2025-06-01T01:00:00Z",
                        "expected_minutes": 0,
                        "status": "empty_after_history_start",
                    },
                    {
                        "dataset_version": DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1,
                        "required_source_start_utc": "2025-05-31T22:30:00Z",
                        "required_source_end_utc": "2026-06-21T15:10:00Z",
                        "safe_source_start_utc": "2025-09-17T15:45:00Z",
                        "safe_source_end_utc": "2026-06-21T15:10:00Z",
                        "expected_minutes": 10,
                        "status": "planned",
                    },
                ],
            },
            {
                "market_id": 2,
                "symbol": "BTCUSDT",
                "source_lower_bound_utc": "2020-01-13T22:30:00Z",
                "windows": [
                    {
                        "dataset_version": DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1,
                        "required_source_start_utc": "2020-01-13T22:30:00Z",
                        "required_source_end_utc": "2025-06-01T01:00:00Z",
                        "safe_source_start_utc": "2020-01-13T22:30:00Z",
                        "safe_source_end_utc": "2025-06-01T01:00:00Z",
                        "expected_minutes": 10,
                        "status": "planned",
                    },
                    {
                        "dataset_version": DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1,
                        "required_source_start_utc": "2025-05-31T22:30:00Z",
                        "required_source_end_utc": "2026-06-21T15:10:00Z",
                        "safe_source_start_utc": "2025-05-31T22:30:00Z",
                        "safe_source_end_utc": "2026-06-21T15:10:00Z",
                        "expected_minutes": 10,
                        "status": "planned",
                    },
                ],
            },
        ],
    }


def _coverage_report() -> dict[str, Any]:
    entries = [
        _coverage_entry(
            symbol="BTCUSDT",
            dataset_version=DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1,
        ),
        _coverage_entry(
            symbol="NEWUSDT",
            dataset_version=DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1,
            start="2025-09-17T15:45:00Z",
            end="2026-06-21T15:10:00Z",
        ),
        _coverage_entry(
            symbol="BTCUSDT",
            dataset_version=DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1,
            start="2025-05-31T22:30:00Z",
            end="2026-06-21T15:10:00Z",
        ),
    ]
    return {
        "schema_version": 1,
        "stage": "04B",
        "market": "binance:futures",
        "coverage_status": "accepted_coverage",
        "latest_binance_futures_candle_utc": "2026-06-21T15:09:00Z",
        "summary": {
            "windows_total": 3,
            "windows_blocked": 0,
            "expected_minutes_total": 30,
            "distinct_minutes_total": 30,
            "missing_minutes_total": 0,
            "duplicate_rows_total": 0,
        },
        "entries": entries,
        "entries_sha256": "coverage-entries-hash",
    }


def _coverage_entry(
    *,
    symbol: str,
    dataset_version: str,
    start: str = "2020-01-13T22:30:00Z",
    end: str = "2025-06-01T01:00:00Z",
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "dataset_version": dataset_version,
        "start_utc": start,
        "end_utc": end,
        "expected_minutes": 10,
        "physical_rows": 10,
        "distinct_minutes": 10,
        "missing_minutes": 0,
        "duplicate_rows": 0,
        "volume_quote_rows": 10,
        "trades_count_rows": 10,
        "zero_volume_rows": 1,
        "vwap_computable_rows": 9,
        "first_candle_utc": start,
        "last_candle_utc": end,
    }
