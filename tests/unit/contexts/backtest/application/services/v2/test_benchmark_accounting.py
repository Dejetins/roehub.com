from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.backtest.run_api_runner_benchmark_parity import (
    _INSTRUMENTATION_COUNTER_FIELDS,
    _ensure_artifact_runtime_env,
    _ensure_postgres_dsn_env,
    _instrumentation_summary,
    _load_runtime_env_file,
    _merged_instrumentation_counters,
    _read_env_file,
)
from trading.contexts.backtest.application.services.v2.benchmark_accounting import (
    CANONICAL_STAGE_ORDER,
    SERVICE_ONLY_TELEMETRY_FIELDS,
    BenchmarkAccountingError,
    build_benchmark_accounting_record,
    normalize_canonical_timers,
    validate_canonical_benchmark_json,
)

CANONICAL_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
)


def test_canonical_json_stage_accounting_normalizes_notebook_aliases() -> None:
    summary = validate_canonical_benchmark_json(CANONICAL_JSON)

    assert summary["run_count"] == 28
    assert summary["request"]["top_n"] == 100
    assert summary["benchmark_top_k"] == 5
    assert summary["sample_warmup_top_k"] == 1
    assert summary["top_results_count_values"] == [5]
    assert summary["heap_capacity"] == 5
    assert summary["canonical_stage_order"] == list(CANONICAL_STAGE_ORDER)
    assert summary["prepare_pools_alias_normalized"] is True
    assert summary["total_alias_is_historical"] is True
    assert summary["service_total_compared_to_canonical"] is False
    assert "prepare_pools" not in summary["stage_presence"]
    assert "total" not in summary["stage_presence"]
    assert "service_total_without_warmup" not in summary["stage_presence"]
    assert "cleanup" not in summary["canonical_stage_order"]
    for service_only_stage in SERVICE_ONLY_TELEMETRY_FIELDS:
        assert service_only_stage not in summary["canonical_stage_order"]


def test_accounting_record_separates_notebook_and_service_totals() -> None:
    record = build_benchmark_accounting_record(
        timers={
            "prepare_pools": 1.0,
            "build_exact_context": 2.0,
            "build_proxy_context": 3.0,
            "combo_iteration": 4.0,
            "proxy_filter": 5.0,
            "self_check": 6.0,
            "exact_scoring": 7.0,
            "heap_update": 8.0,
            "top_result_proxy_fill": 9.0,
            "artifact_context_resolve": 10.0,
            "artifact_array_open": 11.0,
            "request_slice_prepare": 12.0,
            "prepare_pools_total": 13.0,
            "top_result_assembly": 14.0,
            "persist_top_n_io": 15.0,
            "service_total_without_warmup": 100.0,
        },
        risk_mode="none",
        request_top_n=100,
        benchmark_top_k=5,
        sample_warmup_top_k=1,
        top_results_count=5,
        heap_capacity=5,
    )

    assert record["request"]["top_n"] == 100
    assert record["benchmark_top_k"] == 5
    assert record["sample_warmup_top_k"] == 1
    assert record["top_results_count"] == 5
    assert record["heap_capacity"] == 5
    assert record["total_without_warmup"] == pytest.approx(45.0)
    assert record["service_total_without_warmup"] == pytest.approx(100.0)
    assert record["service_total_compared_to_canonical"] is False
    assert "prepare_pools_core" in record["canonical_timers"]
    assert "prepare_pools" not in record["canonical_timers"]
    assert "service_total_without_warmup" not in record["canonical_timers"]
    assert "service_total_without_warmup" in record["service_only_telemetry"]


def test_total_alias_does_not_create_separate_canonical_stage() -> None:
    timers = normalize_canonical_timers(
        {
            "prepare_pools": 1.0,
            "total": 2.0,
            "total_without_warmup": 2.0,
        }
    )

    assert timers == {
        "prepare_pools_core": 1.0,
        "total_without_warmup": 2.0,
    }


def test_accounting_fails_closed_on_unknown_stage() -> None:
    with pytest.raises(BenchmarkAccountingError, match="unknown benchmark stage"):
        normalize_canonical_timers({"prepare_pools": 1.0, "cleanup": 2.0})


def test_accounting_rejects_missing_required_total_target() -> None:
    with pytest.raises(BenchmarkAccountingError, match="missing required"):
        build_benchmark_accounting_record(
            timers={
                "prepare_pools_core": 1.0,
                "build_exact_context": 2.0,
                "build_proxy_context": 3.0,
                "combo_iteration": 4.0,
                "proxy_filter": 5.0,
                "self_check": 6.0,
                "exact_scoring": 7.0,
                "heap_update": 8.0,
                "service_total_without_warmup": 100.0,
            },
            risk_mode="none",
            request_top_n=100,
            benchmark_top_k=5,
            sample_warmup_top_k=1,
            top_results_count=5,
            heap_capacity=5,
        )


def test_stage_01_instrumentation_counters_are_ordered_and_nulls_are_present() -> None:
    instrumentation_counters = dict.fromkeys(_INSTRUMENTATION_COUNTER_FIELDS)
    instrumentation_counters.update(
        {
            "trade_cell_evals_per_sec": None,
            "artifact_load_ms": 12.5,
            "signals_pack_ms": 1.5,
            "signals_pack_bytes": 1152,
            "signals_pack_estimated_peak_bytes": 2448,
            "signals_pack_arrays_released": True,
            "bitset_word_count": 2,
            "bitset_padding_valid": True,
            "bitset_consensus_sample_count": 16,
            "bitset_consensus_sample_mismatches": 0,
            "bitset_consensus_sample_parity": True,
            "combo_iteration_ms": 3.0,
            "proxy_filter_ms": 4.0,
            "exact_scoring_ms": 5.0,
            "tp_sl_exact_scoring_ms": None,
            "top_result_assembly_ms": 6.0,
            "rows_before_prefilter": None,
            "rows_after_prefilter": 426,
            "row_signature_ms": 7.0,
            "unique_rows_after_dedup": 36,
            "duplicate_signal_row_ids": {"ma.ema": [2, 3]},
            "row_signature_collision_count": 0,
            "consensus_signature_count": 2_176_782_336,
            "consensus_signature_mode": "upper_bound_unique_row_product",
            "candidate_upper_bound_after_row_dedup": 2_176_782_336,
            "combo_count_planned": 1000,
            "candidates_after_proxy": 50,
            "exact_candidates": 50,
            "avg_segments_per_candidate": None,
            "avg_trades_per_candidate": None,
            "tp_count": None,
            "sl_count": None,
            "tp_sl_cells": 0,
            "exact_candidates_per_sec": 10.0,
        }
    )
    counters = _merged_instrumentation_counters(
        [{"instrumentation_counters": instrumentation_counters}]
    )

    assert tuple(counters) == _INSTRUMENTATION_COUNTER_FIELDS
    summary = _instrumentation_summary(
        [{"job_name": "none/arity_6/long_only", "instrumentation_counters": counters}]
    )
    row = summary["rows"][0]
    assert summary["pass"] is True
    assert row["missing_fields"] == []
    assert "signals_pack_ms" not in row["null_fields"]
    assert row["counters"]["signals_pack_ms"] == 1.5
    assert row["counters"]["bitset_padding_valid"] == "True"
    assert row["counters"]["bitset_consensus_sample_parity"] == "True"
    assert row["counters"]["unique_rows_after_dedup"] == 36
    assert row["counters"]["duplicate_signal_row_ids"] == "{'ma.ema': [2, 3]}"


def test_api_runner_benchmark_env_file_parser_keeps_conninfo_values(tmp_path: Path) -> None:
    env_file = tmp_path / "roehub.env"
    env_file.write_text(
        "\n".join(
            (
                "# comment",
                "POSTGRES_DB=roehub",
                "export POSTGRES_USER=roe",
                "POSTGRES_PASSWORD='secret value'",
                (
                    'STRATEGY_PG_DSN="host=127.0.0.1 port=5432 '
                    "dbname=${POSTGRES_DB} user=${POSTGRES_USER} "
                    'password=${POSTGRES_PASSWORD}"'
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = _read_env_file(env_file)

    assert parsed["POSTGRES_DB"] == "roehub"
    assert parsed["POSTGRES_USER"] == "roe"
    assert parsed["POSTGRES_PASSWORD"] == "secret value"
    assert parsed["STRATEGY_PG_DSN"] == (
        "host=127.0.0.1 port=5432 dbname=roehub user=roe password=secret value"
    )


def test_api_runner_benchmark_env_file_report_filters_unrelated_secret_key_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for key in (
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "ROEHUB_E2E_BYBIT_MAINNET_TRADE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / "roehub.env"
    env_file.write_text(
        "\n".join(
            (
                "POSTGRES_DB=roehub",
                "POSTGRES_USER=roe",
                "POSTGRES_PASSWORD=secret",
                "ROEHUB_E2E_BYBIT_MAINNET_TRADE_API_KEY=not-reported",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = _load_runtime_env_file(env_file)

    assert report["keys_loaded_count"] == 4
    assert report["keys_loaded"] == ["POSTGRES_DB", "POSTGRES_PASSWORD", "POSTGRES_USER"]


def test_api_runner_benchmark_derives_dsn_env_from_postgres_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "STRATEGY_PG_DSN",
        "POSTGRES_DSN",
        "IDENTITY_PG_DSN",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("POSTGRES_DB", "roehub")
    monkeypatch.setenv("POSTGRES_USER", "roe")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    dsn, filled_keys = _ensure_postgres_dsn_env()

    assert dsn == "host=127.0.0.1 port=5432 dbname=roehub user=roe password=secret"
    assert filled_keys == ["STRATEGY_PG_DSN", "POSTGRES_DSN", "IDENTITY_PG_DSN"]
    assert dsn == os.environ["STRATEGY_PG_DSN"]
    assert dsn == os.environ["POSTGRES_DSN"]
    assert dsn == os.environ["IDENTITY_PG_DSN"]


def test_api_runner_benchmark_defaults_to_prod_artifact_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ROEHUB_BACKTEST_ARTIFACTS_CONFIG", raising=False)
    monkeypatch.delenv("ROEHUB_ENV", raising=False)
    config_path = tmp_path / "configs" / "prod" / "backtest_artifacts.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("version: 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    report = _ensure_artifact_runtime_env()

    assert report["filled_keys"] == ["ROEHUB_ENV", "ROEHUB_BACKTEST_ARTIFACTS_CONFIG"]
    assert report["config_path"] == "configs/prod/backtest_artifacts.yaml"
    assert os.environ["ROEHUB_ENV"] == "prod"
    assert os.environ["ROEHUB_BACKTEST_ARTIFACTS_CONFIG"] == (
        "configs/prod/backtest_artifacts.yaml"
    )
