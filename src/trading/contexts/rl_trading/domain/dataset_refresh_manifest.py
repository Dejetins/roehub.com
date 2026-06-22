from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from .feature_contract import (
    FEATURE_CONTRACT_HASH_V1,
    FEATURE_CONTRACT_ID_V1,
    FEATURE_CONTRACT_VERSION_V1,
    FEATURE_DTYPE_V1,
    FEATURE_NAMES_V1,
)

DATASET_REFRESH_MANIFEST_SCHEMA_VERSION_V1 = 1
DATASET_REFRESH_MANIFEST_KIND_V1 = "rl_trading_dataset_refresh_manifest"
DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1 = "hf_period_rebuild_current_trading"
DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1 = "post_hf_extension_current_trading"
DATASET_REFRESH_RANGE_SEMANTICS_V1 = "half-open UTC [start, end)"

DatasetRefreshStatus = Literal["accepted", "blocked", "partial_rejected"]


class DatasetRefreshManifestError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class Stage04BArtifactRef:
    path: str
    sha256: str

    def as_payload(self) -> dict[str, str]:
        return {
            "path": self.path,
            "sha256": self.sha256,
        }


def build_dataset_refresh_manifest_v1(
    *,
    stage04b_source_window_manifest: Mapping[str, Any],
    stage04b_coverage_report: Mapping[str, Any],
    source_window_artifact: Stage04BArtifactRef,
    coverage_artifact: Stage04BArtifactRef,
    runtime_manifest_path: str,
    generated_at_utc: datetime,
) -> dict[str, Any]:
    """
    Build the Stage 04C dataset refresh manifest from accepted Stage 04B evidence.

    The returned payload is the Stage 05 dataset-builder contract. It does not contain
    raw candles or provider payloads.
    """
    _validate_stage04b_inputs(
        source_window_manifest=stage04b_source_window_manifest,
        coverage_report=stage04b_coverage_report,
    )

    symbols = _symbol_plans(stage04b_source_window_manifest)
    active_symbols = [item["symbol"] for item in symbols]
    coverage_by_key = _coverage_entries_by_key(stage04b_coverage_report)
    source_versions = _source_versions_by_name(stage04b_source_window_manifest)

    dataset_versions = [
        _build_dataset_version_payload(
            dataset_version=DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1,
            source_version=source_versions[DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1],
            symbols=symbols,
            coverage_by_key=coverage_by_key,
            split_windows=_hf_compatible_split_windows_v1(),
            signal_window=None,
        ),
        _build_dataset_version_payload(
            dataset_version=DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1,
            source_version=source_versions[DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1],
            symbols=symbols,
            coverage_by_key=coverage_by_key,
            split_windows=[],
            signal_window=_post_hf_signal_window(
                source_versions[DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1]
            ),
        ),
    ]

    acceptance_status: DatasetRefreshStatus = (
        "accepted"
        if all(item["status"] == "accepted" for item in dataset_versions)
        else "partial_rejected"
    )
    blocked_versions = [
        str(item["dataset_version"])
        for item in dataset_versions
        if item["status"] != "accepted"
    ]
    current_metadata = _required_mapping(stage04b_source_window_manifest, "current_metadata")
    history_start_probe = _required_mapping(stage04b_source_window_manifest, "history_start_probe")
    plan_summary = _required_mapping(stage04b_source_window_manifest, "summary")
    coverage_summary = _required_mapping(stage04b_coverage_report, "summary")

    return {
        "schema_version": DATASET_REFRESH_MANIFEST_SCHEMA_VERSION_V1,
        "manifest_kind": DATASET_REFRESH_MANIFEST_KIND_V1,
        "stage": "04C",
        "acceptance_status": acceptance_status,
        "blocked_dataset_versions": blocked_versions,
        "generated_at_utc": _format_utc(generated_at_utc),
        "runtime_manifest_path": runtime_manifest_path,
        "market": "binance:futures",
        "market_id": int(stage04b_source_window_manifest["market_id"]),
        "range_semantics": DATASET_REFRESH_RANGE_SEMANTICS_V1,
        "universe": {
            "source": "Stage 04B current Binance USD-M Futures TRADING USDT PERPETUAL universe",
            "hf_membership_filter_applied": False,
            "symbols": active_symbols,
            "symbols_count": len(active_symbols),
            "symbols_sha256": _hash_lines(active_symbols),
            "current_symbols_sha256": str(
                current_metadata.get("current_trading_usdt_perpetual_symbols_sha256")
            ),
            "stage04a_reused_symbol_count": int(
                current_metadata.get("active_stage04a_symbol_count") or 0
            ),
            "stage04b_supplement_symbol_count": int(
                current_metadata.get("supplement_symbol_count") or 0
            ),
        },
        "feature_contract_dependency": _feature_contract_dependency_payload(),
        "dataset_versions": dataset_versions,
        "coverage_thresholds": _coverage_thresholds_payload(),
        "lineage": {
            "source_artifacts": {
                "stage04b_source_window_manifest": {
                    **source_window_artifact.as_payload(),
                    "chunks_sha256": str(plan_summary.get("chunks_sha256")),
                    "first_kline_confirmed_symbols_sha256": str(
                        history_start_probe.get("confirmed_symbols_sha256")
                    ),
                    "first_kline_starts_sha256": str(
                        history_start_probe.get("confirmed_starts_sha256")
                    ),
                },
                "stage04b_coverage_report": {
                    **coverage_artifact.as_payload(),
                    "coverage_status": str(stage04b_coverage_report.get("coverage_status")),
                    "entries_sha256": str(stage04b_coverage_report.get("entries_sha256")),
                },
            },
            "source_hashes": {
                "symbol_list_sha256": _hash_lines(active_symbols),
                "dataset_versions_sha256": _hash_json(
                    [
                        {
                            "dataset_version": item["dataset_version"],
                            "status": item["status"],
                            "source_window": item["source_window"],
                            "included_symbols_sha256": item["included_symbols_sha256"],
                            "excluded_symbols_sha256": item["excluded_symbols_sha256"],
                            "coverage_entries_sha256": item["coverage"]["entries_sha256"],
                        }
                        for item in dataset_versions
                    ]
                ),
                "coverage_report_entries_sha256": str(stage04b_coverage_report["entries_sha256"]),
            },
            "query_hashes": {
                "stage04b_coverage_entries_sha256": str(stage04b_coverage_report["entries_sha256"]),
                "stage04b_source_chunks_sha256": str(plan_summary.get("chunks_sha256")),
            },
            "coverage_report_summary": {
                "windows_total": int(coverage_summary.get("windows_total") or 0),
                "windows_blocked": int(coverage_summary.get("windows_blocked") or 0),
                "expected_minutes_total": int(coverage_summary.get("expected_minutes_total") or 0),
                "distinct_minutes_total": int(coverage_summary.get("distinct_minutes_total") or 0),
                "missing_minutes_total": int(coverage_summary.get("missing_minutes_total") or 0),
                "duplicate_rows_total": int(coverage_summary.get("duplicate_rows_total") or 0),
            },
        },
        "stage05_handoff": {
            "input_manifest_path": runtime_manifest_path,
            "input_manifest_sha256_source": "recorded in Stage 04C report after write",
            "status": "ready" if acceptance_status == "accepted" else "blocked",
            "blocked_reason": None
            if acceptance_status == "accepted"
            else "one_or_more_dataset_versions_not_accepted",
            "consumer_contract": (
                "Stage 05 must consume this manifest and must not rediscover "
                "universe/backfill scope."
            ),
        },
        "safety": {
            "contains_raw_candles": False,
            "contains_raw_provider_payloads": False,
            "contains_secrets": False,
            "external_hf_baseline_overwritten": False,
            "market_data_writes": False,
            "exchange_side_effects": False,
        },
    }


def render_dataset_refresh_manifest_json_v1(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def hash_dataset_refresh_payload_v1(payload: Any) -> str:
    return _hash_json(payload)


def _build_dataset_version_payload(
    *,
    dataset_version: str,
    source_version: Mapping[str, Any],
    symbols: Sequence[Mapping[str, Any]],
    coverage_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    split_windows: Sequence[Mapping[str, Any]],
    signal_window: Mapping[str, str] | None,
) -> dict[str, Any]:
    included_symbols: list[str] = []
    excluded_symbols: list[dict[str, str]] = []
    symbol_windows: list[dict[str, Any]] = []
    coverage_entries: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []

    for symbol_plan in symbols:
        symbol = str(symbol_plan["symbol"])
        window = _window_for_dataset_version(symbol_plan, dataset_version)
        expected_minutes = int(window.get("expected_minutes") or 0)
        if expected_minutes <= 0 or window.get("status") == "empty_after_history_start":
            excluded_symbols.append(
                {
                    "reason": "listed_after_dataset_source_window",
                    "required_source_end_utc": str(window["required_source_end_utc"]),
                    "safe_source_start_utc": str(window["safe_source_start_utc"]),
                    "symbol": symbol,
                }
            )
            symbol_windows.append(
                _symbol_window_payload(symbol=symbol, window=window, status="excluded")
            )
            continue

        coverage = coverage_by_key.get((symbol, dataset_version))
        if coverage is None:
            blocking_reasons.append(f"missing_coverage_entry:{symbol}:{dataset_version}")
            symbol_windows.append(
                _symbol_window_payload(symbol=symbol, window=window, status="blocked")
            )
            continue

        coverage_status = _coverage_status(coverage)
        if coverage_status != "accepted":
            blocking_reasons.append(f"coverage_not_accepted:{symbol}:{dataset_version}")
        else:
            included_symbols.append(symbol)
        coverage_entries.append(_coverage_entry_payload(coverage))
        symbol_windows.append(
            _symbol_window_payload(
                symbol=symbol,
                window=window,
                status=coverage_status,
                coverage=coverage,
            )
        )

    included_symbols = sorted(included_symbols)
    excluded_symbols = sorted(excluded_symbols, key=lambda item: item["symbol"])
    symbol_windows = sorted(symbol_windows, key=lambda item: item["symbol"])
    coverage_entries = sorted(
        coverage_entries,
        key=lambda item: (str(item["symbol"]), str(item["start_utc"]), str(item["end_utc"])),
    )
    status: DatasetRefreshStatus
    if blocking_reasons:
        status = "blocked"
    elif not included_symbols:
        status = "blocked"
        blocking_reasons.append("no_included_symbols")
    else:
        status = "accepted"

    return {
        "dataset_version": dataset_version,
        "status": status,
        "blocking_reasons": blocking_reasons,
        "source_window": {
            "start_utc": str(source_version["source_start_utc"]),
            "end_utc": str(source_version["source_end_utc"]),
            "range_semantics": DATASET_REFRESH_RANGE_SEMANTICS_V1,
        },
        "signal_windows": list(split_windows)
        if split_windows
        else [cast(Mapping[str, str], signal_window)],
        "universe_symbols_count": len(symbols),
        "included_symbols": included_symbols,
        "included_symbols_count": len(included_symbols),
        "included_symbols_sha256": _hash_lines(included_symbols),
        "excluded_symbols": excluded_symbols,
        "excluded_symbols_count": len(excluded_symbols),
        "excluded_symbols_sha256": _hash_json(excluded_symbols),
        "symbol_source_windows": symbol_windows,
        "symbol_source_windows_sha256": _hash_json(symbol_windows),
        "coverage": {
            "entries_count": len(coverage_entries),
            "entries_sha256": _hash_json(coverage_entries),
            "windows_blocked": sum(1 for item in coverage_entries if item["status"] != "accepted"),
            "expected_minutes": sum(int(item["expected_minutes"]) for item in coverage_entries),
            "distinct_minutes": sum(int(item["distinct_minutes"]) for item in coverage_entries),
            "missing_minutes": sum(int(item["missing_minutes"]) for item in coverage_entries),
            "duplicate_rows": sum(int(item["duplicate_rows"]) for item in coverage_entries),
            "zero_volume_rows": sum(int(item["zero_volume_rows"]) for item in coverage_entries),
        },
        "feature_contract_dependency": _feature_contract_dependency_payload(),
        "stage05_consumer_status": "ready" if status == "accepted" else "blocked",
    }


def _validate_stage04b_inputs(
    *,
    source_window_manifest: Mapping[str, Any],
    coverage_report: Mapping[str, Any],
) -> None:
    if source_window_manifest.get("stage") != "04B":
        raise DatasetRefreshManifestError(reason="unexpected_source_stage", field="stage")
    if coverage_report.get("stage") != "04B":
        raise DatasetRefreshManifestError(reason="unexpected_coverage_stage", field="stage")
    if source_window_manifest.get("market") != "binance:futures":
        raise DatasetRefreshManifestError(reason="unexpected_source_market", field="market")
    if coverage_report.get("market") != "binance:futures":
        raise DatasetRefreshManifestError(reason="unexpected_coverage_market", field="market")
    if "symbols" not in source_window_manifest:
        raise DatasetRefreshManifestError(reason="missing_symbols", field="symbols")
    if "entries" not in coverage_report:
        raise DatasetRefreshManifestError(reason="missing_coverage_entries", field="entries")

    source_versions = _source_versions_by_name(source_window_manifest)
    for required in (
        DATASET_HF_PERIOD_REBUILD_CURRENT_TRADING_V1,
        DATASET_POST_HF_EXTENSION_CURRENT_TRADING_V1,
    ):
        if required not in source_versions:
            raise DatasetRefreshManifestError(
                reason="missing_dataset_version",
                field=required,
            )


def _symbol_plans(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = manifest.get("symbols")
    if not isinstance(rows, Sequence):
        raise DatasetRefreshManifestError(reason="symbols_not_sequence", field="symbols")
    out = [cast(Mapping[str, Any], item) for item in rows if isinstance(item, Mapping)]
    out.sort(key=lambda item: str(item.get("symbol", "")))
    return out


def _coverage_entries_by_key(report: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = report.get("entries")
    if not isinstance(rows, Sequence):
        raise DatasetRefreshManifestError(reason="coverage_entries_not_sequence", field="entries")
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = (str(row.get("symbol", "")), str(row.get("dataset_version", "")))
        if not key[0] or not key[1]:
            continue
        if key in out:
            raise DatasetRefreshManifestError(reason="duplicate_coverage_entry", field=str(key))
        out[key] = cast(Mapping[str, Any], row)
    return out


def _source_versions_by_name(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("dataset_versions")
    if not isinstance(rows, Sequence):
        raise DatasetRefreshManifestError(
            reason="dataset_versions_not_sequence",
            field="dataset_versions",
        )
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("dataset_version", ""))
        if name:
            out[name] = cast(Mapping[str, Any], row)
    return out


def _window_for_dataset_version(
    symbol_plan: Mapping[str, Any],
    dataset_version: str,
) -> Mapping[str, Any]:
    rows = symbol_plan.get("windows")
    if not isinstance(rows, Sequence):
        raise DatasetRefreshManifestError(reason="missing_symbol_windows", field=str(symbol_plan))
    for row in rows:
        if isinstance(row, Mapping) and row.get("dataset_version") == dataset_version:
            return cast(Mapping[str, Any], row)
    raise DatasetRefreshManifestError(reason="missing_symbol_window", field=dataset_version)


def _coverage_status(entry: Mapping[str, Any]) -> DatasetRefreshStatus:
    expected_minutes = int(entry.get("expected_minutes") or 0)
    distinct_minutes = int(entry.get("distinct_minutes") or 0)
    physical_rows = int(entry.get("physical_rows") or 0)
    missing_minutes = int(entry.get("missing_minutes") or 0)
    duplicate_rows = int(entry.get("duplicate_rows") or 0)
    volume_quote_rows = int(entry.get("volume_quote_rows") or 0)
    trades_count_rows = int(entry.get("trades_count_rows") or 0)
    zero_volume_rows = int(entry.get("zero_volume_rows") or 0)
    vwap_computable_rows = int(entry.get("vwap_computable_rows") or 0)
    if (
        expected_minutes == distinct_minutes
        and missing_minutes == 0
        and duplicate_rows == 0
        and volume_quote_rows == physical_rows
        and trades_count_rows == physical_rows
        and vwap_computable_rows + zero_volume_rows == physical_rows
    ):
        return "accepted"
    return "blocked"


def _coverage_entry_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(entry["symbol"]),
        "dataset_version": str(entry["dataset_version"]),
        "status": _coverage_status(entry),
        "start_utc": str(entry["start_utc"]),
        "end_utc": str(entry["end_utc"]),
        "expected_minutes": int(entry.get("expected_minutes") or 0),
        "physical_rows": int(entry.get("physical_rows") or 0),
        "distinct_minutes": int(entry.get("distinct_minutes") or 0),
        "missing_minutes": int(entry.get("missing_minutes") or 0),
        "duplicate_rows": int(entry.get("duplicate_rows") or 0),
        "volume_quote_rows": int(entry.get("volume_quote_rows") or 0),
        "trades_count_rows": int(entry.get("trades_count_rows") or 0),
        "zero_volume_rows": int(entry.get("zero_volume_rows") or 0),
        "vwap_computable_rows": int(entry.get("vwap_computable_rows") or 0),
        "first_candle_utc": entry.get("first_candle_utc"),
        "last_candle_utc": entry.get("last_candle_utc"),
    }


def _symbol_window_payload(
    *,
    symbol: str,
    window: Mapping[str, Any],
    status: str,
    coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "symbol": symbol,
        "status": status,
        "required_source_start_utc": str(window["required_source_start_utc"]),
        "required_source_end_utc": str(window["required_source_end_utc"]),
        "safe_source_start_utc": str(window["safe_source_start_utc"]),
        "safe_source_end_utc": str(window["safe_source_end_utc"]),
        "expected_minutes": int(window.get("expected_minutes") or 0),
    }
    if coverage is not None:
        out["coverage"] = {
            "distinct_minutes": int(coverage.get("distinct_minutes") or 0),
            "missing_minutes": int(coverage.get("missing_minutes") or 0),
            "duplicate_rows": int(coverage.get("duplicate_rows") or 0),
            "first_candle_utc": coverage.get("first_candle_utc"),
            "last_candle_utc": coverage.get("last_candle_utc"),
        }
    return out


def _hf_compatible_split_windows_v1() -> list[dict[str, str]]:
    return [
        {
            "split": "train",
            "signal_start_utc": "2020-01-14T00:00:00Z",
            "signal_end_utc": "2024-08-31T00:00:00Z",
            "source_start_utc": "2020-01-13T22:30:00Z",
            "source_end_utc": "2024-08-31T01:00:00Z",
            "range_semantics": DATASET_REFRESH_RANGE_SEMANTICS_V1,
        },
        {
            "split": "validation",
            "signal_start_utc": "2024-09-01T00:00:00Z",
            "signal_end_utc": "2024-12-01T00:00:00Z",
            "source_start_utc": "2024-08-31T22:30:00Z",
            "source_end_utc": "2024-12-01T01:00:00Z",
            "range_semantics": DATASET_REFRESH_RANGE_SEMANTICS_V1,
        },
        {
            "split": "test",
            "signal_start_utc": "2024-12-01T00:00:00Z",
            "signal_end_utc": "2025-03-01T00:00:00Z",
            "source_start_utc": "2024-11-30T22:30:00Z",
            "source_end_utc": "2025-03-01T01:00:00Z",
            "range_semantics": DATASET_REFRESH_RANGE_SEMANTICS_V1,
        },
        {
            "split": "backtest",
            "signal_start_utc": "2025-03-01T00:00:00Z",
            "signal_end_utc": "2025-06-01T00:00:00Z",
            "source_start_utc": "2025-02-28T22:30:00Z",
            "source_end_utc": "2025-06-01T01:00:00Z",
            "range_semantics": DATASET_REFRESH_RANGE_SEMANTICS_V1,
        },
    ]


def _post_hf_signal_window(source_version: Mapping[str, Any]) -> dict[str, str]:
    source_end = _parse_utc(str(source_version["source_end_utc"]))
    signal_end = source_end - timedelta(minutes=60)
    return {
        "split": "post_hf_extension",
        "signal_start_utc": "2025-06-01T00:00:00Z",
        "signal_end_utc": _format_utc(signal_end),
        "source_start_utc": str(source_version["source_start_utc"]),
        "source_end_utc": str(source_version["source_end_utc"]),
        "range_semantics": DATASET_REFRESH_RANGE_SEMANTICS_V1,
    }


def _feature_contract_dependency_payload() -> dict[str, Any]:
    return {
        "contract_id": FEATURE_CONTRACT_ID_V1,
        "feature_schema_version": FEATURE_CONTRACT_VERSION_V1,
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "feature_dtype": FEATURE_DTYPE_V1,
        "feature_names": list(FEATURE_NAMES_V1),
        "source": "Stage 02B accepted feature/live-feed contract",
    }


def _coverage_thresholds_payload() -> dict[str, Any]:
    return {
        "missing_minutes": 0,
        "duplicate_rows": 0,
        "volume_quote_rows": "must_equal_physical_rows",
        "trades_count_rows": "must_equal_physical_rows",
        "vwap_computable_rows_plus_zero_volume_rows": "must_equal_physical_rows",
        "fail_behavior": "dataset_version_blocked_or_partial_rejected",
    }


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise DatasetRefreshManifestError(reason="missing_mapping", field=key)
    return cast(Mapping[str, Any], value)


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
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
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
