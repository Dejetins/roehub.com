from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

import numpy as np

from .feature_contract import FEATURE_CONTRACT_HASH_V1, FEATURE_DTYPE_V1, FEATURE_NAMES_V1
from .raw_feature_dataset import (
    BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
    RawFeatureSlab,
    hash_json_payload_v1,
    training_source_gate_payload_v1,
)

SESSIONIZED_DATASET_SCHEMA_VERSION_V1 = 1
SESSIONIZED_DATASET_MANIFEST_KIND_V1 = "rl_trading_sessionized_dataset_manifest"
SESSIONIZED_SPLIT_ARTIFACT_KIND_V1 = "rl_trading_sessionized_split_artifact"
SESSIONIZED_QA_REPORT_KIND_V1 = "rl_trading_sessionized_dataset_qa_report"
SESSIONIZED_RANGE_SEMANTICS_V1 = "half-open UTC [start, end)"
SESSIONIZED_PRE_SIGNAL_LEN_V1 = 90
SESSIONIZED_POST_SIGNAL_LEN_V1 = 60
SESSIONIZED_FULL_SEQ_LEN_V1 = 150
SESSIONIZED_AGENT_HISTORY_LEN_V1 = 30
SESSIONIZED_AGENT_SESSION_LEN_V1 = 10
SESSIONIZED_SIGNAL_STRIDE_MINUTES_V1 = 30
SESSIONIZED_EMBARGO_MINUTES_V1 = SESSIONIZED_FULL_SEQ_LEN_V1
SESSIONIZED_HIGH_VOLATILITY_TOP_FRACTION_V1 = 0.01
SESSIONIZED_MAX_SESSIONS_PER_SYMBOL_SPLIT_V1 = 64
SESSIONIZED_HIGH_VOLATILITY_RULE_V1 = "pre_signal_realized_volatility_plus_range_v1"
SESSIONIZED_POLICY_ID_V1 = "binance_futures_high_volatility_sessions_v1"
SESSIONIZED_MINUTE_MS_V1 = 60_000

SessionizedDatasetStatus = Literal["accepted", "blocked", "partial"]


class SessionizedDatasetError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SessionExtractionPolicy:
    pre_signal_len: int = SESSIONIZED_PRE_SIGNAL_LEN_V1
    post_signal_len: int = SESSIONIZED_POST_SIGNAL_LEN_V1
    signal_stride_minutes: int = SESSIONIZED_SIGNAL_STRIDE_MINUTES_V1
    embargo_minutes: int = SESSIONIZED_EMBARGO_MINUTES_V1
    high_volatility_top_fraction: float = SESSIONIZED_HIGH_VOLATILITY_TOP_FRACTION_V1
    max_sessions_per_symbol_split: int = SESSIONIZED_MAX_SESSIONS_PER_SYMBOL_SPLIT_V1

    def __post_init__(self) -> None:
        if self.pre_signal_len <= 0:
            raise SessionizedDatasetError(reason="invalid_pre_signal_len", field="pre_signal_len")
        if self.post_signal_len <= 0:
            raise SessionizedDatasetError(reason="invalid_post_signal_len", field="post_signal_len")
        if self.signal_stride_minutes <= 0:
            raise SessionizedDatasetError(
                reason="invalid_signal_stride_minutes",
                field="signal_stride_minutes",
            )
        if self.embargo_minutes < self.full_seq_len:
            raise SessionizedDatasetError(reason="invalid_embargo_minutes", field="embargo_minutes")
        if not 0.0 < self.high_volatility_top_fraction <= 1.0:
            raise SessionizedDatasetError(
                reason="invalid_high_volatility_top_fraction",
                field="high_volatility_top_fraction",
            )
        if self.max_sessions_per_symbol_split <= 0:
            raise SessionizedDatasetError(
                reason="invalid_max_sessions_per_symbol_split",
                field="max_sessions_per_symbol_split",
            )

    @property
    def full_seq_len(self) -> int:
        return self.pre_signal_len + self.post_signal_len

    def as_payload(self) -> dict[str, object]:
        return {
            "agent_history_len": SESSIONIZED_AGENT_HISTORY_LEN_V1,
            "agent_session_len": SESSIONIZED_AGENT_SESSION_LEN_V1,
            "embargo_minutes": self.embargo_minutes,
            "full_seq_len": self.full_seq_len,
            "high_volatility_rule": SESSIONIZED_HIGH_VOLATILITY_RULE_V1,
            "high_volatility_top_fraction": self.high_volatility_top_fraction,
            "max_sessions_per_symbol_split": self.max_sessions_per_symbol_split,
            "policy_id": SESSIONIZED_POLICY_ID_V1,
            "post_signal_len": self.post_signal_len,
            "pre_signal_len": self.pre_signal_len,
            "score_uses_post_signal_rows": False,
            "signal_stride_minutes": self.signal_stride_minutes,
        }


@dataclass(frozen=True, slots=True)
class SessionSplitWindow:
    dataset_version: str
    split: str
    signal_start_utc: str
    signal_end_utc: str
    source_start_utc: str
    source_end_utc: str

    def as_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "range_semantics": SESSIONIZED_RANGE_SEMANTICS_V1,
            "signal_end_utc": self.signal_end_utc,
            "signal_start_utc": self.signal_start_utc,
            "source_end_utc": self.source_end_utc,
            "source_start_utc": self.source_start_utc,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class SessionCandidate:
    dataset_version: str
    split: str
    symbol: str
    signal_index: int
    signal_ts_open_ms: int
    session_start_ms: int
    session_end_ms: int
    score_window_start_ms: int
    score_window_end_ms: int
    volatility_score: float
    pre_signal_log_return: float
    pre_signal_realized_volatility: float
    pre_signal_range_ratio: float

    def session_key(self) -> str:
        return "|".join(
            (
                "binance",
                "futures",
                self.symbol,
                f"binance:futures:{self.symbol}",
                _format_utc_from_ms(self.signal_ts_open_ms),
                self.split,
                FEATURE_CONTRACT_HASH_V1,
            )
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "exchange_name": "binance",
            "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
            "instrument_key": f"binance:futures:{self.symbol}",
            "market_type": "futures",
            "score_window_end_utc": _format_utc_from_ms(self.score_window_end_ms),
            "score_window_start_utc": _format_utc_from_ms(self.score_window_start_ms),
            "session_end_utc": _format_utc_from_ms(self.session_end_ms),
            "session_key": self.session_key(),
            "session_start_utc": _format_utc_from_ms(self.session_start_ms),
            "signal_ts_open": _format_utc_from_ms(self.signal_ts_open_ms),
            "split": self.split,
            "symbol": self.symbol,
            "volatility_score": _round_float(self.volatility_score),
        }


def default_session_extraction_policy_v1() -> SessionExtractionPolicy:
    return SessionExtractionPolicy()


def assert_sessionized_trainable_source_v1(*, exchange: str, market_type: str) -> None:
    gate = training_source_gate_payload_v1(exchange=exchange, market_type=market_type)
    if gate["status"] != "trainable":
        raise SessionizedDatasetError(
            reason=BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
            field=f"{gate['exchange']}:{gate['market_type']}",
        )


def session_split_windows_from_stage04c_v1(
    *,
    manifest: Mapping[str, Any],
    dataset_versions: Iterable[str] | None = None,
    splits: Iterable[str] | None = None,
) -> tuple[SessionSplitWindow, ...]:
    _validate_stage04c_session_manifest(manifest)
    allowed_versions = None if dataset_versions is None else {str(v) for v in dataset_versions}
    allowed_splits = None if splits is None else {str(v) for v in splits}

    windows: list[SessionSplitWindow] = []
    for version in _dataset_version_rows(manifest):
        dataset_version = str(version["dataset_version"])
        if allowed_versions is not None and dataset_version not in allowed_versions:
            continue
        if version.get("status") != "accepted":
            raise SessionizedDatasetError(
                reason="dataset_version_not_accepted",
                field=dataset_version,
            )
        for row in _signal_window_rows(version):
            split = str(row["split"])
            if allowed_splits is not None and split not in allowed_splits:
                continue
            windows.append(
                SessionSplitWindow(
                    dataset_version=dataset_version,
                    split=split,
                    signal_start_utc=str(row["signal_start_utc"]),
                    signal_end_utc=str(row["signal_end_utc"]),
                    source_start_utc=str(row["source_start_utc"]),
                    source_end_utc=str(row["source_end_utc"]),
                )
            )

    windows.sort(key=lambda item: (item.dataset_version, _parse_utc_ms(item.signal_start_utc)))
    if allowed_versions is not None:
        found_versions = {item.dataset_version for item in windows}
        missing = sorted(allowed_versions - found_versions)
        if missing:
            raise SessionizedDatasetError(
                reason="requested_dataset_version_not_found",
                field=missing[0],
            )
    if allowed_splits is not None:
        found_splits = {item.split for item in windows}
        missing = sorted(allowed_splits - found_splits)
        if missing:
            raise SessionizedDatasetError(reason="requested_split_not_found", field=missing[0])
    return tuple(windows)


def select_high_volatility_session_candidates_v1(
    *,
    slab: RawFeatureSlab,
    split_window: SessionSplitWindow,
    symbol: str,
    policy: SessionExtractionPolicy | None = None,
) -> tuple[SessionCandidate, ...]:
    selected_policy = default_session_extraction_policy_v1() if policy is None else policy
    _validate_slab_for_sessions(slab=slab, policy=selected_policy)

    signal_start_ms = _parse_utc_ms(split_window.signal_start_utc)
    signal_end_ms = _parse_utc_ms(split_window.signal_end_utc)
    if signal_end_ms <= signal_start_ms:
        raise SessionizedDatasetError(
            reason="invalid_split_signal_window",
            field=split_window.split,
        )

    open_time_ms = slab.open_time_ms
    signal_start_idx = int(np.searchsorted(open_time_ms, signal_start_ms, side="left"))
    signal_end_idx = int(np.searchsorted(open_time_ms, signal_end_ms, side="left"))
    stride_ms = selected_policy.signal_stride_minutes * SESSIONIZED_MINUTE_MS_V1

    candidates: list[SessionCandidate] = []
    for signal_idx in range(signal_start_idx, signal_end_idx):
        signal_ts_ms = int(open_time_ms[signal_idx])
        if (signal_ts_ms - signal_start_ms) % stride_ms != 0:
            continue
        session_start_idx = signal_idx - selected_policy.pre_signal_len
        session_end_idx = signal_idx + selected_policy.post_signal_len
        if session_start_idx < 0 or session_end_idx > slab.row_count():
            continue
        score_payload = _score_pre_signal_window(
            slab.features_f32[session_start_idx:signal_idx, :],
        )
        if score_payload is None:
            continue
        session_start_ms = int(open_time_ms[session_start_idx])
        session_end_ms = int(open_time_ms[session_end_idx - 1]) + SESSIONIZED_MINUTE_MS_V1
        candidates.append(
            SessionCandidate(
                dataset_version=split_window.dataset_version,
                split=split_window.split,
                symbol=symbol.upper(),
                signal_index=signal_idx,
                signal_ts_open_ms=signal_ts_ms,
                session_start_ms=session_start_ms,
                session_end_ms=session_end_ms,
                score_window_start_ms=session_start_ms,
                score_window_end_ms=signal_ts_ms,
                volatility_score=score_payload["volatility_score"],
                pre_signal_log_return=score_payload["pre_signal_log_return"],
                pre_signal_realized_volatility=score_payload["pre_signal_realized_volatility"],
                pre_signal_range_ratio=score_payload["pre_signal_range_ratio"],
            )
        )

    if not candidates:
        return ()
    return _select_top_candidates(candidates, policy=selected_policy)


def materialize_session_features_v1(
    *,
    slab: RawFeatureSlab,
    candidates: Sequence[SessionCandidate],
    policy: SessionExtractionPolicy | None = None,
) -> np.ndarray:
    selected_policy = default_session_extraction_policy_v1() if policy is None else policy
    out = np.empty(
        (len(candidates), selected_policy.full_seq_len, len(FEATURE_NAMES_V1)),
        dtype=np.float32,
    )
    for index, candidate in enumerate(candidates):
        start = candidate.signal_index - selected_policy.pre_signal_len
        end = candidate.signal_index + selected_policy.post_signal_len
        if start < 0 or end > slab.row_count():
            raise SessionizedDatasetError(
                reason="session_candidate_out_of_bounds",
                field=candidate.session_key(),
            )
        out[index, :, :] = slab.features_f32[start:end, :]
    return np.ascontiguousarray(out, dtype=np.float32)


def session_signal_time_array_v1(candidates: Sequence[SessionCandidate]) -> np.ndarray:
    return np.asarray([candidate.signal_ts_open_ms for candidate in candidates], dtype=np.int64)


def session_metadata_payload_v1(candidates: Sequence[SessionCandidate]) -> list[dict[str, object]]:
    return [candidate.as_payload() for candidate in candidates]


def build_gap_report_v1(*, slab: RawFeatureSlab) -> dict[str, object]:
    if slab.row_count() == 0:
        return {
            "status": "blocked",
            "reason": "empty_slab",
            "row_count": 0,
            "gap_count": 0,
            "missing_minutes": 0,
        }
    diffs = np.diff(slab.open_time_ms)
    bad = diffs != SESSIONIZED_MINUTE_MS_V1
    gap_count = int(np.count_nonzero(bad))
    missing_minutes = int(np.sum((diffs[bad] // SESSIONIZED_MINUTE_MS_V1) - 1)) if gap_count else 0
    return {
        "status": "accepted" if gap_count == 0 else "blocked",
        "first_open_utc": _format_utc_from_ms(int(slab.open_time_ms[0])),
        "gap_count": gap_count,
        "last_open_utc": _format_utc_from_ms(int(slab.open_time_ms[-1])),
        "missing_minutes": missing_minutes,
        "row_count": slab.row_count(),
    }


def build_split_artifact_entry_v1(
    *,
    dataset_version: str,
    split: str,
    symbol: str,
    candidates: Sequence[SessionCandidate],
    artifact_files: Mapping[str, Mapping[str, object]],
    gap_report: Mapping[str, object],
    policy: SessionExtractionPolicy | None = None,
) -> dict[str, object]:
    selected_policy = default_session_extraction_policy_v1() if policy is None else policy
    candidate_payload = session_metadata_payload_v1(candidates)
    score_values = [candidate.volatility_score for candidate in candidates]
    return {
        "schema_version": SESSIONIZED_DATASET_SCHEMA_VERSION_V1,
        "artifact_kind": SESSIONIZED_SPLIT_ARTIFACT_KIND_V1,
        "candidate_count": len(candidates),
        "dataset_version": dataset_version,
        "deterministic_rebuild_hash": hash_json_payload_v1(
            {
                "candidates": candidate_payload,
                "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
                "files": dict(artifact_files),
                "policy": selected_policy.as_payload(),
            }
        ),
        "exchange_name": "binance",
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "feature_dtype": FEATURE_DTYPE_V1,
        "feature_names": list(FEATURE_NAMES_V1),
        "files": dict(artifact_files),
        "gap_report": dict(gap_report),
        "market": "binance:futures",
        "market_type": "futures",
        "policy": selected_policy.as_payload(),
        "score_summary": _score_summary_payload(score_values),
        "session_shape": [selected_policy.full_seq_len, len(FEATURE_NAMES_V1)],
        "split": split,
        "symbol": symbol.upper(),
    }


def build_leakage_report_v1(
    *,
    candidates: Sequence[SessionCandidate],
    split_windows: Sequence[SessionSplitWindow],
    policy: SessionExtractionPolicy | None = None,
) -> dict[str, object]:
    selected_policy = default_session_extraction_policy_v1() if policy is None else policy
    cross_split_overlap_violations: list[dict[str, object]] = []
    within_split_overlap_pairs = 0
    lookahead_violations: list[dict[str, object]] = []
    lifecycle_violations: list[dict[str, object]] = []

    for candidate in candidates:
        if candidate.score_window_end_ms > candidate.signal_ts_open_ms:
            lookahead_violations.append(candidate.as_payload())
        if candidate.session_start_ms >= candidate.session_end_ms:
            lifecycle_violations.append(candidate.as_payload())

    by_symbol: dict[tuple[str, str], list[SessionCandidate]] = {}
    for candidate in candidates:
        by_symbol.setdefault((candidate.dataset_version, candidate.symbol), []).append(candidate)

    for rows in by_symbol.values():
        ordered = sorted(rows, key=lambda item: (item.session_start_ms, item.session_end_ms))
        previous: SessionCandidate | None = None
        for current in ordered:
            if previous is not None and previous.session_end_ms > current.session_start_ms:
                if previous.split == current.split:
                    within_split_overlap_pairs += 1
                else:
                    cross_split_overlap_violations.append(
                        {
                            "left": previous.as_payload(),
                            "right": current.as_payload(),
                            "reason": "cross_split_session_window_overlap",
                        }
                    )
            if previous is None or current.session_end_ms > previous.session_end_ms:
                previous = current

    embargo_violations = _split_window_embargo_violations(
        split_windows=split_windows,
        policy=selected_policy,
    )
    status = (
        "accepted"
        if not cross_split_overlap_violations
        and not embargo_violations
        and not lookahead_violations
        and not lifecycle_violations
        else "blocked"
    )
    return {
        "schema_version": SESSIONIZED_DATASET_SCHEMA_VERSION_V1,
        "artifact_kind": SESSIONIZED_QA_REPORT_KIND_V1,
        "cross_split_overlap_violations": cross_split_overlap_violations,
        "cross_split_overlap_violations_count": len(cross_split_overlap_violations),
        "embargo_minutes": selected_policy.embargo_minutes,
        "embargo_violations": embargo_violations,
        "embargo_violations_count": len(embargo_violations),
        "lifecycle_violations": lifecycle_violations,
        "lifecycle_violations_count": len(lifecycle_violations),
        "lookahead_violations": lookahead_violations,
        "lookahead_violations_count": len(lookahead_violations),
        "score_feature_scope": "pre_signal_rows_only",
        "score_uses_post_signal_rows": False,
        "selected_session_count": len(candidates),
        "status": status,
        "within_split_overlap_pairs": within_split_overlap_pairs,
    }


def build_sessionized_dataset_manifest_v1(
    *,
    generated_at_utc: datetime,
    stage04c_manifest_path: str,
    stage04c_manifest_sha256: str,
    output_root: str,
    split_entries: Sequence[Mapping[str, object]],
    leakage_report: Mapping[str, object],
    build_scope: Mapping[str, object],
    policy: SessionExtractionPolicy | None = None,
) -> dict[str, object]:
    selected_policy = default_session_extraction_policy_v1() if policy is None else policy
    entries = sorted(
        (dict(entry) for entry in split_entries),
        key=lambda item: (
            str(item["dataset_version"]),
            str(item["split"]),
            str(item["symbol"]),
        ),
    )
    status: SessionizedDatasetStatus = (
        "accepted" if entries and leakage_report.get("status") == "accepted" else "blocked"
    )
    return {
        "schema_version": SESSIONIZED_DATASET_SCHEMA_VERSION_V1,
        "manifest_kind": SESSIONIZED_DATASET_MANIFEST_KIND_V1,
        "stage": "06",
        "status": status,
        "generated_at_utc": _format_utc(generated_at_utc),
        "market": "binance:futures",
        "stage04c_manifest": {
            "path": stage04c_manifest_path,
            "sha256": stage04c_manifest_sha256,
        },
        "feature_contract_dependency": {
            "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
            "feature_dtype": FEATURE_DTYPE_V1,
            "feature_names": list(FEATURE_NAMES_V1),
            "source": "Stage 02B accepted feature/live-feed contract",
        },
        "policy": selected_policy.as_payload(),
        "build_scope": dict(build_scope),
        "output_root": output_root,
        "split_artifacts": entries,
        "split_artifact_count": len(entries),
        "total_sessions": sum(_int_payload_field(entry, "candidate_count") for entry in entries),
        "leakage_report": dict(leakage_report),
        "deterministic_rebuild_hash": hash_json_payload_v1(
            {
                "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
                "leakage_report": dict(leakage_report),
                "policy": selected_policy.as_payload(),
                "split_artifacts": [
                    {
                        "dataset_version": entry["dataset_version"],
                        "deterministic_rebuild_hash": entry["deterministic_rebuild_hash"],
                        "split": entry["split"],
                        "symbol": entry["symbol"],
                    }
                    for entry in entries
                ],
                "stage04c_manifest_sha256": stage04c_manifest_sha256,
            }
        ),
        "safety": {
            "contains_model_checkpoint": False,
            "contains_raw_provider_payloads": False,
            "contains_secrets": False,
            "exchange_side_effects": False,
            "market_data_writes": False,
            "runtime_artifact_root": "/opt/roehub/state/rl_trading/",
            "score_uses_post_signal_rows": False,
        },
    }


def _validate_stage04c_session_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("stage") != "04C":
        raise SessionizedDatasetError(reason="unexpected_refresh_manifest_stage", field="stage")
    if manifest.get("acceptance_status") != "accepted":
        raise SessionizedDatasetError(
            reason="refresh_manifest_not_accepted",
            field="acceptance_status",
        )
    if manifest.get("market") != "binance:futures":
        raise SessionizedDatasetError(reason="unexpected_refresh_manifest_market", field="market")
    dependency = manifest.get("feature_contract_dependency")
    if not isinstance(dependency, Mapping):
        raise SessionizedDatasetError(reason="missing_feature_contract_dependency")
    if dependency.get("feature_contract_hash") != FEATURE_CONTRACT_HASH_V1:
        raise SessionizedDatasetError(reason="feature_contract_hash_mismatch")


def _dataset_version_rows(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = manifest.get("dataset_versions")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SessionizedDatasetError(reason="dataset_versions_not_sequence")
    return tuple(cast(Mapping[str, Any], row) for row in rows if isinstance(row, Mapping))


def _signal_window_rows(version: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = version.get("signal_windows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SessionizedDatasetError(reason="signal_windows_not_sequence")
    return tuple(cast(Mapping[str, Any], row) for row in rows if isinstance(row, Mapping))


def _validate_slab_for_sessions(*, slab: RawFeatureSlab, policy: SessionExtractionPolicy) -> None:
    if slab.features_f32.dtype != np.dtype(np.float32):
        raise SessionizedDatasetError(reason="invalid_features_dtype", field="features_f32")
    if slab.features_f32.ndim != 2 or slab.features_f32.shape[1] != len(FEATURE_NAMES_V1):
        raise SessionizedDatasetError(reason="invalid_features_shape", field="features_f32")
    if slab.row_count() < policy.full_seq_len:
        raise SessionizedDatasetError(reason="slab_too_short_for_sessions")
    gap_report = build_gap_report_v1(slab=slab)
    if gap_report["status"] != "accepted":
        raise SessionizedDatasetError(reason="slab_has_minute_gaps", field="open_time_ms")


def _score_pre_signal_window(window: np.ndarray) -> dict[str, float] | None:
    close_idx = FEATURE_NAMES_V1.index("close")
    high_idx = FEATURE_NAMES_V1.index("high")
    low_idx = FEATURE_NAMES_V1.index("low")
    close = window[:, close_idx].astype(np.float64)
    if np.any(close <= 0.0):
        return None
    high = window[:, high_idx].astype(np.float64)
    low = window[:, low_idx].astype(np.float64)
    log_close = np.log(close)
    log_returns = np.diff(log_close)
    realized_volatility = float(np.std(log_returns, dtype=np.float64))
    range_ratio = float(np.mean((high - low) / close, dtype=np.float64))
    log_return = float(log_close[-1] - log_close[0])
    volatility_score = realized_volatility + max(range_ratio, 0.0)
    if not math.isfinite(volatility_score):
        return None
    return {
        "pre_signal_log_return": log_return,
        "pre_signal_range_ratio": range_ratio,
        "pre_signal_realized_volatility": realized_volatility,
        "volatility_score": volatility_score,
    }


def _select_top_candidates(
    candidates: Sequence[SessionCandidate],
    *,
    policy: SessionExtractionPolicy,
) -> tuple[SessionCandidate, ...]:
    target_count = max(1, math.ceil(len(candidates) * policy.high_volatility_top_fraction))
    target_count = min(target_count, policy.max_sessions_per_symbol_split)
    ranked = sorted(
        candidates,
        key=lambda item: (-item.volatility_score, item.signal_ts_open_ms, item.symbol),
    )[:target_count]
    ranked.sort(key=lambda item: (item.signal_ts_open_ms, item.symbol))
    return tuple(ranked)


def _split_window_embargo_violations(
    *,
    split_windows: Sequence[SessionSplitWindow],
    policy: SessionExtractionPolicy,
) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    by_version: dict[str, list[SessionSplitWindow]] = {}
    for window in split_windows:
        by_version.setdefault(window.dataset_version, []).append(window)
    embargo_ms = policy.embargo_minutes * SESSIONIZED_MINUTE_MS_V1
    for dataset_version, windows in by_version.items():
        ordered = sorted(windows, key=lambda item: _parse_utc_ms(item.signal_start_utc))
        previous: SessionSplitWindow | None = None
        for current in ordered:
            if previous is not None:
                gap_ms = _parse_utc_ms(current.signal_start_utc) - _parse_utc_ms(
                    previous.signal_end_utc
                )
                if gap_ms < embargo_ms:
                    violations.append(
                        {
                            "dataset_version": dataset_version,
                            "left_split": previous.split,
                            "right_split": current.split,
                            "gap_minutes": gap_ms // SESSIONIZED_MINUTE_MS_V1,
                            "minimum_required_minutes": policy.embargo_minutes,
                            "reason": "split_boundary_embargo_violation",
                        }
                    )
            previous = current
    return violations


def _score_summary_payload(score_values: Sequence[float]) -> dict[str, object]:
    if not score_values:
        return {"count": 0}
    arr = np.asarray(score_values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "max": _round_float(float(np.max(arr))),
        "mean": _round_float(float(np.mean(arr))),
        "min": _round_float(float(np.min(arr))),
        "p50": _round_float(float(np.quantile(arr, 0.5))),
        "p95": _round_float(float(np.quantile(arr, 0.95))),
    }


def _int_payload_field(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int):
        raise SessionizedDatasetError(reason="invalid_integer_payload_field", field=key)
    return value


def _parse_utc_ms(value: str) -> int:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text).astimezone(UTC).replace(microsecond=0)
    return int(parsed.timestamp() * 1000)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _format_utc_from_ms(value: int) -> str:
    return _format_utc(datetime.fromtimestamp(value / 1000, tz=UTC))


def _round_float(value: float) -> float:
    return float(round(value, 12))
