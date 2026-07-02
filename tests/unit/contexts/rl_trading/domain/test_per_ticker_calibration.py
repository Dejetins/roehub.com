from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trading.contexts.rl_trading.domain import model_registry as mr
from trading.contexts.rl_trading.domain import per_ticker_calibration as ptc
from trading.contexts.rl_trading.domain.hf_reproducibility import compute_file_sha256


def test_stage10_calibration_pack_accepts_only_evidence_backed_tickers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    fixture = _write_stage10_fixture(root, monkeypatch=monkeypatch)

    result = ptc.run_stage10_per_ticker_calibration_v1(
        ptc.Stage10CalibrationConfig(
            artifact_root=root,
            output_root=root / "calibration_packs" / ptc.STAGE10_RUNTIME_ARTIFACT_SUBDIR_V1,
            run_id="stage10_test",
            generated_at_utc=datetime(2026, 7, 2, 21, 0, tzinfo=UTC),
            candidate_summary_path=fixture["summary_path"],
            expected_candidate_summary_sha256=fixture["summary_sha256"],
            candidate_manifest_path=fixture["candidate_manifest_path"],
            expected_candidate_manifest_sha256=fixture["candidate_manifest_sha256"],
            source_manifest_path=fixture["source_manifest_path"],
            expected_source_manifest_sha256=fixture["source_manifest_sha256"],
        )
    )

    assert result["status"] == "accepted"
    assert result["accepted_ticker_count"] == 1
    assert result["blocked_ticker_count"] == 3

    pack = _read_json(Path(str(result["calibration_pack_path"])))
    rows = {row["symbol"]: row for row in pack["ticker_calibrations"]}  # type: ignore[index]
    assert pack["global_policy"]["global_only_threshold_activated"] is False  # type: ignore[index]
    assert pack["normalization_reference"]["raw_values_embedded"] is False  # type: ignore[index]
    assert rows["BTCUSDT"]["status"] == "accepted"
    assert rows["BTCUSDT"]["action_thresholds"]["threshold_mode"] == "ticker_market_calibrated"
    assert rows["BTCUSDT"]["risk_sizing_inputs"]["max_position_fraction_multiplier"] > 0.0
    assert rows["ETHUSDT"]["status"] == "blocked"
    assert rows["ETHUSDT"]["risk_sizing_inputs"]["max_position_fraction_multiplier"] == 0.0
    assert "insufficient_ticker_sessions" in rows["ETHUSDT"]["skipped_action_reasons"]
    assert rows["ACTUSDT"]["status"] == "blocked"
    assert "ticker_positive_ratio_below_minimum" in rows["ACTUSDT"]["skipped_action_reasons"]
    assert rows["DOGEUSDT"]["status"] == "blocked"
    assert "non_positive_ticker_pnl_after_costs" in rows["DOGEUSDT"]["skipped_action_reasons"]

    registry_record = _read_json(Path(str(result["registry_record_path"])))
    calibration = registry_record["calibration_pack"]  # type: ignore[index]
    assert calibration["status"] == "accepted"
    assert calibration["calibration_pack_hash"] == pack["calibration_pack_hash"]
    assert calibration["calibration_sha256"] == compute_file_sha256(
        Path(str(result["calibration_pack_path"]))
    )


def test_stage10_calibration_pack_blocks_when_no_ticker_has_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    fixture = _write_stage10_fixture(
        root,
        monkeypatch=monkeypatch,
        ticker_rows=[
            {
                "net_pnl_after_costs_quote": -1.0,
                "positive_ratio": 0.25,
                "session_count": 2,
                "symbol": "BTCUSDT",
            }
        ],
    )

    result = ptc.run_stage10_per_ticker_calibration_v1(
        ptc.Stage10CalibrationConfig(
            artifact_root=root,
            output_root=root / "calibration_packs" / ptc.STAGE10_RUNTIME_ARTIFACT_SUBDIR_V1,
            run_id="stage10_blocked",
            generated_at_utc=datetime(2026, 7, 2, 21, 5, tzinfo=UTC),
            candidate_summary_path=fixture["summary_path"],
            expected_candidate_summary_sha256=fixture["summary_sha256"],
            candidate_manifest_path=fixture["candidate_manifest_path"],
            expected_candidate_manifest_sha256=fixture["candidate_manifest_sha256"],
            source_manifest_path=fixture["source_manifest_path"],
            expected_source_manifest_sha256=fixture["source_manifest_sha256"],
        )
    )

    assert result["status"] == "blocked"
    registry_record = _read_json(Path(str(result["registry_record_path"])))
    assert registry_record["calibration_pack"]["status"] == "rejected"  # type: ignore[index]


def test_stage10_calibration_validates_lineage_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    fixture = _write_stage10_fixture(root, monkeypatch=monkeypatch)

    with pytest.raises(ptc.Stage10CalibrationError, match="candidate_summary_sha256_mismatch"):
        ptc.run_stage10_per_ticker_calibration_v1(
            ptc.Stage10CalibrationConfig(
                artifact_root=root,
                output_root=root / "calibration_packs" / ptc.STAGE10_RUNTIME_ARTIFACT_SUBDIR_V1,
                run_id="stage10_bad_hash",
                generated_at_utc=datetime(2026, 7, 2, 21, 10, tzinfo=UTC),
                candidate_summary_path=fixture["summary_path"],
                expected_candidate_summary_sha256="0" * 64,
                candidate_manifest_path=fixture["candidate_manifest_path"],
                expected_candidate_manifest_sha256=fixture["candidate_manifest_sha256"],
                source_manifest_path=fixture["source_manifest_path"],
                expected_source_manifest_sha256=fixture["source_manifest_sha256"],
            )
        )


def _patch_artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "rl_trading"
    root.mkdir()
    monkeypatch.setattr(ptc, "RL_TRADING_ARTIFACT_ROOT_V1", str(root))
    monkeypatch.setattr(mr, "RL_TRADING_ARTIFACT_ROOT_V1", str(root))
    return root


def _write_stage10_fixture(
    root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    ticker_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_manifest_path = _write_json(
        root / "datasets" / "stage08j" / "stage08j_article_sessionized_manifest.json",
        {"stage": "08J", "status": "accepted"},
    )
    source_manifest_sha256 = compute_file_sha256(source_manifest_path)
    candidate_manifest_path = _write_json(
        root / "evaluation_runs" / "stage08m" / "candidate_manifest.json",
        {
            "candidate_id": ptc.STAGE09_ACCEPTED_CANDIDATE_ID_V1,
            "data_lineage": {"article_manifest_sha256": source_manifest_sha256},
            "model_state": {
                "feature_count": 2,
                "scaler_mean": [1.0, 2.0],
                "scaler_std": [0.5, 1.5],
            },
            "model_state_hash": "a" * 64,
            "policy_name": ptc.STAGE09_ACCEPTED_CANDIDATE_POLICY_V1,
            "stage": "08M",
            "stage09_allowed": True,
            "status": "accepted_candidate",
        },
    )
    candidate_manifest_sha256 = compute_file_sha256(candidate_manifest_path)
    monkeypatch.setattr(
        ptc,
        "STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1",
        candidate_manifest_sha256,
    )
    rows = ticker_rows or [
        {
            "net_pnl_after_costs_quote": 1000.0,
            "positive_ratio": 0.75,
            "session_count": 30,
            "symbol": "BTCUSDT",
        },
        {
            "net_pnl_after_costs_quote": 100.0,
            "positive_ratio": 0.75,
            "session_count": 3,
            "symbol": "ETHUSDT",
        },
        {
            "net_pnl_after_costs_quote": 500.0,
            "positive_ratio": 0.25,
            "session_count": 20,
            "symbol": "ACTUSDT",
        },
        {
            "net_pnl_after_costs_quote": -10.0,
            "positive_ratio": 0.75,
            "session_count": 20,
            "symbol": "DOGEUSDT",
        },
    ]
    summary_path = _write_json(
        root / "evaluation_runs" / "stage08m" / "summary.json",
        {
            "candidate_artifact": {
                "candidate_id": ptc.STAGE09_ACCEPTED_CANDIDATE_ID_V1,
                "manifest_sha256": candidate_manifest_sha256,
            },
            "comparison": {
                "final_holdout_scorecards": [
                    {
                        "policy_name": ptc.STAGE09_ACCEPTED_CANDIDATE_POLICY_V1,
                        "stability_by_ticker": rows,
                    }
                ]
            },
            "cost_model": {"round_trip_cost_ratio": 0.002},
            "data_quality": {"article_manifest_sha256": source_manifest_sha256},
            "final_holdout_gate": {"blockers": [], "stage09_allowed": True},
            "stage": "08M",
            "stage09_allowed": True,
            "status": "accepted",
            "summary_hash": "b" * 64,
        },
    )
    return {
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "source_manifest_path": source_manifest_path,
        "source_manifest_sha256": source_manifest_sha256,
        "summary_path": summary_path,
        "summary_sha256": compute_file_sha256(summary_path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
