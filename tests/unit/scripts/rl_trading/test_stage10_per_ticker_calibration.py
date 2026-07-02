from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.rl_trading import stage10_per_ticker_calibration as stage10
from trading.contexts.rl_trading.domain import model_registry as mr
from trading.contexts.rl_trading.domain import per_ticker_calibration as ptc
from trading.contexts.rl_trading.domain.hf_reproducibility import compute_file_sha256


def test_stage10_cli_writes_calibration_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    fixture = _write_stage10_fixture(root, monkeypatch=monkeypatch)

    exit_code = stage10.main(
        [
            "--artifact-root",
            str(root),
            "--output-root",
            str(root / "calibration_packs" / "stage10_per_ticker_calibration_v1"),
            "--run-id",
            "stage10_cli_test",
            "--generated-at-utc",
            "2026-07-02T21:15:00Z",
            "--candidate-summary-path",
            str(fixture["summary_path"]),
            "--expected-candidate-summary-sha256",
            str(fixture["summary_sha256"]),
            "--candidate-manifest-path",
            str(fixture["candidate_manifest_path"]),
            "--expected-candidate-manifest-sha256",
            str(fixture["candidate_manifest_sha256"]),
            "--source-manifest-path",
            str(fixture["source_manifest_path"]),
            "--expected-source-manifest-sha256",
            str(fixture["source_manifest_sha256"]),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "accepted"
    assert payload["accepted_ticker_count"] == 1
    assert Path(payload["calibration_pack_path"]).is_file()


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
                        "stability_by_ticker": [
                            {
                                "net_pnl_after_costs_quote": 1000.0,
                                "positive_ratio": 0.75,
                                "session_count": 30,
                                "symbol": "BTCUSDT",
                            },
                            {
                                "net_pnl_after_costs_quote": -5.0,
                                "positive_ratio": 0.25,
                                "session_count": 3,
                                "symbol": "ETHUSDT",
                            },
                        ],
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
