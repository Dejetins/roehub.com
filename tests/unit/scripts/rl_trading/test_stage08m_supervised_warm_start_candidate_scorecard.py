from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.rl_trading import stage08g_cpu_optuna_calibration as stage08g
from scripts.rl_trading import stage08m_supervised_warm_start_candidate_scorecard as stage08m
from trading.contexts.rl_trading.domain import FEATURE_NAMES_V1, UpstreamAlphaConfig


def test_stage08m_scorecard_passes_same_strict_native_gate_when_distribution_is_stable() -> None:
    split = Namespace(
        sequences=np.stack(
            [
                _synthetic_session(history_start=95.0, history_end=100.0, future_end=112.0),
                _synthetic_session(history_start=105.0, history_end=100.0, future_end=88.0),
                _synthetic_session(history_start=90.0, history_end=100.0, future_end=111.0),
            ]
        ),
        signal_times_utc=(
            "2025-05-01T00:00:00Z",
            "2025-06-01T00:00:00Z",
            "2025-07-01T00:00:00Z",
        ),
        source_payload={"split_name": "backtest"},
        split_name="backtest",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        volatility_scores=(0.1, 0.2, 0.3),
    )
    alpha = UpstreamAlphaConfig()
    candidate = stage08m._stage08m_scorecard(  # noqa: SLF001
        split=split,
        labels=np.asarray([1, 2, 1], dtype=np.int8),
        profile=(30, 10),
        policy_name=stage08m.STAGE08M_CANDIDATE_POLICY_NAME,
        cost_ratio=0.0,
        alpha=alpha,
        policy_kind="candidate",
    )
    baseline = stage08m._stage08m_scorecard(  # noqa: SLF001
        split=split,
        labels=np.zeros(3, dtype=np.int8),
        profile=(30, 10),
        policy_name="hold_no_trade",
        cost_ratio=0.0,
        alpha=alpha,
        policy_kind="baseline",
    )

    gate = stage08g._final_holdout_gate(  # noqa: SLF001
        args=Namespace(stage_label="08K", min_calibration_closed_trades=2),
        branch="roehub_native",
        final_scorecard=candidate,
        final_manifest={"scorecards": [candidate, baseline]},
    )

    assert candidate["policy_kind"] == "candidate"
    assert "proxy_surface" not in candidate
    assert gate["stage09_allowed"] is True
    assert gate["blockers"] == []


def test_stage08m_validates_accepted_stage08l_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "stage08l_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "candidate_path_decision": {"candidate_path_justified": True},
                "contract_marker": "reward_research_not_contract_replacement",
                "stage": "08L",
                "stage09_allowed": False,
                "status": "accepted",
                "summary_hash": "abc",
            }
        ),
        encoding="utf-8",
    )

    payload = stage08m._load_and_validate_stage08l_summary(  # noqa: SLF001
        path=summary_path,
        expected_file_sha256=None,
        expected_summary_hash="abc",
    )

    assert payload["status"] == "accepted"
    assert payload["summary_hash"] == "abc"


def test_stage08m_candidate_manifest_keeps_research_contract_marker(tmp_path: Path) -> None:
    supervised = {
        "model_state": {"weights": [[1.0]], "scaler_mean": [0.0], "scaler_std": [1.0]},
        "model_state_hash": "f" * 64,
    }
    manifest = stage08m._candidate_manifest_payload(  # noqa: SLF001
        args=Namespace(
            dataset_version="hf_period_rebuild_current_trading",
            stage08j_manifest_path=tmp_path / "stage08j_manifest.json",
        ),
        backtest_split=Namespace(source_payload={"split_name": "backtest"}),
        generated=stage08m._parse_utc("2026-07-02T17:00:00Z"),  # noqa: SLF001
        manifest_sha256="a" * 64,
        profile=(30, 10),
        run_dir=tmp_path,
        run_id="stage08m_test",
        stage08l_summary={"status": "accepted"},
        supervised=supervised,
        strict_gate={"stage09_allowed": True, "blockers": []},
    )

    assert manifest["contract_marker"] == "reward_research_not_contract_replacement"
    assert manifest["safety"]["model_registry_write"] is False
    assert manifest["status"] == "accepted_candidate"


def _synthetic_session(
    *,
    history_start: float,
    history_end: float,
    future_end: float,
) -> np.ndarray:
    session = np.zeros((150, len(FEATURE_NAMES_V1)), dtype=np.float32)
    close_idx = FEATURE_NAMES_V1.index("close")
    open_idx = FEATURE_NAMES_V1.index("open")
    high_idx = FEATURE_NAMES_V1.index("high")
    low_idx = FEATURE_NAMES_V1.index("low")
    vwap_idx = FEATURE_NAMES_V1.index("volume_weighted_average")
    volume_idx = FEATURE_NAMES_V1.index("volume")
    trades_idx = FEATURE_NAMES_V1.index("num_trades")

    close = np.full(150, history_start, dtype=np.float32)
    close[60:90] = np.linspace(history_start, history_end, 30, dtype=np.float32)
    close[89:99] = np.linspace(history_end, future_end, 10, dtype=np.float32)
    close[99:] = future_end
    session[:, close_idx] = close
    session[:, open_idx] = close
    session[:, high_idx] = close * 1.001
    session[:, low_idx] = close * 0.999
    session[:, vwap_idx] = close
    session[:, volume_idx] = 1000.0
    session[:, trades_idx] = 100.0
    return session
