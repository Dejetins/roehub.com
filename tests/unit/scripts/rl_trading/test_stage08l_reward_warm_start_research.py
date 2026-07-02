from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.rl_trading import stage08l_reward_warm_start_research as stage08l
from trading.contexts.rl_trading.domain import FEATURE_NAMES_V1, UpstreamAlphaConfig


def test_stage08l_fixed_horizon_bandit_scores_long_and_short_profit() -> None:
    split = Namespace(
        sequences=np.stack(
            [
                _synthetic_session(history_start=95.0, history_end=100.0, future_end=112.0),
                _synthetic_session(history_start=105.0, history_end=100.0, future_end=88.0),
            ]
        ),
        signal_times_utc=("2025-05-01T00:00:00Z", "2025-05-02T00:00:00Z"),
        source_payload={"split_name": "backtest"},
        symbols=("BTCUSDT", "ETHUSDT"),
        volatility_scores=(0.1, 0.2),
    )

    scorecard = stage08l._fixed_horizon_bandit_scorecard(  # noqa: SLF001
        split=split,
        labels=np.asarray([1, 2], dtype=np.int8),
        profile=(30, 10),
        policy_name="supervised_oracle_label_warm_start_contextual_bandit",
        cost_ratio=0.0,
        alpha=UpstreamAlphaConfig(),
    )

    assert scorecard["closed_trades"] == 2
    assert scorecard["net_pnl_after_costs_quote"] > 0.0
    assert scorecard["action_counts"]["open_long"] == 1
    assert scorecard["action_counts"]["open_short"] == 1
    assert scorecard["stability_summary"]["monthly_positive_group_ratio"] == 1.0


def test_stage08l_candidate_path_decision_requires_baseline_and_classifier_win() -> None:
    scorecards = [
        _scorecard("hold_no_trade", "baseline", pnl=0.0),
        _scorecard("deterministic_random_contextual_bandit", "baseline", pnl=10.0),
        _scorecard("simple_recent_return_threshold_contextual_bandit", "baseline", pnl=20.0),
        _scorecard(
            "supervised_oracle_label_warm_start_contextual_bandit",
            "candidate_proxy",
            pnl=30.0,
        ),
    ]
    supervised = {
        "splits": {
            "backtest": {
                "recent_return_baseline": {"balanced_accuracy": 0.55},
                "ridge_past_window_model": {"balanced_accuracy": 0.65},
            }
        }
    }

    decision = stage08l._candidate_path_decision(  # noqa: SLF001
        scorecards=scorecards,
        supervised=supervised,
        min_trades=100,
    )

    assert decision["candidate_path_justified"] is True
    assert decision["stage09_allowed"] is False

    scorecards[-1] = _scorecard(
        "supervised_oracle_label_warm_start_contextual_bandit",
        "candidate_proxy",
        pnl=5.0,
    )
    blocked = stage08l._candidate_path_decision(  # noqa: SLF001
        scorecards=scorecards,
        supervised=supervised,
        min_trades=100,
    )

    assert blocked["candidate_path_justified"] is False
    assert "warm_start_bandit_does_not_clear_best_technical_baseline" in blocked["blockers"]


def test_stage08l_matrix_validation_requires_all_mandatory_surfaces(tmp_path: Path) -> None:
    i2_path = tmp_path / "i2.json"
    i4_path = tmp_path / "i4.json"
    i2_rows = [
        {"surface": surface, "status": "gap"}
        for surface in sorted(stage08l.MANDATORY_08I2_SURFACES)
    ]
    i4_rows: list[dict[str, object]] = [
        {
            "owner_next_stage": "08K",
            "recheck_disposition": "assigned_to_08k",
            "surface": surface,
        }
        for surface in sorted(stage08l.MANDATORY_08I2_SURFACES)
    ]
    for row in i4_rows:
        if row["surface"] == "full_evaluator_backtest_parity":
            row["owner_next_stage"] = None
            row["recheck_disposition"] = "closed_by_08i3"
        if row["surface"] in {"session_extractor_policy", "dataset_geometry_and_distribution"}:
            row["owner_next_stage"] = "08J"
            row["recheck_disposition"] = "assigned_to_08j"
    i2_path.write_text(
        json.dumps(
            {
                "methodology_discrepancy_matrix": i2_rows,
                "stage09_allowed": False,
                "status": "blocked",
            }
        ),
        encoding="utf-8",
    )
    i4_path.write_text(
        json.dumps(
            {
                "08j_allowed": True,
                "08k_allowed": False,
                "methodology_recheck_matrix": i4_rows,
                "stage09_allowed": False,
                "status": "accepted",
            }
        ),
        encoding="utf-8",
    )

    assert stage08l._load_and_validate_i2_matrix(i2_path)["surface_count"] == 8  # noqa: SLF001
    assert stage08l._load_and_validate_i4_matrix(i4_path)["surface_count"] == 8  # noqa: SLF001

    broken_path = tmp_path / "broken.json"
    broken_path.write_text(
        json.dumps(
            {
                "methodology_discrepancy_matrix": i2_rows[:-1],
                "stage09_allowed": False,
                "status": "blocked",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(stage08l.Stage08LResearchError):
        stage08l._load_and_validate_i2_matrix(broken_path)  # noqa: SLF001


def _scorecard(policy_name: str, policy_kind: str, *, pnl: float) -> dict[str, object]:
    return {
        "action_balance": {
            "open_long": 70,
            "open_short": 50,
            "open_side_dominance_share": 70 / 120,
            "open_total": 120,
        },
        "closed_trades": 120,
        "net_pnl_after_costs_quote": pnl,
        "policy_kind": policy_kind,
        "policy_name": policy_name,
        "stability_summary": {
            "monthly_dominance": {"dominance_share": 0.4},
            "monthly_positive_group_ratio": 0.5,
            "ticker_dominance": {"dominance_share": 0.2},
            "ticker_positive_group_ratio": 0.5,
            "volatility_bucket_dominance": {"dominance_share": 0.4},
        },
    }


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
