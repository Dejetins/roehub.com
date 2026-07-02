from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.rl_trading import stage08g_cpu_optuna_calibration as stage08g_optuna


def test_stage08g_optuna_objective_does_not_reward_fewer_trades() -> None:
    assert stage08g_optuna._objective_values(  # noqa: SLF001
        {
            "closed_trades": 0,
            "return_pct_after_costs": 0.0,
            "win_rate": 0.0,
        }
    ) == (0.0, 0.0)
    assert stage08g_optuna._objective_values(  # noqa: SLF001
        {
            "closed_trades": 3316,
            "return_pct_after_costs": 0.1704538,
            "win_rate": 0.55910736,
        }
    ) == (0.1704538, 0.55910736)


def test_stage08g_optuna_selects_trade_sufficient_best_return_trial() -> None:
    zero_trade = Namespace(number=1, values=[0.0, 0.0], params={})
    profitable = Namespace(number=82, values=[0.1704538, 0.55910736], params={})
    records = [
        {
            "trial_number": 1,
            "scorecard": {
                "baseline_delta_net_pnl_after_costs_quote": 0.0,
                "candidate_beats_best_sanity_baseline": False,
                "closed_trades": 0,
                "max_drawdown_pct": 0.0,
                "return_pct_after_costs": 0.0,
                "win_rate": 0.0,
            },
        },
        {
            "trial_number": 82,
            "scorecard": {
                "baseline_delta_net_pnl_after_costs_quote": -414447.21311246,
                "candidate_beats_best_sanity_baseline": False,
                "closed_trades": 3316,
                "max_drawdown_pct": 0.03472486,
                "return_pct_after_costs": 0.1704538,
                "win_rate": 0.55910736,
            },
        },
    ]

    selected = stage08g_optuna._select_best_trial(  # noqa: SLF001
        completed_trials=[zero_trade, profitable],
        trial_records=records,
        min_closed_trades=100,
    )

    assert selected.number == 82


def test_stage08g_optuna_blocks_when_no_trade_sufficient_trial_exists() -> None:
    trial = Namespace(number=1, values=[0.0, 0.0], params={})

    with pytest.raises(stage08g_optuna.Stage08GOptunaError) as exc:
        stage08g_optuna._select_best_trial(  # noqa: SLF001
            completed_trials=[trial],
            trial_records=[
                {
                    "trial_number": 1,
                    "scorecard": {
                        "closed_trades": 0,
                        "return_pct_after_costs": 0.0,
                        "win_rate": 0.0,
                    },
                }
            ],
            min_closed_trades=100,
        )

    assert exc.value.reason == "optuna_no_trade_sufficient_trials"


def test_stage08k_native_final_gate_requires_strict_baseline_and_distribution_checks() -> None:
    args = Namespace(stage_label="08K", min_calibration_closed_trades=100)
    final_scorecard = {
        "action_counts": {"close": 120, "hold": 760, "open_long": 60, "open_short": 60},
        "closed_trades": 120,
        "metrics_by_period": [
            {"net_pnl_after_costs_quote": 60.0, "period": "2026-01"},
            {"net_pnl_after_costs_quote": 40.0, "period": "2026-02"},
            {"net_pnl_after_costs_quote": -10.0, "period": "2026-03"},
        ],
        "metrics_by_volatility_bucket": [
            {"bucket": "low", "net_pnl_after_costs_quote": 25.0},
            {"bucket": "medium", "net_pnl_after_costs_quote": 35.0},
            {"bucket": "high", "net_pnl_after_costs_quote": 30.0},
        ],
        "net_pnl_after_costs_quote": 90.0,
        "stability_by_ticker": [
            {"net_pnl_after_costs_quote": 30.0, "symbol": "BTCUSDT"},
            {"net_pnl_after_costs_quote": 35.0, "symbol": "ETHUSDT"},
            {"net_pnl_after_costs_quote": 25.0, "symbol": "SOLUSDT"},
        ],
    }
    final_manifest = {
        "scorecards": [
            final_scorecard | {
                "policy_kind": "candidate",
                "policy_name": "roehub_native_candidate_filtered_backtest",
            },
            {
                "net_pnl_after_costs_quote": 25.0,
                "policy_kind": "baseline",
                "policy_name": "hold",
            },
        ]
    }

    gate = stage08g_optuna._final_holdout_gate(  # noqa: SLF001
        args=args,
        branch="roehub_native",
        final_scorecard=final_scorecard,
        final_manifest=final_manifest,
    )

    assert gate["stage09_allowed"] is True
    assert gate["blockers"] == []


def test_stage08k_native_final_gate_blocks_baseline_loser() -> None:
    args = Namespace(stage_label="08K", min_calibration_closed_trades=100)
    final_scorecard = {
        "action_counts": {"close": 120, "hold": 760, "open_long": 60, "open_short": 60},
        "closed_trades": 120,
        "metrics_by_period": [
            {"net_pnl_after_costs_quote": 40.0, "period": "2026-01"},
            {"net_pnl_after_costs_quote": 30.0, "period": "2026-02"},
            {"net_pnl_after_costs_quote": 20.0, "period": "2026-03"},
        ],
        "metrics_by_volatility_bucket": [
            {"bucket": "low", "net_pnl_after_costs_quote": 30.0},
            {"bucket": "medium", "net_pnl_after_costs_quote": 30.0},
            {"bucket": "high", "net_pnl_after_costs_quote": 30.0},
        ],
        "net_pnl_after_costs_quote": 90.0,
        "stability_by_ticker": [
            {"net_pnl_after_costs_quote": 30.0, "symbol": "BTCUSDT"},
            {"net_pnl_after_costs_quote": 30.0, "symbol": "ETHUSDT"},
            {"net_pnl_after_costs_quote": 30.0, "symbol": "SOLUSDT"},
        ],
    }
    final_manifest = {
        "scorecards": [
            final_scorecard | {
                "policy_kind": "candidate",
                "policy_name": "roehub_native_candidate_filtered_backtest",
            },
            {
                "net_pnl_after_costs_quote": 120.0,
                "policy_kind": "baseline",
                "policy_name": "simple_recent_return_threshold",
            },
        ]
    }

    gate = stage08g_optuna._final_holdout_gate(  # noqa: SLF001
        args=args,
        branch="roehub_native",
        final_scorecard=final_scorecard,
        final_manifest=final_manifest,
    )

    assert gate["stage09_allowed"] is False
    assert "candidate_does_not_clear_best_sanity_baseline" in gate["blockers"]
