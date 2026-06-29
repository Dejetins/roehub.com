from __future__ import annotations

from types import SimpleNamespace

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
    zero_trade = SimpleNamespace(number=1, values=[0.0, 0.0], params={})
    profitable = SimpleNamespace(number=82, values=[0.1704538, 0.55910736], params={})
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
    trial = SimpleNamespace(number=1, values=[0.0, 0.0], params={})

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
