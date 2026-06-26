from __future__ import annotations

import numpy as np

from scripts.rl_trading import stage08h_oracle_supervised_dataset_diagnostics as stage08h
from trading.contexts.rl_trading.domain import FEATURE_NAMES_V1


def test_stage08h_oracle_labels_best_long_and_short_after_costs() -> None:
    sequences = np.stack(
        [
            _synthetic_session(history_start=95.0, history_end=100.0, future_end=112.0),
            _synthetic_session(history_start=105.0, history_end=100.0, future_end=88.0),
        ]
    )

    payload = stage08h._oracle_payload(  # noqa: SLF001
        sequences=sequences,
        profile=(30, 10),
        cost_ratio=0.0,
    )

    assert payload["labels"].tolist() == [1, 2]
    assert payload["best_long_return"][0] > 0.0
    assert payload["best_short_return"][1] > 0.0


def test_stage08h_supervised_sanity_reports_model_and_baselines() -> None:
    long_sessions = [
        _synthetic_session(
            history_start=90.0 + offset,
            history_end=100.0 + offset,
            future_end=112.0 + offset,
        )
        for offset in range(4)
    ]
    short_sessions = [
        _synthetic_session(
            history_start=110.0 + offset,
            history_end=100.0 + offset,
            future_end=88.0 + offset,
        )
        for offset in range(4)
    ]
    train = np.stack([*long_sessions, *short_sessions])
    evaluation = np.stack([long_sessions[0], short_sessions[0]])

    payload = stage08h._supervised_sanity(  # noqa: SLF001
        train_sequences=train,
        eval_splits={"train": train, "test": evaluation},
        profile=(30, 10),
        cost_ratio=0.0,
    )

    assert payload["status"] == "completed"
    assert payload["model"] == "closed_form_ridge_classifier_numpy"
    assert payload["splits"]["test"]["ridge_past_window_model"]["accuracy"] >= 0.5
    assert set(payload["splits"]["test"]["label_counts"]) == {"hold", "long", "short"}


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
