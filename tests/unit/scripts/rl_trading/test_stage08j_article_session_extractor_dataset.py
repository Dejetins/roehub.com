from __future__ import annotations

import numpy as np

from scripts.rl_trading import stage08j_article_session_extractor_dataset as stage08j
from trading.contexts.rl_trading.domain import FEATURE_NAMES_V1, RoehubNativeSplitData


def test_stage08j_split_payload_reports_oracle_and_symbol_month_counts() -> None:
    sequences = np.stack(
        [
            _synthetic_session(history_start=95.0, history_end=100.0, future_end=112.0),
            _synthetic_session(history_start=105.0, history_end=100.0, future_end=88.0),
        ]
    )
    split = RoehubNativeSplitData(
        split_name="backtest",
        sequences=sequences,
        symbols=("BTCUSDT", "ETHUSDT"),
        signal_times_utc=("2025-03-01T00:00:00Z", "2025-03-02T00:00:00Z"),
        source_payload={"split_name": "backtest"},
        volatility_scores=(0.06, 0.07),
    )

    payload = stage08j._split_payload(  # noqa: SLF001
        split=split,
        profile=(30, 10),
        cost_ratio=0.0,
    )

    assert payload["session_count"] == 2
    assert payload["oracle"]["label_counts"] == {"hold": 0, "long": 1, "short": 1}
    assert payload["time_month_counts"] == {"2025-03": 2}
    assert payload["symbol_month_counts"] == {"BTCUSDT|2025-03": 1, "ETHUSDT|2025-03": 1}
    assert payload["range_and_volatility"]["pre_signal_realized_volatility"]["max"] >= 0.0


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
