from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts.rl_trading.stage18a_accelerated_monitor_validation import (
    HistoricalSession,
    ScreenedSession,
    _boundary_offsets_ms,
    _p95,
    replay_historical_sessions,
    select_event_enriched_sessions,
)
from trading.contexts.rl_trading.domain import (
    Stage08kArticleSignal,
    Stage08kMonitorDecision,
    Stage08kMonitorPolicyConfig,
)


def test_event_enriched_selection_keeps_natural_actions() -> None:
    rows = [
        _screened(index=index, action_name="open_long" if index < 20 else "hold")
        for index in range(120)
    ]

    selected = select_event_enriched_sessions(
        screened=rows,
        session_count=100,
        natural_open_target=20,
    )

    assert len(selected) == 100
    assert sum(row.action_name == "open_long" for row in selected) == 20
    assert sum(row.action_name == "hold" for row in selected) == 80


def test_historical_replay_pairs_open_close_and_is_idempotent(tmp_path: Path) -> None:
    feature_path = tmp_path / "sessions.f32.npy"
    features = np.ones((50, 150, 7), dtype=np.float32)
    features[:, :, 0] = 100.0
    features[:, :, 1] = 102.0
    features[:, :, 2] = 100.5
    features[:, :, 3] = 99.0
    features[:, :, 4] = 100.0
    features[:, :, 5] = 10.0
    features[:, :, 6] = 42.0
    features[:10, 0, 0] = 101.0
    features[:10, 90, 4] = 101.0
    np.save(feature_path, features)
    selected = tuple(
        _screened(
            index=index,
            action_name="open_long" if index < 10 else "hold",
            feature_path=feature_path,
        )
        for index in range(50)
    )

    result = replay_historical_sessions(
        selected=selected,
        policy=cast(Any, _FakePolicy()),
        run_id="stage18a-unit-test",
    )

    assert result["open_long_count"] == 10
    assert result["valid_close_count"] == 10
    assert result["source_event_count"] == 20
    assert result["duplicate_replay_added_events"] == 0
    assert result["intents"] == 0
    assert result["orders"] == 0
    assert result["virtual_pnl_quote"] > 0.0


def test_boundary_offsets_cover_requested_range_and_p95_is_deterministic() -> None:
    offsets = _boundary_offsets_ms(20)

    assert len(offsets) == 20
    assert offsets[0] == 50.0
    assert offsets[-1] == 500.0
    assert _p95(range(1, 21)) == 19.0


class _FakePolicy:
    model_version_id = "stage08k_roehub_native_best_3e033951"
    policy_config = Stage08kMonitorPolicyConfig()

    def decide(self, candles: tuple[Any, ...]) -> Stage08kMonitorDecision:
        is_open = float(candles[0].open) == 101.0
        return Stage08kMonitorDecision(
            requested_action_id=1 if is_open else 0,
            requested_action_name="open_long" if is_open else "hold",
            action_id=1 if is_open else 0,
            action_name="open_long" if is_open else "hold",
            confidence=0.9 if is_open else 0.0,
            q_values=(0.0, 1.0, 0.0, 0.0),
            feature_hash=f"{'1' if is_open else '0'}" * 64,
            policy_reason="model_action_allowed" if is_open else "model_hold",
            signal=Stage08kArticleSignal(
                eligible=True,
                event_return=0.06,
                volatility_score=0.06,
                contrast_max_abs_return=0.01,
                reason="article_event_eligible",
            ),
        )


def _screened(
    *,
    index: int,
    action_name: str,
    feature_path: Path = Path("/tmp/features.npy"),
) -> ScreenedSession:
    return ScreenedSession(
        session=HistoricalSession(
            signal_time_ms=1_700_000_000_000 + index * 60_000,
            symbol=f"TICKER{index:03d}USDT",
            feature_path=feature_path,
            session_index=index,
        ),
        action_name=action_name,
        requested_action_name=action_name,
        confidence=0.9 if action_name == "open_long" else 0.0,
        volatility_score=0.06,
        policy_reason="model_action_allowed" if action_name == "open_long" else "model_hold",
        decision_latency_ms=1.0,
    )
