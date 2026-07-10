from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from scripts.macos.stage18a_rl_monitor_evidence import (
    _payload_hash,
    _validate_previous_summary_payload,
    assess_window,
)

DECISIONS_METRIC = (
    'rl_trading_inference_decisions_total{mode="monitor_only",'
    'outcome="no_intent",reason="monitor_only_no_intent"}'
)
VALID_EXITS_METRIC = (
    'rl_trading_inference_virtual_exits_total{valid="true",' 'reason="virtual_close_after_1m"}'
)


def test_assess_window_accepts_clean_monitor_only_hour() -> None:
    baseline = _snapshot()
    final = deepcopy(baseline)
    final["recorded_at_utc"] = "2026-07-10T07:00:00Z"
    final["metrics"] = {
        **final["metrics"],
        'rl_trading_inference_candles_total{result="processed"}': 61.0,
    }
    final["database_counts"]["source_events"] = 2

    summary = assess_window(baseline=baseline, final=final, minimum_seconds=3_600)

    assert summary["status"] == "accepted"
    assert summary["metric_deltas"]["candles_total"] == 60.0
    assert summary["database_deltas"] == {
        "source_events": 2,
        "open_long_source_events": 0,
        "close_source_events": 0,
        "intents": 0,
        "orders": 0,
    }
    assert summary["stream_deltas"] == {
        "execution.requests.v1": 0,
        "execution.requests.retry.v1": 0,
        "execution.requests.dlq.v1": 0,
    }


def test_assess_window_blocks_side_effect_and_safety_growth() -> None:
    baseline = _snapshot()
    final = deepcopy(baseline)
    final["recorded_at_utc"] = "2026-07-10T07:00:00Z"
    final["database_counts"]["intents"] = 1
    final["stream_lengths"]["execution.requests.v1"] += 1
    final["metrics"] = {
        **final["metrics"],
        'rl_trading_inference_candles_total{result="processed"}': 61.0,
        'rl_trading_inference_safety_breaches_total{reason="test"}': 1.0,
    }

    summary = assess_window(baseline=baseline, final=final, minimum_seconds=3_600)

    assert summary["status"] == "blocked"
    assert summary["checks"]["intents_zero"] is False
    assert summary["checks"]["dispatch_stream_growth_zero"] is False
    assert summary["checks"]["safety_breaches_zero"] is False


def test_assess_window_accepts_five_ticker_day_without_forcing_market_event() -> None:
    baseline = _snapshot(phase="five_ticker_24h", instrument_count=5)
    final = deepcopy(baseline)
    final["recorded_at_utc"] = "2026-07-11T06:00:00Z"
    final["metrics"] = {
        **final["metrics"],
        'rl_trading_inference_candles_total{result="processed"}': 7_176.0,
    }

    summary = assess_window(baseline=baseline, final=final, minimum_seconds=86_400)

    assert summary["status"] == "accepted"
    assert summary["checks"]["decision_observed_when_required"] is True


def test_assess_window_requires_valid_virtual_exit_for_twenty_ticker_week() -> None:
    baseline = _snapshot(phase="twenty_ticker_7d", instrument_count=20)
    final = deepcopy(baseline)
    final["recorded_at_utc"] = "2026-07-17T06:00:00Z"
    final["metrics"] = {
        **final["metrics"],
        'rl_trading_inference_candles_total{result="processed"}': 201_501.0,
        DECISIONS_METRIC: 2.0,
        VALID_EXITS_METRIC: 1.0,
        "rl_trading_inference_virtual_realized_pnl_quote": 12.5,
    }
    final["database_counts"].update(
        source_events=2,
        open_long_source_events=1,
        close_source_events=1,
    )

    summary = assess_window(baseline=baseline, final=final, minimum_seconds=604_800)

    assert summary["status"] == "accepted"
    assert summary["checks"]["virtual_exit_observed_when_required"] is True


def test_previous_summary_gate_accepts_only_hashed_accepted_prior_phase() -> None:
    payload = {
        "artifact_kind": "stage18a_one_ticker_1h_summary_v1",
        "checks": {"duration_reached": True, "intents_zero": True},
        "elapsed_seconds": 3_660.0,
        "final": {"rollout_phase": "one_ticker_1h"},
        "status": "accepted",
    }
    payload["summary_hash"] = _payload_hash(payload)

    _validate_previous_summary_payload(payload, expected_phase="one_ticker_1h")

    payload["status"] = "blocked"
    with pytest.raises(ValueError, match="hash validation failed"):
        _validate_previous_summary_payload(payload, expected_phase="one_ticker_1h")


def _snapshot(*, phase: str = "one_ticker_1h", instrument_count: int = 1) -> dict[str, Any]:
    return {
        "database_counts": {
            "source_events": 0,
            "open_long_source_events": 0,
            "close_source_events": 0,
            "intents": 0,
            "orders": 0,
        },
        "health": {"ready": True},
        "log_json_valid": True,
        "metrics": {
            'rl_trading_inference_candles_total{result="processed"}': 1.0,
            "rl_trading_inference_model_loaded": 1.0,
        },
        "process_rss_mb": 256.0,
        "process_pid": 1234,
        "prometheus_target_health": "up",
        "recorded_at_utc": "2026-07-10T06:00:00Z",
        "revision": "a" * 40,
        "rollout_phase": phase,
        "instrument_keys": [
            f"binance:futures:TICKER{index}USDT" for index in range(instrument_count)
        ],
        "runtime_config_sha256": "c" * 64,
        "runtime_policy_sha256": "b" * 64,
        "stream_lengths": {
            "execution.requests.v1": 49,
            "execution.requests.retry.v1": 1,
            "execution.requests.dlq.v1": 2,
        },
    }
