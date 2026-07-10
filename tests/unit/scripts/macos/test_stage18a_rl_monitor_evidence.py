from __future__ import annotations

from copy import deepcopy
from typing import Any

from scripts.macos.stage18a_rl_monitor_evidence import assess_window


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


def _snapshot() -> dict[str, Any]:
    return {
        "database_counts": {"source_events": 0, "intents": 0, "orders": 0},
        "health": {"ready": True},
        "log_json_valid": True,
        "metrics": {
            'rl_trading_inference_candles_total{result="processed"}': 1.0,
            "rl_trading_inference_model_loaded": 1.0,
        },
        "process_rss_mb": 256.0,
        "prometheus_target_health": "up",
        "recorded_at_utc": "2026-07-10T06:00:00Z",
        "revision": "a" * 40,
        "runtime_policy_sha256": "b" * 64,
        "stream_lengths": {
            "execution.requests.v1": 49,
            "execution.requests.retry.v1": 1,
            "execution.requests.dlq.v1": 2,
        },
    }
