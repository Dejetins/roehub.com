from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.rl_trading.stage18_rl_soak_incident_drills import (
    run_stage18_rl_soak_incident_drills,
)


def test_stage18_cli_writes_sanitized_summary_from_stage17_input(tmp_path: Path) -> None:
    stage17_summary = tmp_path / "stage17_summary.json"
    stage17_summary.write_text(
        json.dumps(_stage17_summary(), ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        allow_fixture_manifest_hash=False,
        candidate_manifest=None,
        config=None,
        execution_stream=None,
        generated_at_utc="2026-07-05T19:00:00Z",
        iterations_per_scenario=1,
        market=None,
        max_feed_lag_seconds=300.0,
        output_root=str(tmp_path),
        redis_auth_env=None,
        redis_db=None,
        redis_host=None,
        redis_port=None,
        run_id="stage18_cli_test",
        scan_limit=2000,
        stage17_summary=str(stage17_summary),
        stream_prefix=None,
        ui_evidence_json=json.dumps(_ui_evidence(), ensure_ascii=True, sort_keys=True),
        window_size=None,
    )

    payload = run_stage18_rl_soak_incident_drills(args=args)

    summary_path = Path(str(payload["summary_path"]))
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "accepted"
    assert written["summary_hash"] == payload["summary_hash"]
    assert written["incident_drills"][0]["exchange_side_effect"] == "none"
    assert "password" not in summary_path.read_text(encoding="utf-8").lower()


def _stage17_summary() -> dict[str, object]:
    return {
        "acceptance_checks": {
            "monitor_only_source_events_only": True,
            "redis_execution_stream_growth_zero": True,
        },
        "feed_lag": {"max_seconds": 5.0, "p95_seconds": 4.0, "threshold_seconds": 300.0},
        "latency_p95_ms": {
            "candle_close_to_feature_ready": 1.0,
            "decision_to_source_event": 1.0,
            "feature_to_decision": 1.0,
        },
        "observations": [
            {
                "action_name": "hold",
                "feature_hash": "1" * 64,
                "instrument_key": "binance:futures:BTCUSDT",
                "outcome": "no_intent",
                "outcome_reason": "monitor_only_no_intent",
                "source_type": "ml_agent_decision",
            }
        ],
        "quota_scenarios": [
            {
                "label": "free",
                "observed_tickers": 1,
                "requested_live_tickers": 1,
            }
        ],
        "redis_execution_streams": {
            "delta": {
                "execution.requests.dlq.v1": 0,
                "execution.requests.retry.v1": 0,
                "execution.requests.v1": 0,
            }
        },
        "resource_usage": {"rss_mb_after": 128.0},
        "stage18_handoff": {
            "allowed_mode": "monitor_only_technical_soak",
            "stage18_allowed": True,
        },
        "status": "accepted",
        "summary_hash": "2" * 64,
    }


def _ui_evidence() -> dict[str, object]:
    return {
        "observed_states": [
            "rl_ml.state=degraded",
            "operator_controls.disabled",
            "active_mode=monitor_only",
        ],
        "status": "observed",
        "surface": "GET /ui/strategies/dashboard and /strategies browser",
    }
