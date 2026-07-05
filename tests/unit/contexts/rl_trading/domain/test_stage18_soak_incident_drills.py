from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from trading.contexts.rl_trading.domain import (
    STAGE18_MAX_TICKERS_V1,
    Stage18IncidentDrill,
    build_stage18_default_incident_drills_v1,
    summarize_stage18_monitor_only_technical_soak_v1,
)


def test_stage18_summary_accepts_monitor_only_technical_soak_and_drills() -> None:
    payload = summarize_stage18_monitor_only_technical_soak_v1(
        stage17_summary=_stage17_summary(),
        incident_drills=build_stage18_default_incident_drills_v1(),
        ui_evidence=_ui_evidence(),
        generated_at_utc="2026-07-05T19:00:00Z",
        prompt_path=".codex/agents/generated/rl-trading-agent-platform-v1/18-rl-soak-incident-drills.md",
        prompt_sha256="a" * 64,
        git_revision="b" * 40,
        run_id="stage18_test",
    )

    assert payload["status"] == "accepted"
    assert payload["mode"] == "monitor_only_technical_soak"
    quality_claims = cast(dict[str, object], payload["quality_claims"])
    soak_observation = cast(dict[str, object], payload["soak_observation"])
    stage19_handoff = cast(dict[str, object], payload["stage19_handoff"])
    assert quality_claims == {
        "full_trade_readiness": False,
        "mainnet_readiness": False,
        "model_quality": False,
        "paper_testnet_live_execution_readiness": False,
        "product_readiness": False,
        "trading_edge": False,
    }
    assert soak_observation["max_observed_tickers"] == STAGE18_MAX_TICKERS_V1
    assert stage19_handoff["stage19_mainnet_readiness_allowed"] is False


def test_stage18_summary_blocks_unknown_state_without_reconciliation() -> None:
    drills = tuple(
        replace(drill, reconciliation_before_retry=False)
        if drill.name == "unknown_state"
        else drill
        for drill in build_stage18_default_incident_drills_v1()
    )

    payload = summarize_stage18_monitor_only_technical_soak_v1(
        stage17_summary=_stage17_summary(),
        incident_drills=drills,
        ui_evidence=_ui_evidence(),
        generated_at_utc="2026-07-05T19:00:00Z",
        prompt_path=".codex/agents/generated/rl-trading-agent-platform-v1/18-rl-soak-incident-drills.md",
        prompt_sha256="a" * 64,
        git_revision="b" * 40,
        run_id="stage18_test",
    )

    assert payload["status"] == "blocked"
    drill_checks = cast(dict[str, object], payload["drill_checks"])
    acceptance_checks = cast(dict[str, object], payload["acceptance_checks"])
    assert drill_checks["unknown_state_reconciles_before_retry"] is False
    assert acceptance_checks["required_incident_drills_passed"] is False


def test_stage18_summary_blocks_any_order_state_drill() -> None:
    drills = (
        Stage18IncidentDrill(
            name="unknown_state",
            status="passed",
            operator_action="dry_run_order_state_probe",
            detection="provider_order_state_unknown",
            fail_closed_result="blocked",
            recovery_evidence="requires_external_order_reconciliation",
            degraded_state_reason="order_state_unknown",
            order_state_involved=True,
        ),
        *[
            drill
            for drill in build_stage18_default_incident_drills_v1()
            if drill.name != "unknown_state"
        ],
    )

    payload = summarize_stage18_monitor_only_technical_soak_v1(
        stage17_summary=_stage17_summary(),
        incident_drills=drills,
        ui_evidence=_ui_evidence(),
        generated_at_utc="2026-07-05T19:00:00Z",
        prompt_path=".codex/agents/generated/rl-trading-agent-platform-v1/18-rl-soak-incident-drills.md",
        prompt_sha256="a" * 64,
        git_revision="b" * 40,
        run_id="stage18_test",
    )

    assert payload["status"] == "blocked"
    acceptance_checks = cast(dict[str, object], payload["acceptance_checks"])
    assert acceptance_checks["no_order_state_involved"] is False


def _stage17_summary() -> dict[str, Any]:
    observations = []
    for index in range(STAGE18_MAX_TICKERS_V1):
        observations.append(
            {
                "action_name": "open_long" if index % 2 else "hold",
                "feature_hash": f"{index:064x}",
                "instrument_key": f"binance:futures:TEST{index}USDT",
                "outcome": "no_intent",
                "outcome_reason": "monitor_only_no_intent",
                "source_type": "ml_agent_decision",
            }
        )
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
        "observations": observations,
        "quota_scenarios": [
            {
                "label": "premium",
                "observed_tickers": STAGE18_MAX_TICKERS_V1,
                "requested_live_tickers": STAGE18_MAX_TICKERS_V1,
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
        "summary_hash": "c" * 64,
        "summary_path": "/opt/roehub/state/rl_trading/stage17.json",
    }


def _ui_evidence() -> dict[str, Any]:
    return {
        "observed_states": [
            "rl_ml.state=degraded",
            "operator_controls.disabled",
            "active_mode=monitor_only",
        ],
        "status": "observed",
        "surface": "GET /ui/strategies/dashboard and /strategies browser",
    }
