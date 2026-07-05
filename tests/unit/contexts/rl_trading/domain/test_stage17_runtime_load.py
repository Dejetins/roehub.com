from __future__ import annotations

from typing import cast

from trading.contexts.rl_trading.domain import (
    STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_V1,
    Stage17LoadObservation,
    build_stage17_default_quota_scenarios_v1,
    summarize_stage17_runtime_load_v1,
)


def test_stage17_runtime_load_summary_accepts_bounded_infrastructure_load() -> None:
    scenarios = build_stage17_default_quota_scenarios_v1()
    observations = [
        _observation(
            scenario_label=scenario.label,
            paid_level=scenario.paid_level,
            product_label=scenario.product_label,
            live_slots_allowed=scenario.live_slots_allowed,
            index=index,
        )
        for scenario in scenarios
        for index in range(scenario.live_slots_allowed)
    ]

    summary = summarize_stage17_runtime_load_v1(
        observations=observations,
        quota_scenarios=scenarios,
        latency_budget_ms={
            "candle_close_to_feature_ready": 250,
            "feature_to_decision": 100,
            "decision_to_source_event": 50,
        },
        redis_stream_lengths_before={
            "execution.requests.dlq.v1": 2,
            "execution.requests.retry.v1": 1,
            "execution.requests.v1": 49,
        },
        redis_stream_lengths_after={
            "execution.requests.dlq.v1": 2,
            "execution.requests.retry.v1": 1,
            "execution.requests.v1": 49,
        },
        resource_usage={
            "max_rss_mb": 4096,
            "rss_mb_after": 128.0,
            "wall_time_seconds": 0.05,
        },
        contention={"status": "observed_overlap", "active_process_count": 1},
        max_feed_lag_seconds=300.0,
        generated_at_utc="2026-07-05T18:30:00Z",
        prompt_path=".codex/agents/generated/rl-trading-agent-platform-v1/17-multi-ticker-runtime-load.md",
        prompt_sha256="a" * 64,
        git_revision="b" * 40,
        config_profile="prod",
    )

    assert summary["status"] == "accepted"
    assert summary["mode"] == "infrastructure_only"
    assert summary["latency_p95_ms"] == {
        "candle_close_to_feature_ready": 4.0,
        "decision_to_source_event": 1.0,
        "feature_to_decision": 2.0,
    }
    redis_streams = cast(dict[str, object], summary["redis_execution_streams"])
    redis_delta = cast(dict[str, int], redis_streams["delta"])
    assert redis_delta["execution.requests.dlq.v1"] == 0
    assert summary["stage18_handoff"] == {
        "allowed_mode": "monitor_only_technical_soak",
        "forbidden_claims": [
            "model_quality",
            "trading_edge",
            "product_readiness",
            "mainnet_readiness",
        ],
        "max_monitor_only_tickers_for_technical_soak": 20,
        "reason": "stage17_infrastructure_only_load_gate_accepted",
        "stage18_allowed": True,
    }


def test_stage17_runtime_load_summary_blocks_dlq_growth() -> None:
    scenarios = build_stage17_default_quota_scenarios_v1()
    observations = [
        _observation(
            scenario_label=scenarios[0].label,
            paid_level=scenarios[0].paid_level,
            product_label=scenarios[0].product_label,
            live_slots_allowed=scenarios[0].live_slots_allowed,
            index=0,
        )
    ]

    summary = summarize_stage17_runtime_load_v1(
        observations=observations,
        quota_scenarios=scenarios,
        latency_budget_ms={
            "candle_close_to_feature_ready": 250,
            "feature_to_decision": 100,
            "decision_to_source_event": 50,
        },
        redis_stream_lengths_before={"execution.requests.dlq.v1": 2},
        redis_stream_lengths_after={"execution.requests.dlq.v1": 3},
        resource_usage={
            "max_rss_mb": 4096,
            "rss_mb_after": 128.0,
            "wall_time_seconds": 0.05,
        },
        contention={"status": "blocked_by_config", "active_process_count": 0},
        max_feed_lag_seconds=300.0,
        generated_at_utc="2026-07-05T18:30:00Z",
        prompt_path=".codex/agents/generated/rl-trading-agent-platform-v1/17-multi-ticker-runtime-load.md",
        prompt_sha256="a" * 64,
        git_revision="b" * 40,
        config_profile="prod",
    )

    assert summary["status"] == "blocked"
    acceptance_checks = cast(dict[str, bool], summary["acceptance_checks"])
    assert acceptance_checks["dlq_growth_zero"] is False
    assert summary["stage18_handoff"] == {
        "max_monitor_only_tickers_for_technical_soak": 0,
        "reason": "stage17_runtime_load_blocked",
        "stage18_allowed": False,
    }


def _observation(
    *,
    scenario_label: str,
    paid_level: str,
    product_label: str,
    live_slots_allowed: int,
    index: int,
) -> Stage17LoadObservation:
    return Stage17LoadObservation(
        scenario_label=scenario_label,
        paid_level=paid_level,
        product_label=product_label,
        live_slots_allowed=live_slots_allowed,
        exchange="binance",
        market_type="futures",
        symbol=f"TEST{index}USDT",
        instrument_key=f"binance:futures:TEST{index}USDT",
        feed_source="redis_streams_live_feed",
        feed_lag_seconds=30.0,
        feature_window_rows=30,
        redis_stream_length=120,
        action_name="hold",
        outcome=STAGE13_SOURCE_EVENT_OUTCOME_V1,
        outcome_reason=STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
        feature_hash="c" * 64,
        source_event_ref=f"rl:{'d' * 64}",
        latency_seconds={
            "candle_close_to_feature_ready": 0.004,
            "decision_to_source_event": 0.001,
            "feature_to_decision": 0.002,
        },
    )
