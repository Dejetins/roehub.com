from __future__ import annotations

from apps.exchange_execution import load_harness
from apps.exchange_execution.load_harness import LoadHarnessConfig, run_controlled_load


def test_controlled_load_harness_passes_testnet_mode_run() -> None:
    result = run_controlled_load(
        LoadHarnessConfig(
            strategy_count=12,
            exchange_read_count=4,
            rate_limit_per_second=1_000.0,
            rate_limit_burst=2,
        )
    )

    metrics = result["metrics"]
    assert result["passed"] is True
    assert result["violations"] == []
    assert metrics["mode_mix"] == {"testnet": 12, "paper": 0}
    assert metrics["orders_by_environment"] == {"testnet": 12}
    assert metrics["submitted_count"] == 12
    assert metrics["retry_count"] == 0
    assert metrics["dlq_count"] == 0
    assert metrics["redis_pending_final"] == 0
    assert metrics["limiter_wait"]["total_seconds"] > 0
    assert metrics["probe"]["backpressure"]["result"] == "retry"
    assert metrics["probe"]["backpressure"]["request_count"] == 0
    assert metrics["probe"]["retry_budget"]["result"] == "dlq"


def test_controlled_load_harness_rejects_non_testnet_order_environment() -> None:
    metrics = {
        "strategy_count": 2,
        "mode_mix": {"testnet": 2, "paper": 0},
        "orders_by_environment": {"testnet": 2, "mainnet": 1},
        "submitted_count": 2,
        "guard_rejected_count": 0,
        "adapter_error_count": 0,
        "quarantined_count": 0,
        "retry_count": 0,
        "dlq_count": 0,
        "redis_pending_final": 0,
        "redis_max_pending": 2,
        "config_read_count": 2,
        "queue_lag_ms": {"p95": 0.0, "p99": 0.0},
        "signal_to_source_ms": {"p99": 0.0},
        "source_to_intent_ms": {"p99": 0.0},
        "risk_ms": {"p99": 0.0},
        "dispatch_ms": {"p99": 0.0},
        "limiter_wait": {"total_seconds": 0.1, "p99_ms": 1.0},
        "ack_fill_latency_ms": {"p95": 0.0, "p99": 0.0},
        "reconciliation": {"pending": 0},
        "cpu_seconds": 0.0,
        "max_rss_delta_mb": 0.0,
        "probe": {
            "backpressure": {"retry_count": 1, "request_count": 0},
            "retry_budget": {"dlq_count": 1, "result": "dlq"},
        },
    }

    violations = load_harness._violations(
        metrics=metrics,
        thresholds=load_harness._thresholds(strategy_count=2),
    )

    assert "non_testnet_or_missing_order_environment" in violations
