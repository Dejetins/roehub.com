from tools.release.verify_market_data_readiness import (
    _metrics_by_name,
    _readiness_snapshot,
)


def test_metrics_by_name_aggregates_labeled_samples_without_created_series() -> None:
    metrics = _metrics_by_name(
        "\n".join(
            (
                "# HELP scheduler_job_errors_total Scheduler job errors",
                'scheduler_job_errors_total{job="sync"} 0',
                'scheduler_job_errors_total{job="seed"} 0',
                "ws_messages_total 3",
            )
        )
    )

    assert metrics == {"scheduler_job_errors_total": 0.0, "ws_messages_total": 3.0}


def test_readiness_snapshot_requires_all_live_signals() -> None:
    ready = _readiness_snapshot(
        ws_metrics={
            "ws_connected": 1.0,
            "ws_messages_total": 2.0,
            "insert_rows_total": 1.0,
            "ws_errors_total": 0.0,
            "insert_errors_total": 0.0,
            "rest_fill_errors_total": 0.0,
        },
        scheduler_metrics={
            "scheduler_job_runs_total": 2.0,
            "scheduler_job_errors_total": 0.0,
        },
        newest_candle_timestamp=1_000.0,
        now_timestamp=1_100.0,
        max_candle_age_seconds=180.0,
        min_ws_messages=1,
    )

    assert ready["ready"] is True

    stale = _readiness_snapshot(
        ws_metrics={**ready["ws"], "ws_errors_total": 1.0},
        scheduler_metrics=ready["scheduler"],
        newest_candle_timestamp=1_000.0,
        now_timestamp=1_400.0,
        max_candle_age_seconds=180.0,
        min_ws_messages=1,
    )

    assert stale["ready"] is False
    assert stale["checks"]["ws_errors"] is False
    assert stale["checks"]["fresh_candle"] is False
