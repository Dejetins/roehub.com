from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(*, relative_path: str) -> dict[str, Any]:
    payload = yaml.safe_load((_repo_root() / relative_path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"{relative_path} is empty")
    return cast(dict[str, Any], payload)


def test_oidc_provider_alert_covers_transport_deadline_and_validation_failures() -> None:
    payload = _load_yaml(relative_path=("infra/monitoring/rules/" "platform-monitoring.rules.yml"))
    alerts = {
        rule["alert"]: rule
        for group in payload["groups"]
        for rule in group["rules"]
        if "alert" in rule
    }
    expression = alerts["OidcProviderUnavailable"]["expr"]
    for outcome in (
        "transport_error",
        "result_unknown",
        "deadline_exceeded",
        "validation_error",
    ):
        assert outcome in expression


def test_prometheus_stage17_rules_are_repo_managed() -> None:
    rules_payload = _load_yaml(
        relative_path="infra/monitoring/rules/live-execution-stage17.rules.yml"
    )
    groups = rules_payload["groups"]
    assert [group["name"] for group in groups] == ["live-execution-production-readiness"]
    rules = groups[0]["rules"]
    alerts = {rule["alert"]: rule for rule in rules}
    assert set(alerts) == {
        "LiveExecutionDlqGrowing",
        "LiveExecutionClockDriftUnsafe",
        "LiveExecutionPrivateStreamMissingForSubmit",
        "LiveExecutionLimiterWaitHigh",
        "LiveExecutionSubmitLatencyHigh",
        "LiveExecutionDispatchBackpressure",
        "LiveExecutionReconciliationPending",
        "LiveExecutionPitrNotVerified",
        "LiveExecutionUnknownState",
    }
    for rule in alerts.values():
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["labels"]["owner"] == "live-execution"
        assert rule["annotations"]["runbook"] == (
            "docs/runbooks/exchange-execution.md#stage-17-alert-actions"
        )
        assert rule["annotations"]["escalation"]
        assert rule["annotations"]["action"]


def test_prometheus_rl_monitor_rules_are_repo_managed() -> None:
    rules_payload = _load_yaml(
        relative_path="infra/monitoring/rules/rl-trading-inference.rules.yml"
    )
    groups = rules_payload["groups"]
    assert [group["name"] for group in groups] == ["rl-trading-inference-monitor-only"]
    alerts = {rule["alert"]: rule for rule in groups[0]["rules"]}
    assert set(alerts) == {
        "RlTradingInferenceDown",
        "RlTradingInferenceErrors",
        "RlTradingInferenceFeedLag",
        "RlTradingInferenceNotReady",
        "RlTradingInferenceSafetyBreach",
    }
    for rule in alerts.values():
        assert rule["labels"]["owner"] == "rl-trading"
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["annotations"]["runbook"].startswith("docs/runbooks/rl-trading-operations.md#")
        assert rule["annotations"]["action"]
    feed_lag_expr = alerts["RlTradingInferenceFeedLag"]["expr"]
    assert "time() - rl_trading_inference_last_candle_unixtime" in feed_lag_expr
    assert "rl_trading_inference_last_candle_unixtime" in feed_lag_expr


def test_prometheus_strategy_producer_rules_are_repo_managed() -> None:
    rules_payload = _load_yaml(relative_path="infra/monitoring/rules/strategy-producer.rules.yml")
    groups = rules_payload["groups"]
    assert [group["name"] for group in groups] == ["strategy-producer-supervision"]
    rules = groups[0]["rules"]
    alerts = {rule["alert"]: rule for rule in rules}
    assert set(alerts) == {
        "StrategyProducerDown",
        "StrategyProducerIterationErrors",
        "StrategyProducerModeBlocked",
        "StrategyProducerExecutionRejected",
        "StrategyProducerCriticalIncidentNotification",
        "StrategyProducerRunStateNotification",
    }
    for rule in alerts.values():
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["labels"]["owner"] == "strategy-producer"
        assert rule["annotations"]["runbook"].startswith("docs/runbooks/strategy-live-worker.md#")
        assert rule["annotations"]["escalation"]
        assert rule["annotations"]["action"]
    assert "producer_signal_rejected" in json.dumps(rules_payload)
    assert "producer_resource_threshold_breached" in json.dumps(rules_payload)


def test_prometheus_notifications_admin_rules_are_repo_managed() -> None:
    rules_payload = _load_yaml(relative_path="infra/monitoring/rules/notifications-admin.rules.yml")
    groups = rules_payload["groups"]
    assert [group["name"] for group in groups] == ["notifications-admin"]
    rules = groups[0]["rules"]
    alerts = {rule["alert"]: rule for rule in rules}
    assert set(alerts) == {
        "NotificationsCriticalUnknownDelivery",
        "NotificationsProviderInstanceDegraded",
        "NotificationsDispatcherPendingOld",
        "NotificationsWorkerDown",
        "NotificationsTelegramApiDown",
        "NotificationsRetry429High",
        "NotificationsMissedReportSchedule",
    }
    for rule in alerts.values():
        assert rule["labels"]["owner"] == "notifications"
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["annotations"]["runbook"].startswith(
            "docs/runbooks/notifications-admin-alerts.md#"
        )
        assert rule["annotations"]["escalation"]
        assert rule["annotations"]["action"]
    telegram_expr = alerts["NotificationsTelegramApiDown"]["expr"]
    assert "absent(notifications_telegram_api_up)" in telegram_expr
    assert "notifications_telegram_api_last_success_unixtime" in telegram_expr
    assert (
        alerts["NotificationsProviderInstanceDegraded"]["expr"]
        == "notification_provider_instance_ready == 0"
    )
    critical_unknown_expr = alerts["NotificationsCriticalUnknownDelivery"]["expr"]
    assert "strategy_run_failed" in critical_unknown_expr
    assert "trade_fill" in critical_unknown_expr


def test_prometheus_funding_rules_are_repo_managed() -> None:
    rules_payload = _load_yaml(relative_path="infra/monitoring/rules/market-data-funding.rules.yml")
    groups = rules_payload["groups"]
    assert [group["name"] for group in groups] == ["market-data-funding"]
    rules = groups[0]["rules"]
    alerts = {rule["alert"]: rule for rule in rules}
    assert set(alerts) == {
        "MarketDataFundingCatchupErrors",
        "MarketDataFundingNoRecentSuccess",
        "MarketDataFundingLagHigh",
        "MarketDataFundingMissingIntervals",
    }
    for rule in rules:
        assert "symbol" not in rule.get("labels", {})
    assert "scheduler_funding_catchup_" in json.dumps(rules_payload)


def test_prometheus_live_tail_repair_rules_are_repo_managed() -> None:
    rules_payload = _load_yaml(
        relative_path="infra/monitoring/rules/market-data-live-tail-repair.rules.yml"
    )
    groups = rules_payload["groups"]
    assert [group["name"] for group in groups] == ["market-data-live-tail-repair"]
    rules = groups[0]["rules"]
    alerts = {rule["alert"]: rule for rule in rules}
    assert set(alerts) == {
        "MarketDataLiveTailUnrepairedGapBeyondPolicy",
        "MarketDataClickHouseRepairCircuitOpenTooLong",
        "MarketDataRestTailRepairErrors",
        "MarketDataHotCacheShortTailMiss",
        "StrategyProducerNoSignalGrowth",
    }
    forbidden_labels = {"user_id", "run_id", "order_id", "symbol", "instrument_key"}
    for rule in rules:
        labels = rule["labels"]
        annotations = rule["annotations"]
        assert labels["severity"] in {"warning", "critical"}
        assert labels["owner"] in {"market-data", "strategy-producer"}
        assert not forbidden_labels.intersection(labels)
        assert annotations["runbook"].startswith("docs/runbooks/market-data-live-tail-repair.md#")
        assert annotations["action"]
    assert "market_data_live_tail_repair_total" in json.dumps(rules_payload)
    assert "strategy_live_runner_deferred_ack_total" in json.dumps(rules_payload)
