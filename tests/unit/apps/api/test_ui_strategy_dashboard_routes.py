from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_ui_strategies_dashboard_router
from apps.api.wiring.modules.research_tenancy import DevelopmentOrganizationScopeResolver
from apps.api.wiring.modules.ui_strategies_dashboard import StrategyDashboardQueryService
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.live_execution.domain import ExecutionProducerOutcomeLink
from trading.contexts.market_data.application.dto.reference_api import (
    BTCUSDTMarketReadinessReport,
    BTCUSDTMarketReadinessRow,
)
from trading.contexts.rl_trading.adapters.outbound.persistence import (
    InMemoryRlLiveTickerEntitlementRepository,
    InMemoryRlRiskSizingPolicyRepository,
)
from trading.contexts.rl_trading.domain.live_entitlements import (
    RlLiveTickerEntitlementService,
)
from trading.contexts.rl_trading.domain.risk_sizing_policy import (
    RlRiskSizingPolicyConfig,
    RlRiskSizingPolicyKey,
    RlRiskSizingPolicyService,
)
from trading.contexts.strategy.domain.entities import (
    LiveStrategyProfile,
    Strategy,
    StrategyRun,
    StrategySignal,
)
from trading.contexts.strategy.domain.entities.strategy_spec_v1 import StrategySpecV1
from trading.shared_kernel.primitives import OrganizationId, PaidLevel, UserId

_USER_ID = "00000000-0000-0000-0000-000000006006"
_ORGANIZATION_ID = OrganizationId.from_string("00000000-0000-4000-8000-000000000010")
_STRATEGY_ID = UUID("00000000-0000-0000-0000-000000006101")

_STRATEGY_DASHBOARD_ENDPOINT_CONTRACT = {
    "method_path": {
        "browser": "GET /api/ui/strategies/dashboard",
        "backend": "GET /ui/strategies/dashboard",
    },
    "owner_scope": "current identity principal; strategy rows are loaded by owner user id",
    "request_dto": (
        "query strategy_id optional, state=active|all default all, cursor optional, "
        "refresh=initial|auto|manual default initial"
    ),
    "response_dto": (
        "bounded StrategyDashboardResponse with sources[], freshness, selected strategy, "
        "panel states, selector and refresh_control"
    ),
    "status_codes": "200, 401, 422; manual refresh rate limit is represented in DTO",
    "error_payload": "RoehubError envelope for auth.required and validation_error",
    "pagination": (
        "cursor accepted for compatibility; first implementation caps selector/trades arrays"
    ),
    "cache_identity": "none; no persisted request hash or cache key",
    "compatibility": "compatible-change",
}


def test_strategy_dashboard_exposes_reference_panel_inventory_and_degraded_stats() -> None:
    strategy = _strategy(symbol="BTCUSDT")
    run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-000000006201"),
        organization_id=_ORGANIZATION_ID,
        user_id=strategy.user_id,
        strategy_id=strategy.strategy_id,
        started_at=datetime(2026, 5, 6, 9, 0, tzinfo=UTC),
        metadata_json={
            "warmup": {
                "algorithm": "numeric_max_param_v1",
                "bars": 5,
                "processed_bars": 3,
                "satisfied": False,
            },
            "rollup": {
                "timeframe": "1m",
                "bucket_open_ts": "2026-05-06T09:00:00Z",
                "bucket_count_1m": 2,
            },
        },
    )
    risk_policy_service = RlRiskSizingPolicyService(
        repository=InMemoryRlRiskSizingPolicyRepository(),
    )
    risk_policy_service.upsert_policy(
        key=RlRiskSizingPolicyKey(
            organization_id=strategy.organization_id,
            owner_user_id=strategy.user_id,
            strategy_id=strategy.strategy_id,
            exchange_name="binance",
            market_type="spot",
            symbol="BTCUSDT",
        ),
        config=_valid_risk_policy_config(),
        observed_at=datetime(2026, 5, 6, 8, 59, tzinfo=UTC),
    )
    service = StrategyDashboardQueryService(
        organization_scope_resolver=DevelopmentOrganizationScopeResolver(),
        strategy_repository=_FakeStrategyRepository(strategies=(strategy,)),
        run_repository=_FakeRunRepository(runs=(run,)),
        signal_repository=_FakeSignalRepository(signals=(_signal(strategy=strategy, run=run),)),
        btcusdt_market_readiness_service=_FakeBTCUSDTMarketReadinessService(),  # type: ignore[arg-type]
        rl_live_ticker_entitlement_service=RlLiveTickerEntitlementService(
            repository=InMemoryRlLiveTickerEntitlementRepository(),
        ),
        rl_risk_sizing_policy_service=risk_policy_service,
    )
    client = _build_client(service=service)

    response = client.get(
        f"/ui/strategies/dashboard?strategy_id={_STRATEGY_ID}&state=active",
        headers={"x-user-id": _USER_ID},
    )

    assert response.status_code == 200
    payload = response.json()
    assert _STRATEGY_DASHBOARD_ENDPOINT_CONTRACT["method_path"]["browser"] == (
        "GET /api/ui/strategies/dashboard"
    )
    assert payload["selected_strategy"]["strategy_id"] == str(_STRATEGY_ID)
    assert payload["selected_strategy"]["status"] == "live"
    assert payload["selected_strategy"]["exchange"] == "binance"
    assert payload["selected_strategy"]["symbols"] == ["BTCUSDT"]
    assert payload["live_profile"]["mode"] == "monitor_only"
    assert payload["live_profile"]["readiness_status"] == "ready"
    assert payload["live_profile"]["readiness_reason"] == "monitor_only_no_exchange_submit"
    assert payload["runtime_status"]["environment"] == "monitor_only"
    assert payload["runtime_status"]["producer_status"] == "running"
    assert payload["runtime_status"]["mainnet_available"] is False
    assert payload["strategy_selector"]["filters"]["state"] == "active"
    assert payload["strategy_selector"]["totals"]["strategies"] == 1
    assert payload["strategy_selector"]["items"][0]["status"] == "live"
    for panel in [
        "chart",
        "metric_grid",
        "monthly_stats",
        "long_short",
        "drawdown",
        "equity_curve",
        "hourly_results",
        "trades",
    ]:
        assert payload[panel]["state"] == "unavailable"
        assert payload[panel]["source"]
        assert payload[panel]["degradation_reason"]
    assert payload["signal_journal"]["state"] == "ready"
    assert payload["signal_journal"]["items"][0]["outcome"] == "signal"
    assert payload["signal_journal"]["items"][0]["mode"] == "monitor_only"
    assert payload["signal_journal"]["items"][0]["reason_code"].endswith("monitor_only_no_intent")
    assert payload["exchange_account_readiness"]["status"] == "degraded"
    assert payload["exchange_account_readiness"]["reason_codes"] == [
        "account_projection_not_configured"
    ]
    assert payload["exchange_account_readiness"]["ready_for_risk"] is False
    assert payload["market_readiness"]["symbol"] == "BTCUSDT"
    assert payload["market_readiness"]["state"] == "degraded"
    assert payload["market_readiness"]["items"][0]["instrument_key"] == "binance:spot:BTCUSDT"
    assert payload["market_readiness"]["items"][0]["readiness_state"] == "ready"
    assert payload["market_readiness"]["items"][1]["instrument_key"] == "bybit:spot:BTCUSDT"
    assert payload["market_readiness"]["items"][1]["reason_codes"] == ["reference_market_missing"]
    assert "summary" not in payload["monthly_stats"]
    assert "symbol_results" not in payload
    assert payload["refresh_control"]["interval_seconds"] == 15
    assert payload["refresh_control"]["preset_key"] == "15s"
    source_statuses = {source["name"]: source["status"] for source in payload["sources"]}
    assert source_statuses["strategy_strategies"] == "available"
    assert source_statuses["strategy_runs"] == "available"
    assert source_statuses["strategy_live_profiles"] == "unavailable"
    assert source_statuses["strategy_signals"] == "available"
    assert source_statuses["exchange_account_projection"] == "unavailable"
    assert source_statuses["btcusdt_market_readiness"] == "degraded"
    assert source_statuses["strategy_run_metadata"] == "available"
    assert source_statuses["strategy_stat_projections"] == "unavailable"
    assert source_statuses["execution_fills"] == "unavailable"
    assert payload["rl_ml"]["state"] == "degraded"
    assert payload["rl_ml"]["model_status"]["model_family"] == "rl-trading-agent-platform-v1"
    assert payload["rl_ml"]["model_status"]["artifact_root"] == ("/opt/roehub/state/rl_trading/")
    assert payload["rl_ml"]["model_status"]["registry_status"] == "not_configured"
    assert payload["rl_ml"]["ticker_slots"]["paid_level"] == "free"
    assert payload["rl_ml"]["ticker_slots"]["product_label"] == "Free"
    assert payload["rl_ml"]["ticker_slots"]["entitlement_source"] == "paid_level"
    assert payload["rl_ml"]["ticker_slots"]["live_slots_allowed"] == 1
    assert payload["rl_ml"]["ticker_slots"]["live_slots_used"] == 0
    assert payload["rl_ml"]["ticker_slots"]["degradation_reason"] is None
    assert payload["rl_ml"]["ticker_slots"]["items"][0]["symbol"] == "BTCUSDT"
    assert payload["rl_ml"]["ticker_slots"]["items"][0]["readiness_reason"] == (
        "rl_live_ticker_not_counted_for_mode"
    )
    assert payload["rl_ml"]["modes"]["active_mode"] == "monitor_only"
    assert payload["rl_ml"]["modes"]["options"][0] == {
        "mode": "monitor_only",
        "enabled": True,
        "reason": "safe_read_only_mode_available",
    }
    assert payload["rl_ml"]["modes"]["options"][1]["reason"] == (
        "stage15_classic_paper_prerequisites_blocked"
    )
    assert payload["rl_ml"]["risk_config"]["risk_gate_status"] == "ready"
    assert payload["rl_ml"]["risk_config"]["policy_status"] == "ready"
    assert payload["rl_ml"]["risk_config"]["base_quote_notional"] == "25"
    assert payload["rl_ml"]["risk_config"]["validation_reasons"] == ["rl_risk_policy_ready"]
    synthetic_rule_types = [
        rule["rule_type"] for rule in payload["rl_ml"]["risk_config"]["synthetic_exit_rules"]
    ]
    assert synthetic_rule_types == [
        "take_profit",
        "stop_loss",
        "trailing_stop",
    ]
    assert payload["rl_ml"]["risk_config"]["notes"] == [
        "stage14_synthetic_exits_platform_side_only",
        "stage14_no_exchange_submit",
        "live_execution_risk_gate_required_before_execution",
    ]
    assert payload["rl_ml"]["operator_controls"]["guard_available"] is False
    assert payload["rl_ml"]["operator_controls"]["operator_authorized"] is False
    assert {
        control["action"]: (control["enabled"], control["blocked_reason"])
        for control in payload["rl_ml"]["operator_controls"]["controls"]
    } == {
        "request_retraining": (False, "operator_admin_guard_not_available"),
        "request_rollback": (False, "operator_admin_guard_not_available"),
    }
    assert payload["rl_ml"]["source_event_outcomes"]["state"] == "empty"
    assert payload["rl_ml"]["source_event_outcomes"]["degradation_reason"] == (
        "ml_agent_decision_outcomes_empty"
    )
    source_freshness = {source["name"]: source["age_seconds"] for source in payload["sources"]}
    assert isinstance(source_freshness["strategy_strategies"], int)
    assert isinstance(source_freshness["strategy_runs"], int)
    assert isinstance(source_freshness["strategy_run_metadata"], int)
    risk_rows = {row["key"]: row for row in payload["risk_execution"]["rows"]}
    assert payload["risk_execution"]["source"] == "strategy_run_metadata"
    assert payload["risk_execution"]["state"] == "ready"
    assert risk_rows["run_state"]["total_value"] == "starting"
    assert risk_rows["warmup_progress"]["total_value"] == "3/5"
    assert risk_rows["warmup_satisfied"]["total_value"] == "no"
    assert risk_rows["rollup_bucket_count_1m"]["total_value"] == "2"
    compressed = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    assert len(compressed) < 96 * 1024


def test_strategy_dashboard_exposes_runtime_status_allocation_and_execution_journal() -> None:
    strategy = _strategy(symbol="BTCUSDT")
    run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-000000006202"),
        organization_id=_ORGANIZATION_ID,
        user_id=strategy.user_id,
        strategy_id=strategy.strategy_id,
        started_at=datetime(2026, 5, 6, 9, 0, tzinfo=UTC),
    )
    profile = _profile(strategy=strategy, mode="paper")
    signal = _signal(strategy=strategy, run=run)
    source_event_received_at = datetime(2026, 5, 6, 9, 2, 3, tzinfo=UTC)
    execution_updated_at = source_event_received_at + timedelta(seconds=7)
    service = StrategyDashboardQueryService(
        organization_scope_resolver=DevelopmentOrganizationScopeResolver(),
        strategy_repository=_FakeStrategyRepository(strategies=(strategy,)),
        run_repository=_FakeRunRepository(runs=(run,)),
        profile_repository=_FakeLiveProfileRepository(profile=profile),
        signal_repository=_FakeSignalRepository(signals=(signal,)),
        execution_outcome_service=_FakeExecutionOutcomeService(
            links=(
                ExecutionProducerOutcomeLink(
                    source_event_id=UUID("00000000-0000-0000-0000-000000006401"),
                    organization_id=_ORGANIZATION_ID,
                    owner_user_id=strategy.user_id,
                    source_type="strategy_signal",
                    source_event_ref=str(signal.signal_id),
                    source_event_received_at=source_event_received_at,
                    strategy_signal_id=signal.signal_id,
                    outcome="dispatched",
                    outcome_reason="intent_recorded",
                    intent_id=UUID("00000000-0000-0000-0000-000000006501"),
                    intent_status="accepted",
                    intent_status_reason="risk_accepted",
                    risk_status="accepted",
                    risk_reason="risk_passed",
                    order_status="filled",
                    order_status_reason="exchange_fill_recorded",
                    fill_count=1,
                    latest_fill_at=execution_updated_at,
                    reconciliation_status="matched",
                    reconciliation_reason="status_and_fills_matched",
                    notification_event_type="producer_fill",
                    notification_reason="order_filled",
                    updated_at=execution_updated_at,
                ),
                ExecutionProducerOutcomeLink(
                    source_event_id=UUID("00000000-0000-0000-0000-000000006402"),
                    organization_id=_ORGANIZATION_ID,
                    owner_user_id=strategy.user_id,
                    source_type="ml_agent_decision",
                    source_event_ref="rl-decision-202605060902",
                    source_event_received_at=source_event_received_at + timedelta(seconds=1),
                    strategy_signal_id=None,
                    outcome="no_intent",
                    outcome_reason="monitor_only_no_intent",
                    intent_id=None,
                    intent_status=None,
                    intent_status_reason=None,
                    risk_status=None,
                    risk_reason=None,
                    order_status=None,
                    order_status_reason=None,
                    fill_count=0,
                    latest_fill_at=None,
                    reconciliation_status=None,
                    reconciliation_reason=None,
                    notification_event_type="ml_agent_decision",
                    notification_reason="monitor_only_journal",
                    updated_at=execution_updated_at + timedelta(seconds=1),
                ),
            )
        ),
    )
    client = _build_client(service=service)

    response = client.get(
        f"/ui/strategies/dashboard?strategy_id={_STRATEGY_ID}",
        headers={"x-user-id": _USER_ID},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_strategy"]["capital_usdt"] == 50.0
    assert payload["live_profile"]["mode"] == "paper"
    assert payload["runtime_status"]["environment"] == "paper"
    assert payload["runtime_status"]["producer_status"] == "running"
    assert payload["runtime_status"]["producer_reason"] == "starting"
    assert payload["runtime_status"]["mainnet_available"] is False
    assert payload["runtime_status"]["latest_signal_at"] == "2026-05-06T09:02:00Z"
    assert payload["runtime_status"]["latest_source_event_at"] == "2026-05-06T09:02:04Z"
    assert payload["runtime_status"]["latest_execution_update_at"] == "2026-05-06T09:02:11Z"
    assert payload["runtime_status"]["observed_latency_gap_seconds"] == 7
    assert payload["runtime_status"]["observed_latency_gap_status"] == "observed"
    outcome = payload["execution_outcomes"]["items"][0]
    assert outcome["source_event_received_at"] == "2026-05-06T09:02:03Z"
    assert outcome["fill_count"] == 1
    assert outcome["latest_fill_at"] == "2026-05-06T09:02:10Z"
    assert outcome["reconciliation_status"] == "matched"
    assert outcome["reconciliation_reason"] == "status_and_fills_matched"
    assert outcome["latency_gap_seconds"] == 7
    assert outcome["latency_gap_status"] == "observed"
    assert outcome["latency_gap_reason"] == "source_event_to_latest_update"
    rl_outcomes = payload["rl_ml"]["source_event_outcomes"]
    assert rl_outcomes["source"] == "execution_producer_outcomes"
    assert rl_outcomes["state"] == "ready"
    assert len(rl_outcomes["items"]) == 1
    assert rl_outcomes["items"][0]["source_type"] == "ml_agent_decision"
    assert rl_outcomes["items"][0]["strategy_signal_id"] is None
    assert rl_outcomes["items"][0]["outcome_reason"] == "monitor_only_no_intent"
    assert rl_outcomes["items"][0]["latency_gap_status"] == "observed"


def test_strategy_dashboard_auth_failure_uses_auth_required_code() -> None:
    service = StrategyDashboardQueryService(
        organization_scope_resolver=DevelopmentOrganizationScopeResolver(),
        strategy_repository=_FakeStrategyRepository(strategies=()),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    response = client.get("/ui/strategies/dashboard")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.required"


def test_strategy_dashboard_source_failure_degrades_panels_without_500() -> None:
    service = StrategyDashboardQueryService(
        organization_scope_resolver=DevelopmentOrganizationScopeResolver(),
        strategy_repository=_FailingStrategyRepository(),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    response = client.get("/ui/strategies/dashboard", headers={"x-user-id": _USER_ID})

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_strategy"]["state"] == "empty"
    assert payload["strategy_selector"]["state"] == "empty"
    source_statuses = {source["name"]: source for source in payload["sources"]}
    assert source_statuses["strategy_strategies"]["status"] == "degraded"
    assert payload["refresh_status"] == "degraded"


def test_strategy_dashboard_manual_refresh_reports_rate_limit_in_dto() -> None:
    service = StrategyDashboardQueryService(
        organization_scope_resolver=DevelopmentOrganizationScopeResolver(),
        strategy_repository=_FakeStrategyRepository(strategies=()),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    first = client.get("/ui/strategies/dashboard?refresh=manual", headers={"x-user-id": _USER_ID})
    second = client.get("/ui/strategies/dashboard?refresh=manual", headers={"x-user-id": _USER_ID})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["refresh_status"] == "rate_limited"
    assert second.json()["retry_after_seconds"] >= 1
    assert second.json()["refresh_control"]["next_allowed_refresh_at"] is not None


class _HeaderCurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Authentication required",
                },
            )
        raw_paid_level = request.headers.get("x-paid-level", "free")
        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel(raw_paid_level),
        )


class _FakeStrategyRepository:
    def __init__(self, *, strategies: tuple[Strategy, ...]) -> None:
        self._strategies = strategies

    def list_for_user(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        include_deleted: bool = False,
    ) -> tuple[Strategy, ...]:
        return tuple(
            strategy
            for strategy in self._strategies
            if strategy.organization_id == organization_id
            and strategy.user_id == user_id
            and (include_deleted or not strategy.is_deleted)
        )

    def create(self, *, strategy: Strategy) -> Strategy:
        raise NotImplementedError

    def find_by_strategy_id(
        self, *, organization_id: OrganizationId, user_id: UserId, strategy_id: UUID
    ) -> Strategy | None:
        raise NotImplementedError

    def find_any_by_strategy_id(
        self, *, organization_id: OrganizationId, strategy_id: UUID
    ) -> Strategy | None:
        raise NotImplementedError

    def soft_delete(
        self, *, organization_id: OrganizationId, user_id: UserId, strategy_id: UUID
    ) -> bool:
        raise NotImplementedError


class _FailingStrategyRepository(_FakeStrategyRepository):
    def __init__(self) -> None:
        super().__init__(strategies=())

    def list_for_user(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        include_deleted: bool = False,
    ) -> tuple[Strategy, ...]:
        raise ValueError("strategy source unavailable")


class _FakeRunRepository:
    def __init__(self, *, runs: tuple[StrategyRun, ...] = ()) -> None:
        self._runs = runs

    def find_active_for_strategy(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        strategy_id: UUID,
    ) -> StrategyRun | None:
        for run in self._runs:
            if (
                run.organization_id == organization_id
                and run.user_id == user_id
                and run.strategy_id == strategy_id
                and run.is_active()
            ):
                return run
        return None

    def create(self, *, run: StrategyRun) -> StrategyRun:
        raise NotImplementedError

    def update(self, *, run: StrategyRun) -> StrategyRun:
        raise NotImplementedError

    def find_by_run_id(
        self, *, organization_id: OrganizationId, user_id: UserId, run_id: UUID
    ) -> StrategyRun | None:
        raise NotImplementedError

    def list_for_strategy(
        self, *, organization_id: OrganizationId, user_id: UserId, strategy_id: UUID
    ) -> tuple[StrategyRun, ...]:
        raise NotImplementedError

    def list_active_runs(self) -> tuple[StrategyRun, ...]:
        raise NotImplementedError


class _FakeSignalRepository:
    def __init__(self, *, signals: tuple[StrategySignal, ...]) -> None:
        self._signals = signals

    def record(self, *, signal: StrategySignal) -> StrategySignal:
        raise NotImplementedError

    def list_latest_for_strategy(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
        limit: int,
    ) -> tuple[StrategySignal, ...]:
        _ = limit
        return tuple(
            signal
            for signal in self._signals
            if signal.organization_id == organization_id
            and signal.owner_user_id == owner_user_id
            and signal.strategy_id == strategy_id
        )


class _FakeLiveProfileRepository:
    def __init__(self, *, profile: LiveStrategyProfile | None) -> None:
        self._profile = profile

    def get_for_strategy(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
    ) -> LiveStrategyProfile | None:
        if (
            self._profile is not None
            and self._profile.organization_id == organization_id
            and self._profile.owner_user_id == owner_user_id
            and self._profile.strategy_id == strategy_id
        ):
            return self._profile
        return None

    def create(self, *, profile: LiveStrategyProfile) -> LiveStrategyProfile | None:
        raise NotImplementedError

    def update(self, *, profile: LiveStrategyProfile) -> LiveStrategyProfile:
        raise NotImplementedError


class _FakeExecutionOutcomeService:
    def __init__(self, *, links: tuple[ExecutionProducerOutcomeLink, ...]) -> None:
        self._links = links

    def list_producer_outcome_links_for_strategy(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
        limit: int,
    ) -> tuple[ExecutionProducerOutcomeLink, ...]:
        _ = limit
        return tuple(
            link
            for link in self._links
            if link.organization_id == organization_id
            and link.owner_user_id == owner_user_id
            and link.source_event_ref
            and (link.strategy_signal_id is not None or link.source_type == "ml_agent_decision")
            and strategy_id == _STRATEGY_ID
        )


class _FakeBTCUSDTMarketReadinessService:
    def execute(self, *, observed_at: datetime | None = None) -> BTCUSDTMarketReadinessReport:
        checked_at = observed_at or datetime(2027, 1, 15, 8, 0, tzinfo=UTC)
        return BTCUSDTMarketReadinessReport(
            symbol="BTCUSDT",
            freshness_threshold_seconds=180,
            rows=(
                _market_readiness_row(
                    exchange_name="binance",
                    market_type="spot",
                    readiness_state="ready",
                    reason_codes=("btcusdt_market_ready",),
                    reference_state="ready",
                    stream_state="ready",
                    checked_at=checked_at,
                ),
                _market_readiness_row(
                    exchange_name="bybit",
                    market_type="spot",
                    readiness_state="blocked",
                    reason_codes=("reference_market_missing",),
                    reference_state="missing",
                    stream_state="pending",
                    checked_at=checked_at,
                ),
            ),
            checked_at=checked_at,
        )


def _market_readiness_row(
    *,
    exchange_name: str,
    market_type: str,
    readiness_state: str,
    reason_codes: tuple[str, ...],
    reference_state: str,
    stream_state: str,
    checked_at: datetime,
) -> BTCUSDTMarketReadinessRow:
    reference_ready = reference_state == "ready"
    return BTCUSDTMarketReadinessRow(
        market_id=None,
        exchange_name=exchange_name,
        market_type=market_type,
        market_code=f"{exchange_name}:{market_type}",
        symbol="BTCUSDT",
        instrument_key=f"{exchange_name}:{market_type}:BTCUSDT",
        readiness_state=readiness_state,  # type: ignore[arg-type]
        reason_codes=reason_codes,
        reference_state=reference_state,  # type: ignore[arg-type]
        reference_reason_codes=("reference_ready",) if reference_ready else reason_codes,
        market_enabled=reference_ready,
        status="ENABLED" if reference_ready else None,
        is_tradable=1 if reference_ready else None,
        base_asset="BTC" if reference_ready else None,
        quote_asset="USDT" if reference_ready else None,
        price_step=0.01 if reference_ready else None,
        qty_step=0.00001 if reference_ready else None,
        min_notional=10.0 if reference_ready else None,
        stream_state=stream_state,  # type: ignore[arg-type]
        stream_reason_code=(
            "market_data_stream_ready"
            if stream_state == "ready"
            else "market_data_readiness_reader_unavailable"
        ),
        stream_name=f"md.candles.1m.{exchange_name}:{market_type}:BTCUSDT",
        stream_length=12 if stream_state == "ready" else None,
        stream_last_message_id="1800000000000-0" if stream_state == "ready" else None,
        stream_last_observed_at=checked_at if stream_state == "ready" else None,
        stream_age_seconds=30 if stream_state == "ready" else None,
        checked_at=checked_at,
    )


def _build_client(*, service: StrategyDashboardQueryService) -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_strategies_dashboard_router(
            dashboard_service=service,
            current_user_dependency=_HeaderCurrentUserDependency(),
        )
    )
    return TestClient(app)


def _strategy(*, symbol: str) -> Strategy:
    user_id = UserId.from_string(_USER_ID)
    spec = StrategySpecV1.from_json(payload=_strategy_spec_payload(symbol=symbol))
    return Strategy.create(
        organization_id=_ORGANIZATION_ID,
        user_id=user_id,
        spec=spec,
        created_at=datetime(2026, 5, 6, 8, 0, tzinfo=UTC),
        strategy_id=_STRATEGY_ID,
    )


def _signal(*, strategy: Strategy, run: StrategyRun) -> StrategySignal:
    return StrategySignal(
        signal_id=UUID("00000000-0000-0000-0000-000000006301"),
        organization_id=_ORGANIZATION_ID,
        owner_user_id=strategy.user_id,
        strategy_id=strategy.strategy_id,
        strategy_run_id=run.run_id,
        live_profile_id=None,
        mode="monitor_only",
        instrument_key=strategy.spec.instrument_key,
        market_type=strategy.spec.market_type,
        timeframe=strategy.spec.timeframe.code,
        bar_ts_open=datetime(2026, 5, 6, 9, 1, tzinfo=UTC),
        bar_ts_close=datetime(2026, 5, 6, 9, 2, tzinfo=UTC),
        signal_action="open",
        side="buy",
        outcome="signal",
        reason_code="ma_fast_crossed_above_slow_monitor_only_no_intent",
        reference_price=Decimal("101.5"),
        confidence=Decimal("1"),
        expected_order_json={},
        source_message_id="1746522060000-0",
        evaluator_version="ma_cross_close_v1",
        created_at=datetime(2026, 5, 6, 9, 2, tzinfo=UTC),
    )


def _profile(
    *, strategy: Strategy, mode: Literal["monitor_only", "paper", "live", "testnet"]
) -> LiveStrategyProfile:
    return LiveStrategyProfile(
        profile_id=UUID("00000000-0000-0000-0000-000000006701"),
        organization_id=_ORGANIZATION_ID,
        owner_user_id=strategy.user_id,
        strategy_id=strategy.strategy_id,
        mode=mode,
        exchange_connection_id=UUID("00000000-0000-0000-0000-000000006801"),
        sizing_method="fixed_quote",
        sizing_value=Decimal("50"),
        max_position_notional=Decimal("50"),
        max_orders_per_run=3,
        max_notional_per_run=Decimal("50"),
        readiness_status="ready",
        readiness_reason="ready_for_paper",
        created_at=datetime(2026, 5, 6, 8, 50, tzinfo=UTC),
        updated_at=datetime(2026, 5, 6, 8, 55, tzinfo=UTC),
    )


def _valid_risk_policy_config() -> RlRiskSizingPolicyConfig:
    return RlRiskSizingPolicyConfig(
        sizing_method="fixed_quote",
        base_quote_notional=Decimal("25"),
        max_position_notional=Decimal("100"),
        max_daily_loss_notional=Decimal("50"),
        max_drawdown_pct=Decimal("0.10"),
        max_turnover_notional=Decimal("500"),
        max_exposure_notional=Decimal("250"),
        min_expected_pnl_pct=Decimal("0.01"),
        min_confidence=Decimal("0.80"),
        take_profit_pct=Decimal("0.05"),
        stop_loss_pct=Decimal("0.02"),
        trailing_stop_pct=Decimal("0.03"),
    )


def _strategy_spec_payload(*, symbol: str) -> dict[str, Any]:
    return {
        "instrument_id": {
            "market_id": 1,
            "symbol": symbol,
        },
        "instrument_key": f"binance:spot:{symbol}",
        "market_type": "spot",
        "timeframe": "1m",
        "indicators": [
            {
                "name": "MA",
                "params": {
                    "fast": 20,
                    "slow": 50,
                },
            }
        ],
        "signal_template": "MA(20,50)",
    }
