from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_strategies_router
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExecutionIntentRepository,
    InMemoryPaperAccountingRepository,
)
from trading.contexts.live_execution.application import (
    CapitalReservationPaperAccountingService,
    ExecutionIngressService,
)
from trading.contexts.rl_trading.adapters.outbound.persistence import (
    InMemoryRlLiveTickerEntitlementRepository,
    InMemoryRlRiskSizingPolicyRepository,
)
from trading.contexts.rl_trading.domain.live_entitlements import (
    RlLiveTickerEntitlementService,
)
from trading.contexts.rl_trading.domain.risk_sizing_policy import RlRiskSizingPolicyService
from trading.contexts.strategy.adapters.outbound.persistence.in_memory import (
    InMemoryLiveStrategyProfileRepository,
    InMemoryStrategyEventRepository,
    InMemoryStrategyRepository,
    InMemoryStrategyRunRepository,
)
from trading.contexts.strategy.application import (
    CloneStrategyUseCase,
    CreateStrategyFromBacktestVariantResult,
    CreateStrategyUseCase,
    CurrentUser,
    CurrentUserProvider,
    DeleteStrategyUseCase,
    ExchangeConnectionReadiness,
    ExchangeConnectionReadinessContext,
    GetMyStrategyUseCase,
    ListMyStrategiesUseCase,
    LiveStrategyProfileService,
    RestartStrategyUseCase,
    RunStrategyUseCase,
    StopStrategyUseCase,
)
from trading.contexts.strategy.domain.entities import (
    Strategy,
    StrategyBacktestVariantProvenance,
    StrategySpecV1,
)
from trading.shared_kernel.primitives import OrganizationId, PaidLevel, UserId

_ORGANIZATION_ID = OrganizationId.from_string("00000000-0000-4000-8000-000000000700")


class _SequenceClock:
    """
    Deterministic UTC clock stub returning preconfigured timestamps in FIFO order.
    """

    def __init__(self, *, start: datetime, steps: int) -> None:
        """
        Initialize deterministic timestamp queue.

        Args:
            start: Start UTC datetime.
            steps: Number of timestamps to pre-generate with one-minute step.
        Returns:
            None.
        Assumptions:
            Generated timestamps are timezone-aware UTC values.
        Raises:
            ValueError: If invalid arguments are provided.
        Side Effects:
            Stores mutable internal queue state.
        """
        if steps <= 0:
            raise ValueError("_SequenceClock steps must be > 0")
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("_SequenceClock start must be timezone-aware UTC datetime")
        self._values = [start + timedelta(minutes=index) for index in range(steps)]

    def now(self) -> datetime:
        """
        Return next configured UTC datetime value.

        Args:
            None.
        Returns:
            datetime: Next queued timestamp.
        Assumptions:
            Tests pre-generate enough timestamps.
        Raises:
            ValueError: If queue is exhausted.
        Side Effects:
            Pops one timestamp from internal queue.
        """
        if not self._values:
            raise ValueError("_SequenceClock exhausted")
        return self._values.pop(0)


class _StaticCurrentUserProvider(CurrentUserProvider):
    """
    Strategy CurrentUserProvider implementation returning one pre-resolved user context.
    """

    def __init__(self, *, user_id: UserId) -> None:
        """
        Store current user identifier for request scope.

        Args:
            user_id: Current authenticated user identifier.
        Returns:
            None.
        Assumptions:
            User id comes from deterministic test request header parsing.
        Raises:
            ValueError: If user_id is missing.
        Side Effects:
            None.
        """
        if user_id is None:  # type: ignore[truthy-bool]
            raise ValueError("_StaticCurrentUserProvider requires user_id")
        self._user_id = user_id

    def require_current_user(self) -> CurrentUser:
        """
        Return current strategy user context.

        Args:
            None.
        Returns:
            CurrentUser: Strategy current user context.
        Assumptions:
            Identity validation is out of scope for this route-contract unit test.
        Raises:
            None.
        Side Effects:
            None.
        """
        return CurrentUser(organization_id=_ORGANIZATION_ID, user_id=self._user_id)


class _HeaderCurrentUserDependency:
    """
    Request dependency reading `X-User-Id` header and producing Strategy CurrentUserProvider.
    """

    def __call__(self, request: Request) -> CurrentUserProvider:
        """
        Resolve Strategy CurrentUserProvider from request header.

        Args:
            request: HTTP request carrying `X-User-Id` header.
        Returns:
            CurrentUserProvider: Header-derived current user provider.
        Assumptions:
            Header value is valid UUID string used by tests.
        Raises:
            ValueError: If header is missing or malformed UUID string.
        Side Effects:
            None.
        """
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise ValueError("x-user-id header is required for strategy route tests")

        principal = CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )
        return _StaticCurrentUserProvider(user_id=principal.user_id)


class _HeaderCurrentUserPrincipalDependency:
    def __init__(self, *, session_created_at: datetime | None = None) -> None:
        self._session_created_at = session_created_at

    def __call__(self, request: Request) -> CurrentUserPrincipal:
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise ValueError("x-user-id header is required for strategy route tests")
        raw_paid_level = request.headers.get("x-paid-level", "free")
        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel(raw_paid_level),
            session_created_at=self._session_created_at,
        )


class _StaticExchangeReadinessChecker:
    def __init__(
        self,
        *,
        eligible: bool,
        reason: str = "ready_for_trading",
        exchange_name: str = "binance",
        market_type: str = "spot",
    ) -> None:
        self._eligible = eligible
        self._reason = reason
        self._exchange_name = exchange_name
        self._market_type = market_type
        self.contexts: list[ExchangeConnectionReadinessContext | None] = []

    def check_trading_ready(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        context: ExchangeConnectionReadinessContext | None = None,
    ) -> ExchangeConnectionReadiness:
        self.contexts.append(context)
        return ExchangeConnectionReadiness(
            eligible=self._eligible,
            reason=self._reason,
            exchange_name=self._exchange_name,
            market_type=self._market_type,
        )


def _build_client() -> TestClient:
    """
    Build TestClient with fully wired in-memory Strategy API router.

    Args:
        None.
    Returns:
        TestClient: Ready API test client.
    Assumptions:
        Shared in-memory repositories are sufficient for route-contract tests.
    Raises:
        ValueError: If dependency construction is invalid.
    Side Effects:
        Creates in-memory FastAPI app and mutable repository state.
    """
    strategy_repository = InMemoryStrategyRepository()
    run_repository = InMemoryStrategyRunRepository()
    event_repository = InMemoryStrategyEventRepository()
    profile_repository = InMemoryLiveStrategyProfileRepository()
    clock = _SequenceClock(
        start=datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc),
        steps=60,
    )

    router = build_strategies_router(
        create_use_case=CreateStrategyUseCase(
            repository=strategy_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        clone_use_case=CloneStrategyUseCase(
            repository=strategy_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        list_use_case=ListMyStrategiesUseCase(repository=strategy_repository),
        get_use_case=GetMyStrategyUseCase(repository=strategy_repository),
        run_use_case=RunStrategyUseCase(
            strategy_repository=strategy_repository,
            run_repository=run_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        stop_use_case=StopStrategyUseCase(
            strategy_repository=strategy_repository,
            run_repository=run_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        restart_use_case=RestartStrategyUseCase(
            strategy_repository=strategy_repository,
            run_repository=run_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        delete_use_case=DeleteStrategyUseCase(
            repository=strategy_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        current_user_provider_dependency=_HeaderCurrentUserDependency(),
        live_profile_service=LiveStrategyProfileService(
            strategy_repository=strategy_repository,
            profile_repository=profile_repository,
            event_repository=event_repository,
            clock=clock,
            exchange_connection_checker=_StaticExchangeReadinessChecker(eligible=True),
        ),
        current_user_principal_dependency=_HeaderCurrentUserPrincipalDependency(
            session_created_at=datetime.now(timezone.utc),
        ),
        rl_risk_sizing_policy_service=RlRiskSizingPolicyService(
            repository=InMemoryRlRiskSizingPolicyRepository(),
        ),
    )

    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(router)
    return TestClient(app)


def _build_live_profile_client(
    *,
    session_created_at: datetime | None,
    exchange_eligible: bool = True,
    create_strategy_from_variant_use_case: Any | None = None,
    exchange_connection_checker: _StaticExchangeReadinessChecker | None = None,
) -> TestClient:
    strategy_repository = InMemoryStrategyRepository()
    if create_strategy_from_variant_use_case is not None and hasattr(
        create_strategy_from_variant_use_case,
        "bind_strategy_repository",
    ):
        create_strategy_from_variant_use_case.bind_strategy_repository(strategy_repository)
    event_repository = InMemoryStrategyEventRepository()
    run_repository = InMemoryStrategyRunRepository()
    profile_repository = InMemoryLiveStrategyProfileRepository()
    rl_entitlement_repository = InMemoryRlLiveTickerEntitlementRepository()
    clock = _SequenceClock(
        start=datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc),
        steps=60,
    )
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_strategies_router(
            create_use_case=CreateStrategyUseCase(
                repository=strategy_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            clone_use_case=CloneStrategyUseCase(
                repository=strategy_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            list_use_case=ListMyStrategiesUseCase(repository=strategy_repository),
            get_use_case=GetMyStrategyUseCase(repository=strategy_repository),
            run_use_case=RunStrategyUseCase(
                strategy_repository=strategy_repository,
                run_repository=run_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            stop_use_case=StopStrategyUseCase(
                strategy_repository=strategy_repository,
                run_repository=run_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            restart_use_case=RestartStrategyUseCase(
                strategy_repository=strategy_repository,
                run_repository=run_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            delete_use_case=DeleteStrategyUseCase(
                repository=strategy_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            current_user_provider_dependency=_HeaderCurrentUserDependency(),
            live_profile_service=LiveStrategyProfileService(
                strategy_repository=strategy_repository,
                profile_repository=profile_repository,
                event_repository=event_repository,
                clock=clock,
                exchange_connection_checker=(
                    exchange_connection_checker
                    or _StaticExchangeReadinessChecker(
                        eligible=exchange_eligible,
                        reason=(
                            "ready_for_trading"
                            if exchange_eligible
                            else "exchange_connection_not_ready_for_trading"
                        ),
                    )
                ),
            ),
            current_user_principal_dependency=_HeaderCurrentUserPrincipalDependency(
                session_created_at=session_created_at,
            ),
            create_strategy_from_variant_use_case=create_strategy_from_variant_use_case,
            strategy_run_repository=run_repository,
            live_profile_repository=profile_repository,
            rl_live_ticker_entitlement_service=RlLiveTickerEntitlementService(
                repository=rl_entitlement_repository,
            ),
            rl_risk_sizing_policy_service=RlRiskSizingPolicyService(
                repository=InMemoryRlRiskSizingPolicyRepository(),
            ),
        )
    )
    return TestClient(app)


def _build_manual_execution_client() -> (
    tuple[
        TestClient,
        InMemoryExecutionIntentRepository,
        InMemoryPaperAccountingRepository,
    ]
):
    strategy_repository = InMemoryStrategyRepository()
    event_repository = InMemoryStrategyEventRepository()
    run_repository = InMemoryStrategyRunRepository()
    profile_repository = InMemoryLiveStrategyProfileRepository()
    execution_repository = InMemoryExecutionIntentRepository()
    paper_repository = InMemoryPaperAccountingRepository()
    clock = _SequenceClock(
        start=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
        steps=120,
    )
    live_clock = _SequenceClock(
        start=datetime(2026, 6, 18, 13, 0, tzinfo=timezone.utc),
        steps=120,
    )
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_strategies_router(
            create_use_case=CreateStrategyUseCase(
                repository=strategy_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            clone_use_case=CloneStrategyUseCase(
                repository=strategy_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            list_use_case=ListMyStrategiesUseCase(repository=strategy_repository),
            get_use_case=GetMyStrategyUseCase(repository=strategy_repository),
            run_use_case=RunStrategyUseCase(
                strategy_repository=strategy_repository,
                run_repository=run_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            stop_use_case=StopStrategyUseCase(
                strategy_repository=strategy_repository,
                run_repository=run_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            restart_use_case=RestartStrategyUseCase(
                strategy_repository=strategy_repository,
                run_repository=run_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            delete_use_case=DeleteStrategyUseCase(
                repository=strategy_repository,
                event_repository=event_repository,
                clock=clock,
            ),
            current_user_provider_dependency=_HeaderCurrentUserDependency(),
            live_profile_service=LiveStrategyProfileService(
                strategy_repository=strategy_repository,
                profile_repository=profile_repository,
                event_repository=event_repository,
                clock=clock,
                exchange_connection_checker=_StaticExchangeReadinessChecker(eligible=True),
            ),
            current_user_principal_dependency=_HeaderCurrentUserPrincipalDependency(
                session_created_at=datetime.now(timezone.utc),
            ),
            strategy_run_repository=run_repository,
            live_profile_repository=profile_repository,
            execution_ingress_service=ExecutionIngressService(
                repository=execution_repository,
                clock=live_clock,
            ),
            paper_accounting_service=CapitalReservationPaperAccountingService(
                repository=paper_repository,
                account_projection_repository=None,
                clock=live_clock,
            ),
        )
    )
    return TestClient(app), execution_repository, paper_repository


def test_strategies_list_endpoint_returns_deterministic_sort_order() -> None:
    """
    Verify `/strategies` response is deterministically sorted by `created_at`, then `strategy_id`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Use-case and repository enforce deterministic ordering contract.
    Raises:
        AssertionError: If ordering is unstable or differs from documented keys.
    Side Effects:
        None.
    """
    client = _build_client()
    headers = {"x-user-id": "00000000-0000-0000-0000-000000001111"}

    create_payload = _build_create_payload(symbol="BTCUSDT")
    response_a = client.post("/strategies", json=create_payload, headers=headers)
    assert response_a.status_code == 201

    response_b = client.post("/strategies", json=create_payload, headers=headers)
    assert response_b.status_code == 201

    list_response = client.get("/strategies", headers=headers)
    assert list_response.status_code == 200
    items = list_response.json()
    assert len(items) == 2

    sorted_items = sorted(
        items,
        key=lambda item: (item["created_at"], item["strategy_id"]),
    )
    assert items == sorted_items


def test_live_profile_defaults_to_monitor_only_and_blocks_live_without_recent_auth() -> None:
    client = _build_live_profile_client(session_created_at=None)
    headers = {"x-user-id": "00000000-0000-0000-0000-000000001333"}
    created_strategy = client.post(
        "/strategies",
        json=_build_create_payload(symbol="BTCUSDT"),
        headers=headers,
    )
    assert created_strategy.status_code == 201
    strategy_id = created_strategy.json()["strategy_id"]

    created_profile = client.post(f"/strategies/{strategy_id}/live-profile", headers=headers)
    assert created_profile.status_code == 201
    default_payload = created_profile.json()
    assert default_payload["mode"] == "monitor_only"
    assert default_payload["exchange_connection_id"] is None
    assert default_payload["readiness_status"] == "ready"
    assert default_payload["readiness_reason"] == "monitor_only_no_exchange_submit"

    blocked_live = client.put(
        f"/strategies/{strategy_id}/live-profile",
        json={
            "mode": "live",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e001",
            "sizing_method": "fixed_quote",
            "sizing_value": "25",
            "max_orders_per_run": 1,
            "max_notional_per_run": "25",
        },
        headers=headers,
    )
    assert blocked_live.status_code == 200
    blocked_payload = blocked_live.json()
    assert blocked_payload["mode"] == "live"
    assert blocked_payload["readiness_status"] == "blocked"
    assert blocked_payload["readiness_reason"] == "recent_auth_required"
    assert "api_secret" not in blocked_live.text


def test_stage14_rl_risk_policy_api_persists_valid_policy_and_synthetic_exits() -> None:
    client = _build_client()
    headers = {"x-user-id": "00000000-0000-0000-0000-000000001444"}
    created_strategy = client.post(
        "/strategies",
        json=_build_create_payload(symbol="BTCUSDT"),
        headers=headers,
    )
    strategy_id = created_strategy.json()["strategy_id"]

    response = client.put(
        f"/strategies/{strategy_id}/rl-risk-policy",
        json={
            "active": True,
            "sizing_method": "fixed_quote",
            "base_quote_notional": "25",
            "max_position_notional": "100",
            "max_daily_loss_notional": "50",
            "max_drawdown_pct": "0.10",
            "max_turnover_notional": "500",
            "max_exposure_notional": "250",
            "min_expected_pnl_pct": "0.01",
            "min_confidence": "0.80",
            "take_profit_pct": "0.05",
            "stop_loss_pct": "0.02",
            "trailing_stop_pct": "0.03",
        },
        headers=headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["validation_status"] == "ready"
    assert payload["exchange_name"] == "binance"
    assert payload["market_type"] == "spot"
    assert payload["symbol"] == "BTCUSDT"
    assert [rule["rule_type"] for rule in payload["synthetic_exit_rules"]] == [
        "take_profit",
        "stop_loss",
        "trailing_stop",
    ]
    assert all(rule["platform_side"] for rule in payload["synthetic_exit_rules"])

    loaded = client.get(f"/strategies/{strategy_id}/rl-risk-policy", headers=headers)
    assert loaded.status_code == 200
    assert loaded.json()["policy_id"] == payload["policy_id"]


def test_stage14_invalid_saved_policy_blocks_activation_without_exchange_submit() -> None:
    client = _build_live_profile_client(session_created_at=datetime.now(timezone.utc))
    headers = {"x-user-id": "00000000-0000-0000-0000-000000001445"}
    created_strategy = client.post(
        "/strategies",
        json=_build_create_payload(symbol="BTCUSDT"),
        headers=headers,
    )
    strategy_id = created_strategy.json()["strategy_id"]

    invalid_policy = client.put(
        f"/strategies/{strategy_id}/rl-risk-policy",
        json={
            "active": True,
            "sizing_method": "fixed_quote",
            "base_quote_notional": "25",
            "max_position_notional": "100",
            "max_daily_loss_notional": "50",
            "max_drawdown_pct": "0.10",
            "max_turnover_notional": "500",
            "max_exposure_notional": "250",
            "min_expected_pnl_pct": "0.01",
        },
        headers=headers,
    )
    assert invalid_policy.status_code == 200
    assert invalid_policy.json()["validation_status"] == "blocked"
    assert "rl_risk_policy_stop_loss_required" in invalid_policy.json()["validation_reasons"]

    paper = client.put(
        f"/strategies/{strategy_id}/live-profile",
        json={
            "mode": "paper",
            "sizing_method": "fixed_quote",
            "sizing_value": "25",
            "max_orders_per_run": 1,
            "max_notional_per_run": "25",
        },
        headers=headers,
    )

    assert paper.status_code == 200
    assert paper.json()["readiness_status"] == "blocked"
    assert paper.json()["readiness_reason"] == "rl_risk_policy_stop_loss_required"


def test_live_strategy_profile_allows_paper_and_recent_auth_live_with_ready_connection() -> None:
    client = _build_live_profile_client(session_created_at=datetime.now(timezone.utc))
    headers = {"x-user-id": "00000000-0000-0000-0000-000000001334"}
    created_strategy = client.post(
        "/strategies",
        json=_build_create_payload(symbol="ETHUSDT"),
        headers=headers,
    )
    assert created_strategy.status_code == 201
    strategy_id = created_strategy.json()["strategy_id"]

    paper = client.put(
        f"/strategies/{strategy_id}/live-profile",
        json={
            "mode": "paper",
            "sizing_method": "fixed_equity_pct",
            "sizing_value": "0.15",
            "max_orders_per_run": 3,
            "max_notional_per_run": "100",
        },
        headers=headers,
    )
    assert paper.status_code == 200
    assert paper.json()["readiness_reason"] == "paper_no_exchange_submit"

    live = client.put(
        f"/strategies/{strategy_id}/live-profile",
        json={
            "mode": "live",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e002",
            "sizing_method": "fixed_quote",
            "sizing_value": "10",
            "max_position_notional": "100",
            "max_orders_per_run": 2,
            "max_notional_per_run": "50",
        },
        headers=headers,
    )
    assert live.status_code == 200
    assert live.json()["readiness_status"] == "ready"


def test_live_strategy_profile_blocks_second_free_rl_live_ticker() -> None:
    client = _build_live_profile_client(session_created_at=datetime.now(timezone.utc))
    headers = {
        "x-user-id": "00000000-0000-0000-0000-000000001434",
        "x-paid-level": "free",
    }
    first_strategy = client.post(
        "/strategies",
        json=_build_create_payload(symbol="BTCUSDT"),
        headers=headers,
    )
    second_strategy = client.post(
        "/strategies",
        json=_build_create_payload(symbol="ETHUSDT"),
        headers=headers,
    )
    assert first_strategy.status_code == 201
    assert second_strategy.status_code == 201

    first_live = client.put(
        f"/strategies/{first_strategy.json()['strategy_id']}/live-profile",
        json={
            "mode": "live",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e101",
            "sizing_method": "fixed_quote",
            "sizing_value": "10",
            "max_orders_per_run": 1,
            "max_notional_per_run": "10",
        },
        headers=headers,
    )
    assert first_live.status_code == 200
    assert first_live.json()["readiness_status"] == "ready"

    second_live = client.put(
        f"/strategies/{second_strategy.json()['strategy_id']}/live-profile",
        json={
            "mode": "live",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e102",
            "sizing_method": "fixed_quote",
            "sizing_value": "10",
            "max_orders_per_run": 1,
            "max_notional_per_run": "10",
        },
        headers=headers,
    )
    assert second_live.status_code == 200
    assert second_live.json()["mode"] == "live"
    assert second_live.json()["readiness_status"] == "blocked"
    assert second_live.json()["readiness_reason"] == "rl_live_ticker_quota_exceeded"


def test_live_strategy_profile_does_not_count_paper_and_releases_stopped_slot() -> None:
    client = _build_live_profile_client(session_created_at=datetime.now(timezone.utc))
    headers = {
        "x-user-id": "00000000-0000-0000-0000-000000001435",
        "x-paid-level": "free",
    }
    first_strategy = client.post(
        "/strategies",
        json=_build_create_payload(symbol="BTCUSDT"),
        headers=headers,
    )
    second_strategy = client.post(
        "/strategies",
        json=_build_create_payload(symbol="ETHUSDT"),
        headers=headers,
    )
    assert first_strategy.status_code == 201
    assert second_strategy.status_code == 201
    first_id = first_strategy.json()["strategy_id"]
    second_id = second_strategy.json()["strategy_id"]

    paper = client.put(
        f"/strategies/{first_id}/live-profile",
        json={
            "mode": "paper",
            "sizing_method": "fixed_quote",
            "sizing_value": "10",
            "max_orders_per_run": 1,
            "max_notional_per_run": "10",
        },
        headers=headers,
    )
    assert paper.status_code == 200
    assert paper.json()["readiness_status"] == "ready"

    second_live = client.put(
        f"/strategies/{second_id}/live-profile",
        json={
            "mode": "live",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e202",
            "sizing_method": "fixed_quote",
            "sizing_value": "10",
            "max_orders_per_run": 1,
            "max_notional_per_run": "10",
        },
        headers=headers,
    )
    assert second_live.status_code == 200
    assert second_live.json()["readiness_status"] == "ready"

    stopped = client.put(
        f"/strategies/{second_id}/live-profile",
        json={
            "mode": "monitor_only",
            "sizing_method": "fixed_quote",
            "sizing_value": "0",
            "max_orders_per_run": 0,
            "max_notional_per_run": "0",
        },
        headers=headers,
    )
    assert stopped.status_code == 200
    assert stopped.json()["readiness_reason"] == "monitor_only_no_exchange_submit"

    first_live = client.put(
        f"/strategies/{first_id}/live-profile",
        json={
            "mode": "live",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e201",
            "sizing_method": "fixed_quote",
            "sizing_value": "10",
            "max_orders_per_run": 1,
            "max_notional_per_run": "10",
        },
        headers=headers,
    )
    assert first_live.status_code == 200
    assert first_live.json()["readiness_status"] == "ready"


def test_live_strategy_profile_blocks_base_paid_level_fail_closed() -> None:
    client = _build_live_profile_client(session_created_at=datetime.now(timezone.utc))
    headers = {
        "x-user-id": "00000000-0000-0000-0000-000000001436",
        "x-paid-level": "base",
    }
    created_strategy = client.post(
        "/strategies",
        json=_build_create_payload(symbol="BTCUSDT"),
        headers=headers,
    )
    assert created_strategy.status_code == 201

    live = client.put(
        f"/strategies/{created_strategy.json()['strategy_id']}/live-profile",
        json={
            "mode": "live",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e301",
            "sizing_method": "fixed_quote",
            "sizing_value": "10",
            "max_orders_per_run": 1,
            "max_notional_per_run": "10",
        },
        headers=headers,
    )
    assert live.status_code == 200
    assert live.json()["readiness_status"] == "blocked"
    assert live.json()["readiness_reason"] == "rl_live_ticker_paid_level_base_fail_closed"


def test_launch_from_backtest_variant_creates_profile_and_run_config() -> None:
    use_case = _FakeCreateStrategyFromVariantUseCase()
    client = _build_live_profile_client(
        session_created_at=datetime.now(timezone.utc),
        create_strategy_from_variant_use_case=use_case,
    )
    headers = {"x-user-id": "00000000-0000-0000-0000-000000000205"}

    launched = client.post(
        "/strategies/launch-from-backtest-variant",
        headers={**headers, "Idempotency-Key": "launch-paper-1"},
        json={
            "job_id": "00000000-0000-0000-0000-00000000b001",
            "variant_key": "job_demo",
            "mode": "paper",
            "market_type": "spot",
            "symbol": "BTCUSDT",
            "capital_allocation_usd": "50",
            "entry_sizing": "fixed_quote",
            "risk_mode": "single_position_cap",
            "direction": "long",
        },
    )

    assert launched.status_code == 201
    payload = launched.json()
    assert payload["status"] == "started"
    assert payload["strategy"]["spec"]["instrument_key"] == "binance:spot:BTCUSDT"
    assert payload["profile"]["mode"] == "paper"
    assert payload["profile"]["sizing_value"] == "50"
    assert payload["profile"]["max_notional_per_run"] == "50"
    assert payload["profile"]["readiness_reason"] == "paper_no_exchange_submit"
    assert payload["run"]["state"] == "starting"
    assert payload["run"]["metadata"]["launch_config"]["symbol"] == "BTCUSDT"
    assert payload["run"]["metadata"]["launch_config"]["capital_allocation_usd"] == "50"
    assert payload["run"]["metadata"]["launch_config"]["direction"] == "long"
    assert payload["provenance"]["source_variant_key"] == "job_demo"
    assert use_case.calls == (("job_demo", "launch-paper-1", "paper"),)


def test_launch_from_backtest_variant_preserves_long_short_direction_mode() -> None:
    use_case = _FakeCreateStrategyFromVariantUseCase()
    client = _build_live_profile_client(
        session_created_at=datetime.now(timezone.utc),
        create_strategy_from_variant_use_case=use_case,
    )
    headers = {"x-user-id": "00000000-0000-0000-0000-000000000205"}

    launched = client.post(
        "/strategies/launch-from-backtest-variant",
        headers={**headers, "Idempotency-Key": "launch-paper-long-short"},
        json={
            "job_id": "00000000-0000-0000-0000-00000000b001",
            "variant_key": "job_demo",
            "mode": "paper",
            "market_type": "futures",
            "symbol": "BTCUSDT",
            "capital_allocation_usd": "50",
            "entry_sizing": "fixed_quote",
            "risk_mode": "single_position_cap",
            "direction": "long_short_reversal",
        },
    )

    assert launched.status_code == 201
    payload = launched.json()
    assert payload["run"]["metadata"]["launch_config"]["direction"] == "long_short_reversal"
    assert payload["profile"]["readiness_reason"] == "paper_no_exchange_submit"


def test_launch_from_backtest_variant_blocks_paper_spot_long_short() -> None:
    client = _build_live_profile_client(session_created_at=datetime.now(timezone.utc))
    response = client.post(
        "/strategies/launch-from-backtest-variant",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000205",
            "Idempotency-Key": "launch-paper-spot-long-short",
        },
        json={
            "job_id": "00000000-0000-0000-0000-00000000b001",
            "variant_key": "job_demo",
            "mode": "paper",
            "market_type": "spot",
            "symbol": "BTCUSDT",
            "capital_allocation_usd": "50",
            "entry_sizing": "fixed_quote",
            "risk_mode": "single_position_cap",
            "direction": "long_short_reversal",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "strategy_launch.invalid_config"
    assert (
        response.json()["error"]["details"]["reason"] == "short_direction_requires_futures_market"
    )


def test_manual_entry_paper_creates_idempotent_source_intent_and_paper_order() -> None:
    client, execution_repository, paper_repository = _build_manual_execution_client()
    headers = {"x-user-id": "00000000-0000-0000-0000-000000000805"}
    created = client.post(
        "/strategies",
        json=_build_create_payload(symbol="BTCUSDT"),
        headers=headers,
    )
    assert created.status_code == 201
    strategy_id = created.json()["strategy_id"]
    profile = client.put(
        f"/strategies/{strategy_id}/live-profile",
        json={
            "mode": "paper",
            "sizing_method": "fixed_quote",
            "sizing_value": "50",
            "max_position_notional": "50",
            "max_orders_per_run": 1,
            "max_notional_per_run": "50",
        },
        headers=headers,
    )
    assert profile.status_code == 200
    run = client.post(f"/strategies/{strategy_id}/run", headers=headers)
    assert run.status_code == 200

    request_headers = {**headers, "Idempotency-Key": "manual-paper-entry-1"}
    payload = {"client_request_id": "manual-paper-entry-1", "reference_price": "50000"}
    first = client.post(
        f"/strategies/{strategy_id}/manual-entry",
        json=payload,
        headers=request_headers,
    )
    replay = client.post(
        f"/strategies/{strategy_id}/manual-entry",
        json=payload,
        headers=request_headers,
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    first_payload = first.json()
    replay_payload = replay.json()
    assert first_payload["status"] == "accepted"
    assert first_payload["risk_status"] == "rejected"
    assert first_payload["risk_reason"] == "paper_no_exchange_submit"
    assert first_payload["paper_order_state"] == "filled"
    assert replay_payload["duplicate"] is True
    assert replay_payload["source_event_id"] == first_payload["source_event_id"]
    assert replay_payload["intent_id"] == first_payload["intent_id"]
    assert len(execution_repository.source_events) == 1
    assert len(execution_repository.intents) == 1
    assert len(paper_repository.orders) == 1
    assert paper_repository.orders[0].source_event_id == UUID(first_payload["source_event_id"])
    assert len(paper_repository.fills) == 1
    assert len(paper_repository.accounting) == 1


def test_manual_entry_without_active_run_returns_conflict() -> None:
    client, execution_repository, paper_repository = _build_manual_execution_client()
    headers = {"x-user-id": "00000000-0000-0000-0000-000000000806"}
    created = client.post(
        "/strategies",
        json=_build_create_payload(symbol="BTCUSDT"),
        headers=headers,
    )
    assert created.status_code == 201
    strategy_id = created.json()["strategy_id"]
    profile = client.put(
        f"/strategies/{strategy_id}/live-profile",
        json={
            "mode": "paper",
            "sizing_method": "fixed_quote",
            "sizing_value": "50",
            "max_position_notional": "50",
            "max_orders_per_run": 1,
            "max_notional_per_run": "50",
        },
        headers=headers,
    )
    assert profile.status_code == 200

    response = client.post(
        f"/strategies/{strategy_id}/manual-entry",
        json={"client_request_id": "manual-paper-entry-no-run"},
        headers={**headers, "Idempotency-Key": "manual-paper-entry-no-run"},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "strategy_manual_execution.blocked"
    assert response.json()["error"]["details"]["reason"] == "strategy_run_inactive"
    assert len(execution_repository.source_events) == 0
    assert len(paper_repository.orders) == 0


def test_launch_from_backtest_variant_blocks_testnet_without_exchange() -> None:
    client = _build_live_profile_client(session_created_at=datetime.now(timezone.utc))
    response = client.post(
        "/strategies/launch-from-backtest-variant",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000205",
            "Idempotency-Key": "launch-testnet-1",
        },
        json={
            "job_id": "00000000-0000-0000-0000-00000000b001",
            "variant_key": "job_demo",
            "mode": "testnet",
            "market_type": "spot",
            "symbol": "BTCUSDT",
            "capital_allocation_usd": "50",
            "entry_sizing": "fixed_quote",
            "risk_mode": "single_position_cap",
            "direction": "long",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "strategy_launch.invalid_config"
    assert response.json()["error"]["details"]["reason"] == "exchange_connection_required"


def test_launch_from_backtest_variant_blocks_testnet_spot_long_short() -> None:
    client = _build_live_profile_client(session_created_at=datetime.now(timezone.utc))
    response = client.post(
        "/strategies/launch-from-backtest-variant",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000205",
            "Idempotency-Key": "launch-testnet-spot-long-short",
        },
        json={
            "job_id": "00000000-0000-0000-0000-00000000b001",
            "variant_key": "job_demo",
            "mode": "testnet",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e101",
            "market_type": "spot",
            "symbol": "BTCUSDT",
            "capital_allocation_usd": "50",
            "entry_sizing": "fixed_quote",
            "risk_mode": "single_position_cap",
            "direction": "long_short_reversal",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "strategy_launch.invalid_config"
    assert (
        response.json()["error"]["details"]["reason"] == "short_direction_requires_futures_market"
    )


def test_launch_from_backtest_variant_blocks_unsafe_testnet_futures_short() -> None:
    use_case = _FakeCreateStrategyFromVariantUseCase()
    checker = _StaticExchangeReadinessChecker(
        eligible=False,
        reason="unsafe_futures_short",
        market_type="futures",
    )
    client = _build_live_profile_client(
        session_created_at=datetime.now(timezone.utc),
        create_strategy_from_variant_use_case=use_case,
        exchange_connection_checker=checker,
    )
    response = client.post(
        "/strategies/launch-from-backtest-variant",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000205",
            "Idempotency-Key": "launch-testnet-short-unsafe",
        },
        json={
            "job_id": "00000000-0000-0000-0000-00000000b001",
            "variant_key": "job_demo",
            "mode": "testnet",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e101",
            "market_type": "futures",
            "symbol": "BTCUSDT",
            "capital_allocation_usd": "50",
            "entry_sizing": "fixed_quote",
            "risk_mode": "single_position_cap",
            "direction": "short",
        },
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "strategy_launch.readiness_blocked"
    assert response.json()["error"]["details"]["reason"] == "unsafe_futures_short"
    assert checker.contexts
    context = checker.contexts[-1]
    assert context is not None
    assert context.mode == "testnet"
    assert context.market_type == "futures"
    assert context.symbol == "BTCUSDT"
    assert context.direction == "short"
    assert str(context.notional) == "50"


def test_launch_from_backtest_variant_accepts_verified_testnet_futures_short() -> None:
    use_case = _FakeCreateStrategyFromVariantUseCase()
    checker = _StaticExchangeReadinessChecker(
        eligible=True,
        reason="safe_testnet_futures_short_1x_isolated_verified",
        market_type="futures",
    )
    client = _build_live_profile_client(
        session_created_at=datetime.now(timezone.utc),
        create_strategy_from_variant_use_case=use_case,
        exchange_connection_checker=checker,
    )
    response = client.post(
        "/strategies/launch-from-backtest-variant",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000205",
            "Idempotency-Key": "launch-testnet-short-safe",
        },
        json={
            "job_id": "00000000-0000-0000-0000-00000000b001",
            "variant_key": "job_demo",
            "mode": "testnet",
            "exchange_connection_id": "00000000-0000-0000-0000-00000000e102",
            "market_type": "futures",
            "symbol": "BTCUSDT",
            "capital_allocation_usd": "50",
            "entry_sizing": "fixed_quote",
            "risk_mode": "single_position_cap",
            "direction": "short",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["profile"]["mode"] == "testnet"
    assert (
        payload["profile"]["readiness_reason"] == "safe_testnet_futures_short_1x_isolated_verified"
    )
    assert payload["run"]["metadata"]["launch_config"]["direction"] == "short"
    assert checker.contexts[-1] is not None


def test_launch_from_backtest_variant_blocks_invalid_sizing_and_min_notional() -> None:
    client = _build_live_profile_client(session_created_at=datetime.now(timezone.utc))
    headers = {
        "x-user-id": "00000000-0000-0000-0000-000000000205",
        "Idempotency-Key": "launch-invalid-1",
    }
    payload = {
        "job_id": "00000000-0000-0000-0000-00000000b001",
        "variant_key": "job_demo",
        "mode": "paper",
        "market_type": "spot",
        "symbol": "BTCUSDT",
        "capital_allocation_usd": "50",
        "entry_sizing": "all_in",
        "risk_mode": "single_position_cap",
        "direction": "long",
    }

    invalid_sizing = client.post(
        "/strategies/launch-from-backtest-variant",
        headers=headers,
        json=payload,
    )
    low_notional = client.post(
        "/strategies/launch-from-backtest-variant",
        headers={**headers, "Idempotency-Key": "launch-invalid-2"},
        json={**payload, "entry_sizing": "fixed_quote", "capital_allocation_usd": "5"},
    )

    assert invalid_sizing.status_code == 422
    assert invalid_sizing.json()["error"]["details"]["reason"] == "invalid_entry_sizing"
    assert low_notional.status_code == 422
    assert (
        low_notional.json()["error"]["details"]["reason"] == "insufficient_allocation_min_notional"
    )


def test_strategy_get_endpoint_enforces_owner_only_visibility() -> None:
    """
    Verify `/strategies/{id}` rejects access by non-owner user.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Ownership checks are implemented in strategy use-case layer.
    Raises:
        AssertionError: If non-owner request is not rejected with deterministic forbidden payload.
    Side Effects:
        None.
    """
    client = _build_client()
    owner_headers = {"x-user-id": "00000000-0000-0000-0000-000000002222"}
    outsider_headers = {"x-user-id": "00000000-0000-0000-0000-000000003333"}

    create_response = client.post(
        "/strategies",
        json=_build_create_payload(symbol="ETHUSDT"),
        headers=owner_headers,
    )
    assert create_response.status_code == 201

    strategy_id = create_response.json()["strategy_id"]
    outsider_response = client.get(f"/strategies/{strategy_id}", headers=outsider_headers)

    assert outsider_response.status_code == 403
    assert outsider_response.json() == {
        "error": {
            "code": "forbidden",
            "message": "Strategy does not belong to current user",
            "details": {"strategy_id": strategy_id},
        }
    }


def test_strategy_run_stop_endpoints_expose_starting_and_stopping_states() -> None:
    """
    Verify run/stop endpoints expose `starting` and `stopping` states for live-runner orchestration.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Live runner worker finalizes `stopping -> stopped` asynchronously after API request.
    Raises:
        AssertionError: If run-control endpoint contract is violated.
    Side Effects:
        None.
    """
    client = _build_client()
    headers = {"x-user-id": "00000000-0000-0000-0000-000000004444"}

    create_response = client.post(
        "/strategies",
        json=_build_create_payload(symbol="SOLUSDT"),
        headers=headers,
    )
    assert create_response.status_code == 201

    strategy_id = create_response.json()["strategy_id"]

    first_run = client.post(f"/strategies/{strategy_id}/run", headers=headers)
    assert first_run.status_code == 200
    assert first_run.json()["state"] == "starting"

    conflict_run = client.post(f"/strategies/{strategy_id}/run", headers=headers)
    assert conflict_run.status_code == 409
    assert conflict_run.json()["error"]["code"] == "conflict"

    stop_response = client.post(f"/strategies/{strategy_id}/stop", headers=headers)
    assert stop_response.status_code == 200
    assert stop_response.json()["state"] == "stopping"

    second_run = client.post(f"/strategies/{strategy_id}/run", headers=headers)
    assert second_run.status_code == 409
    assert second_run.json()["error"]["code"] == "conflict"


def test_strategy_restart_endpoint_persists_pending_restart_and_rejects_duplicate() -> None:
    client = _build_client()
    headers = {"x-user-id": "00000000-0000-0000-0000-000000004445"}

    create_response = client.post(
        "/strategies",
        json=_build_create_payload(symbol="ADAUSDT"),
        headers=headers,
    )
    assert create_response.status_code == 201
    strategy_id = create_response.json()["strategy_id"]

    no_active_restart = client.post(f"/strategies/{strategy_id}/restart", headers=headers)
    assert no_active_restart.status_code == 409
    assert no_active_restart.json()["error"]["message"] == "Strategy has no active run to restart"

    first_run = client.post(f"/strategies/{strategy_id}/run", headers=headers)
    assert first_run.status_code == 200
    assert first_run.json()["state"] == "starting"

    restart = client.post(f"/strategies/{strategy_id}/restart", headers=headers)
    assert restart.status_code == 200
    restart_payload = restart.json()
    assert restart_payload["state"] == "stopping"
    assert restart_payload["metadata"]["restart"]["state"] == "pending_start"
    assert restart_payload["metadata"]["restart"]["operation_id"]

    duplicate_restart = client.post(f"/strategies/{strategy_id}/restart", headers=headers)
    assert duplicate_restart.status_code == 409
    assert duplicate_restart.json()["error"]["message"] == "Strategy restart is already pending"


class _FakeCreateStrategyFromVariantUseCase:
    def __init__(self) -> None:
        self.calls: tuple[tuple[str, str | None, str], ...] = ()
        self._strategy_repository: InMemoryStrategyRepository | None = None

    def bind_strategy_repository(self, repository: InMemoryStrategyRepository) -> None:
        self._strategy_repository = repository

    def execute(
        self,
        *,
        current_user: CurrentUser,
        job_id: UUID,
        variant_key: str,
        idempotency_key: str | None,
        launch_config: dict[str, Any] | None = None,
    ) -> CreateStrategyFromBacktestVariantResult:
        self.calls = (
            *self.calls,
            (variant_key, idempotency_key, str((launch_config or {}).get("mode"))),
        )
        strategy = Strategy.create(
            organization_id=current_user.organization_id,
            user_id=current_user.user_id,
            spec=StrategySpecV1.from_json(payload=_build_create_payload(symbol="BTCUSDT")),
            created_at=datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),
            strategy_id=UUID("00000000-0000-0000-0000-00000000c001"),
        )
        if self._strategy_repository is not None:
            self._strategy_repository.create(strategy=strategy)
        provenance = StrategyBacktestVariantProvenance(
            organization_id=current_user.organization_id,
            strategy_id=strategy.strategy_id,
            user_id=current_user.user_id,
            source_job_id=job_id,
            source_variant_key=variant_key,
            source_variant_hash="a" * 64,
            source_indicator_variant_hash="b" * 64,
            backtest_request_hash="d" * 64,
            backtest_result_config_hash="e" * 64,
            strategy_spec_hash="f" * 64,
            launch_request_hash="1" * 64,
            idempotency_key_hash="2" * 64,
            created_at=datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),
            metadata_json={"launch_config": dict(launch_config or {})},
        )
        return CreateStrategyFromBacktestVariantResult(
            strategy=strategy,
            provenance=provenance,
            duplicate=False,
        )


def _build_create_payload(*, symbol: str) -> dict[str, Any]:
    """
    Build deterministic `POST /strategies` payload fixture.

    Args:
        symbol: Symbol value for instrument payload.
    Returns:
        dict[str, Any]: Valid create-strategy payload.
    Assumptions:
        Payload follows immutable StrategySpecV1 contract.
    Raises:
        None.
    Side Effects:
        None.
    """
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
