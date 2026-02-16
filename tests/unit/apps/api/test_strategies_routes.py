from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_strategies_router
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.adapters.outbound.persistence.in_memory import (
    InMemoryStrategyEventRepository,
    InMemoryStrategyRepository,
    InMemoryStrategyRunRepository,
)
from trading.contexts.strategy.application import (
    CloneStrategyUseCase,
    CreateStrategyUseCase,
    CurrentUser,
    CurrentUserProvider,
    DeleteStrategyUseCase,
    GetMyStrategyUseCase,
    ListMyStrategiesUseCase,
    RunStrategyUseCase,
    StopStrategyUseCase,
)
from trading.shared_kernel.primitives import PaidLevel, UserId


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
        return CurrentUser(user_id=self._user_id)


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
        delete_use_case=DeleteStrategyUseCase(
            repository=strategy_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        current_user_provider_dependency=_HeaderCurrentUserDependency(),
    )

    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(router)
    return TestClient(app)



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



def test_strategy_run_stop_endpoints_allow_second_run_after_stop() -> None:
    """
    Verify run/stop endpoints enforce single-active-run invariant and allow second run after stop.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Run lifecycle follows deterministic transitions documented in Strategy API v1.
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
    assert first_run.json()["state"] == "running"

    conflict_run = client.post(f"/strategies/{strategy_id}/run", headers=headers)
    assert conflict_run.status_code == 409
    assert conflict_run.json()["error"]["code"] == "conflict"

    stop_response = client.post(f"/strategies/{strategy_id}/stop", headers=headers)
    assert stop_response.status_code == 200
    assert stop_response.json()["state"] == "stopped"

    second_run = client.post(f"/strategies/{strategy_id}/run", headers=headers)
    assert second_run.status_code == 200
    assert second_run.json()["state"] == "running"
    assert second_run.json()["run_id"] != first_run.json()["run_id"]



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
