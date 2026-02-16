"""
Strategy API routes for immutable CRUD, clone, and run control endpoints.

Docs:
  - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Mapping
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict
from starlette.requests import Request

from trading.contexts.strategy.application.ports.current_user import CurrentUserProvider
from trading.contexts.strategy.application.use_cases import (
    CloneStrategyUseCase,
    CreateStrategyUseCase,
    DeleteStrategyUseCase,
    GetMyStrategyUseCase,
    ListMyStrategiesUseCase,
    RunStrategyUseCase,
    StopStrategyUseCase,
)
from trading.contexts.strategy.domain.entities import Strategy, StrategyRun

CurrentUserProviderDependency = Callable[[Request], CurrentUserProvider]


class StrategyInstrumentIdRequest(BaseModel):
    """
    API payload for Strategy instrument identity tuple.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/shared_kernel/primitives/instrument_id.py
      - apps/api/routes/strategies.py
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
    """

    model_config = ConfigDict(extra="forbid")

    market_id: int
    symbol: str


class StrategySpecRequest(BaseModel):
    """
    API payload for immutable StrategySpecV1 creation/clone template.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
      - apps/api/routes/strategies.py
      - src/trading/contexts/strategy/application/use_cases/create_strategy.py
    """

    model_config = ConfigDict(extra="forbid")

    instrument_id: StrategyInstrumentIdRequest
    instrument_key: str
    market_type: str
    timeframe: str
    indicators: list[dict[str, Any]]
    signal_template: str | None = None


class StrategyCloneOverridesRequest(BaseModel):
    """
    Clone overrides payload with explicit whitelist (`instrument_id`, `timeframe`).

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/use_cases/clone_strategy.py
      - apps/api/routes/strategies.py
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    """

    model_config = ConfigDict(extra="forbid")

    instrument_id: StrategyInstrumentIdRequest | None = None
    timeframe: str | None = None


class CreateStrategyRequest(StrategySpecRequest):
    """
    Request payload for `POST /strategies` endpoint.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/routes/strategies.py
      - src/trading/contexts/strategy/application/use_cases/create_strategy.py
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    """


class CloneStrategyRequest(BaseModel):
    """
    Request payload for `POST /strategies/clone` endpoint.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/routes/strategies.py
      - src/trading/contexts/strategy/application/use_cases/clone_strategy.py
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    """

    model_config = ConfigDict(extra="forbid")

    source_strategy_id: UUID | None = None
    template: StrategySpecRequest | None = None
    overrides: StrategyCloneOverridesRequest | None = None


class StrategyInstrumentIdResponse(BaseModel):
    """
    API response payload for Strategy instrument identity tuple.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/shared_kernel/primitives/instrument_id.py
      - apps/api/routes/strategies.py
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
    """

    market_id: int
    symbol: str


class StrategySpecResponse(BaseModel):
    """
    API response payload for immutable strategy specification snapshot.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/routes/strategies.py
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
      - src/trading/contexts/strategy/application/use_cases
    """

    instrument_id: StrategyInstrumentIdResponse
    instrument_key: str
    market_type: str
    timeframe: str
    indicators: list[dict[str, Any]]
    signal_template: str
    schema_version: int
    spec_kind: str


class StrategyResponse(BaseModel):
    """
    API response payload for immutable strategy snapshot.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/routes/strategies.py
      - src/trading/contexts/strategy/domain/entities/strategy.py
      - src/trading/contexts/strategy/application/use_cases
    """

    strategy_id: UUID
    user_id: UUID
    name: str
    created_at: datetime
    is_deleted: bool
    spec: StrategySpecResponse


class StrategyRunResponse(BaseModel):
    """
    API response payload for strategy run control endpoints.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/routes/strategies.py
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
      - src/trading/contexts/strategy/application/use_cases/run_strategy.py
    """

    run_id: UUID
    strategy_id: UUID
    state: str
    started_at: datetime
    stopped_at: datetime | None
    checkpoint_ts_open: datetime | None
    last_error: str | None
    updated_at: datetime
    metadata: dict[str, Any]



def build_strategies_router(
    *,
    create_use_case: CreateStrategyUseCase,
    clone_use_case: CloneStrategyUseCase,
    list_use_case: ListMyStrategiesUseCase,
    get_use_case: GetMyStrategyUseCase,
    run_use_case: RunStrategyUseCase,
    stop_use_case: StopStrategyUseCase,
    delete_use_case: DeleteStrategyUseCase,
    current_user_provider_dependency: CurrentUserProviderDependency,
) -> APIRouter:
    """
    Build Strategy API router with immutable CRUD, clone, and run-control endpoints.

    Args:
        create_use_case: Immutable strategy creation use-case.
        clone_use_case: Strategy clone use-case.
        list_use_case: Owner-scoped strategy list use-case.
        get_use_case: Owner-scoped strategy get use-case.
        run_use_case: Strategy run start use-case.
        stop_use_case: Strategy run stop use-case.
        delete_use_case: Strategy soft-delete use-case.
        current_user_provider_dependency: Dependency resolving CurrentUserProvider port.
    Returns:
        APIRouter: Configured router exposing Strategy v1 endpoints.
    Assumptions:
        Business logic is implemented in use-cases; route layer only maps DTOs and dependencies.
    Raises:
        ValueError: If any required dependency is missing.
    Side Effects:
        None.
    """
    if create_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategies_router requires create_use_case")
    if clone_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategies_router requires clone_use_case")
    if list_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategies_router requires list_use_case")
    if get_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategies_router requires get_use_case")
    if run_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategies_router requires run_use_case")
    if stop_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategies_router requires stop_use_case")
    if delete_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategies_router requires delete_use_case")

    router = APIRouter(tags=["strategy"])

    @router.post("/strategies", response_model=StrategyResponse, status_code=201)
    def post_strategies(
        request: CreateStrategyRequest,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> StrategyResponse:
        """
        Create immutable strategy snapshot owned by current authenticated user.

        Args:
            request: Strategy spec request payload.
            current_user_provider: Dependency resolving current user context.
        Returns:
            StrategyResponse: Persisted strategy snapshot payload.
        Assumptions:
            All strategy updates are represented through clone endpoint, not mutable updates.
        Raises:
            RoehubError: Propagated from use-case and mapped by global API error handlers.
        Side Effects:
            Persists strategy snapshot and append-only event.
        """
        current_user = current_user_provider.require_current_user()
        strategy = create_use_case.execute(
            spec_payload=_spec_request_to_payload(request),
            current_user=current_user,
        )
        return _to_strategy_response(strategy=strategy)

    @router.post("/strategies/clone", response_model=StrategyResponse, status_code=201)
    def post_strategies_clone(
        request: CloneStrategyRequest,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> StrategyResponse:
        """
        Clone strategy from template/existing source with explicit whitelisted overrides.

        Args:
            request: Clone strategy request payload.
            current_user_provider: Dependency resolving current user context.
        Returns:
            StrategyResponse: Persisted cloned strategy snapshot payload.
        Assumptions:
            Clone source xor contract is validated in use-case layer.
        Raises:
            RoehubError: Propagated from use-case and mapped by global API error handlers.
        Side Effects:
            Persists cloned strategy snapshot and append-only event.
        """
        current_user = current_user_provider.require_current_user()
        cloned_strategy = clone_use_case.execute(
            current_user=current_user,
            source_strategy_id=request.source_strategy_id,
            template_spec_payload=(
                _spec_request_to_payload(request.template)
                if request.template is not None
                else None
            ),
            overrides=_overrides_request_to_payload(request.overrides),
        )
        return _to_strategy_response(strategy=cloned_strategy)

    @router.get("/strategies", response_model=list[StrategyResponse])
    def get_strategies(
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> list[StrategyResponse]:
        """
        List owner strategies in deterministic ordering.

        Args:
            current_user_provider: Dependency resolving current user context.
        Returns:
            list[StrategyResponse]: Deterministically ordered owned strategy list payload.
        Assumptions:
            Soft-deleted strategies are excluded by default.
        Raises:
            RoehubError: Propagated from use-case and mapped by global API error handlers.
        Side Effects:
            Reads strategy snapshots from storage.
        """
        current_user = current_user_provider.require_current_user()
        strategies = list_use_case.execute(current_user=current_user)
        return [_to_strategy_response(strategy=item) for item in strategies]

    @router.get("/strategies/{strategy_id}", response_model=StrategyResponse)
    def get_strategy_by_id(
        strategy_id: UUID,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> StrategyResponse:
        """
        Fetch one owned strategy snapshot by identifier.

        Args:
            strategy_id: Target strategy identifier.
            current_user_provider: Dependency resolving current user context.
        Returns:
            StrategyResponse: Owned strategy snapshot payload.
        Assumptions:
            Ownership check is explicit business rule in use-case layer.
        Raises:
            RoehubError: Propagated from use-case and mapped by global API error handlers.
        Side Effects:
            Reads one strategy snapshot from storage.
        """
        current_user = current_user_provider.require_current_user()
        strategy = get_use_case.execute(strategy_id=strategy_id, current_user=current_user)
        return _to_strategy_response(strategy=strategy)

    @router.post("/strategies/{strategy_id}/run", response_model=StrategyRunResponse)
    def post_strategy_run(
        strategy_id: UUID,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> StrategyRunResponse:
        """
        Start strategy run with deterministic warmup metadata and state transitions.

        Args:
            strategy_id: Target strategy identifier.
            current_user_provider: Dependency resolving current user context.
        Returns:
            StrategyRunResponse: Running strategy run snapshot payload.
        Assumptions:
            One-active-run invariant is enforced in use-case and repository layers.
        Raises:
            RoehubError: Propagated from use-case and mapped by global API error handlers.
        Side Effects:
            Persists run lifecycle snapshots and append-only events.
        """
        current_user = current_user_provider.require_current_user()
        run = run_use_case.execute(strategy_id=strategy_id, current_user=current_user)
        return _to_strategy_run_response(run=run)

    @router.post("/strategies/{strategy_id}/stop", response_model=StrategyRunResponse)
    def post_strategy_stop(
        strategy_id: UUID,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> StrategyRunResponse:
        """
        Stop active strategy run with deterministic stopping->stopped transitions.

        Args:
            strategy_id: Target strategy identifier.
            current_user_provider: Dependency resolving current user context.
        Returns:
            StrategyRunResponse: Stopped strategy run snapshot payload.
        Assumptions:
            Stop is conflict when no active run exists.
        Raises:
            RoehubError: Propagated from use-case and mapped by global API error handlers.
        Side Effects:
            Persists run lifecycle snapshots and append-only events.
        """
        current_user = current_user_provider.require_current_user()
        stopped_run = stop_use_case.execute(strategy_id=strategy_id, current_user=current_user)
        return _to_strategy_run_response(run=stopped_run)

    @router.delete("/strategies/{strategy_id}", status_code=204, response_model=None)
    def delete_strategy(
        strategy_id: UUID,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> Response:
        """
        Soft-delete one owned strategy snapshot.

        Args:
            strategy_id: Target strategy identifier.
            current_user_provider: Dependency resolving current user context.
        Returns:
            Response: Empty HTTP 204 response.
        Assumptions:
            Delete operation updates `is_deleted` flag and preserves immutable strategy spec fields.
        Raises:
            RoehubError: Propagated from use-case and mapped by global API error handlers.
        Side Effects:
            Updates strategy storage row and appends strategy-deleted event.
        """
        current_user = current_user_provider.require_current_user()
        delete_use_case.execute(strategy_id=strategy_id, current_user=current_user)
        return Response(status_code=204)

    return router



def _spec_request_to_payload(request: StrategySpecRequest | None) -> Mapping[str, Any]:
    """
    Convert StrategySpec request DTO to deterministic mapping payload consumed by use-cases.

    Args:
        request: Strategy spec request model.
    Returns:
        Mapping[str, Any]: Deterministic spec payload mapping.
    Assumptions:
        Pydantic model validation already ensured shape correctness.
    Raises:
        ValueError: If request is unexpectedly missing.
    Side Effects:
        None.
    """
    if request is None:
        raise ValueError("Strategy spec request is required")

    payload: dict[str, Any] = {
        "instrument_id": {
            "market_id": request.instrument_id.market_id,
            "symbol": request.instrument_id.symbol,
        },
        "instrument_key": request.instrument_key,
        "market_type": request.market_type,
        "timeframe": request.timeframe,
        "indicators": request.indicators,
    }
    if request.signal_template is not None:
        payload["signal_template"] = request.signal_template
    return payload



def _overrides_request_to_payload(
    request: StrategyCloneOverridesRequest | None,
) -> Mapping[str, Any] | None:
    """
    Convert clone overrides request DTO into deterministic mapping payload.

    Args:
        request: Clone overrides request model.
    Returns:
        Mapping[str, Any] | None: Overrides mapping or `None` when absent.
    Assumptions:
        Only explicit whitelist fields are represented by request DTO.
    Raises:
        None.
    Side Effects:
        None.
    """
    if request is None:
        return None

    payload: dict[str, Any] = {}
    if request.instrument_id is not None:
        payload["instrument_id"] = {
            "market_id": request.instrument_id.market_id,
            "symbol": request.instrument_id.symbol,
        }
    if request.timeframe is not None:
        payload["timeframe"] = request.timeframe
    return payload



def _to_strategy_response(*, strategy: Strategy) -> StrategyResponse:
    """
    Convert Strategy domain entity into strict API response DTO.

    Args:
        strategy: Strategy domain snapshot.
    Returns:
        StrategyResponse: API response DTO.
    Assumptions:
        Strategy domain entity already satisfies immutable invariants.
    Raises:
        ValueError: If response mapping encounters unsupported domain value types.
    Side Effects:
        None.
    """
    return StrategyResponse(
        strategy_id=strategy.strategy_id,
        user_id=strategy.user_id.value,
        name=strategy.name,
        created_at=strategy.created_at,
        is_deleted=strategy.is_deleted,
        spec=StrategySpecResponse(
            instrument_id=StrategyInstrumentIdResponse(
                market_id=strategy.spec.instrument_id.market_id.value,
                symbol=str(strategy.spec.instrument_id.symbol),
            ),
            instrument_key=strategy.spec.instrument_key,
            market_type=strategy.spec.market_type,
            timeframe=strategy.spec.timeframe.code,
            indicators=[dict(indicator) for indicator in strategy.spec.indicators],
            signal_template=strategy.spec.signal_template,
            schema_version=strategy.spec.schema_version,
            spec_kind=strategy.spec.spec_kind,
        ),
    )



def _to_strategy_run_response(*, run: StrategyRun) -> StrategyRunResponse:
    """
    Convert StrategyRun domain entity into strict API response DTO.

    Args:
        run: Strategy run domain snapshot.
    Returns:
        StrategyRunResponse: API response DTO.
    Assumptions:
        Run metadata payload is JSON-compatible and deterministic.
    Raises:
        ValueError: If response mapping encounters unsupported metadata value types.
    Side Effects:
        None.
    """
    return StrategyRunResponse(
        run_id=run.run_id,
        strategy_id=run.strategy_id,
        state=run.state,
        started_at=run.started_at,
        stopped_at=run.stopped_at,
        checkpoint_ts_open=run.checkpoint_ts_open,
        last_error=run.last_error,
        updated_at=run.updated_at,
        metadata=dict(run.metadata_json),
    )
