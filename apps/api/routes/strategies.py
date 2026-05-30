"""
Strategy API routes for immutable CRUD, clone, and run control endpoints.

Docs:
  - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Literal, Mapping
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request

from apps.api.monitoring import record_live_strategy_profile_readiness
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.application.ports.current_user import CurrentUserProvider
from trading.contexts.strategy.application.use_cases import (
    CloneStrategyUseCase,
    CreateStrategyUseCase,
    DeleteStrategyUseCase,
    GetMyStrategyUseCase,
    ListMyStrategiesUseCase,
    LiveStrategyProfileConfig,
    LiveStrategyProfileService,
    RestartStrategyUseCase,
    RunStrategyUseCase,
    StopStrategyUseCase,
)
from trading.contexts.strategy.domain.entities import LiveStrategyProfile, Strategy, StrategyRun

CurrentUserProviderDependency = Callable[[Request], CurrentUserProvider]
CurrentUserPrincipalDependency = Callable[[Request], CurrentUserPrincipal]
_RECENT_AUTH_WINDOW = timedelta(minutes=10)


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


class LiveStrategyProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["monitor_only", "paper", "live"] = "monitor_only"
    exchange_connection_id: UUID | None = None
    sizing_method: Literal["fixed_quote", "fixed_equity_pct"] = "fixed_quote"
    sizing_value: Decimal = Field(default=Decimal("0"), ge=0)
    max_position_notional: Decimal | None = Field(default=None, ge=0)
    max_orders_per_run: int = Field(default=0, ge=0)
    max_notional_per_run: Decimal = Field(default=Decimal("0"), ge=0)


class LiveStrategyProfileResponse(BaseModel):
    profile_id: UUID
    owner_user_id: UUID
    strategy_id: UUID
    mode: Literal["monitor_only", "paper", "live"]
    exchange_connection_id: UUID | None
    sizing_method: Literal["fixed_quote", "fixed_equity_pct"]
    sizing_value: Decimal
    max_position_notional: Decimal | None
    max_orders_per_run: int
    max_notional_per_run: Decimal
    readiness_status: Literal["ready", "blocked"]
    readiness_reason: str
    created_at: datetime
    updated_at: datetime



def build_strategies_router(
    *,
    create_use_case: CreateStrategyUseCase,
    clone_use_case: CloneStrategyUseCase,
    list_use_case: ListMyStrategiesUseCase,
    get_use_case: GetMyStrategyUseCase,
    run_use_case: RunStrategyUseCase,
    stop_use_case: StopStrategyUseCase,
    restart_use_case: RestartStrategyUseCase,
    delete_use_case: DeleteStrategyUseCase,
    current_user_provider_dependency: CurrentUserProviderDependency,
    live_profile_service: LiveStrategyProfileService | None = None,
    current_user_principal_dependency: CurrentUserPrincipalDependency | None = None,
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
    if restart_use_case is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategies_router requires restart_use_case")
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

    @router.post(
        "/strategies/{strategy_id}/live-profile",
        response_model=LiveStrategyProfileResponse,
        status_code=201,
    )
    def post_strategy_live_profile(
        strategy_id: UUID,
        request: Request,
        payload: LiveStrategyProfileRequest | None = None,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> LiveStrategyProfileResponse:
        service = _require_live_profile_service(service=live_profile_service)
        current_user = current_user_provider.require_current_user()
        if payload is None:
            profile = service.get_or_create_default(
                strategy_id=strategy_id,
                current_user=current_user,
            )
        else:
            profile = service.update_profile(
                strategy_id=strategy_id,
                current_user=current_user,
                config=_profile_request_to_config(request=payload),
                recent_auth_confirmed=_recent_auth_confirmed(
                    request=request,
                    principal_dependency=current_user_principal_dependency,
                ),
            )
        record_live_strategy_profile_readiness(
            status=profile.readiness_status,
            reason=profile.readiness_reason,
        )
        return _to_live_strategy_profile_response(profile=profile)

    @router.get(
        "/strategies/{strategy_id}/live-profile",
        response_model=LiveStrategyProfileResponse,
    )
    def get_strategy_live_profile(
        strategy_id: UUID,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> LiveStrategyProfileResponse:
        service = _require_live_profile_service(service=live_profile_service)
        current_user = current_user_provider.require_current_user()
        profile = service.get_or_create_default(
            strategy_id=strategy_id,
            current_user=current_user,
        )
        return _to_live_strategy_profile_response(profile=profile)

    @router.put(
        "/strategies/{strategy_id}/live-profile",
        response_model=LiveStrategyProfileResponse,
    )
    def put_strategy_live_profile(
        strategy_id: UUID,
        request: Request,
        payload: LiveStrategyProfileRequest,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> LiveStrategyProfileResponse:
        service = _require_live_profile_service(service=live_profile_service)
        current_user = current_user_provider.require_current_user()
        profile = service.update_profile(
            strategy_id=strategy_id,
            current_user=current_user,
            config=_profile_request_to_config(request=payload),
            recent_auth_confirmed=_recent_auth_confirmed(
                request=request,
                principal_dependency=current_user_principal_dependency,
            ),
        )
        record_live_strategy_profile_readiness(
            status=profile.readiness_status,
            reason=profile.readiness_reason,
        )
        return _to_live_strategy_profile_response(profile=profile)

    @router.get(
        "/strategies/{strategy_id}/live-profile/readiness",
        response_model=LiveStrategyProfileResponse,
    )
    def get_strategy_live_profile_readiness(
        strategy_id: UUID,
        request: Request,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> LiveStrategyProfileResponse:
        service = _require_live_profile_service(service=live_profile_service)
        current_user = current_user_provider.require_current_user()
        profile = service.refresh_readiness(
            strategy_id=strategy_id,
            current_user=current_user,
            recent_auth_confirmed=_recent_auth_confirmed(
                request=request,
                principal_dependency=current_user_principal_dependency,
            ),
        )
        record_live_strategy_profile_readiness(
            status=profile.readiness_status,
            reason=profile.readiness_reason,
        )
        return _to_live_strategy_profile_response(profile=profile)

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

    @router.post("/strategies/{strategy_id}/restart", response_model=StrategyRunResponse)
    def post_strategy_restart(
        strategy_id: UUID,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> StrategyRunResponse:
        """
        Queue durable restart for an active strategy run.

        The API records restart intent and transitions the current run to `stopping`.
        The live-runner drains it to `stopped` and creates the successor run.
        """
        current_user = current_user_provider.require_current_user()
        restarting_run = restart_use_case.execute(
            strategy_id=strategy_id,
            current_user=current_user,
        )
        return _to_strategy_run_response(run=restarting_run)

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


def _profile_request_to_config(
    *, request: LiveStrategyProfileRequest
) -> LiveStrategyProfileConfig:
    return LiveStrategyProfileConfig(
        mode=request.mode,
        exchange_connection_id=request.exchange_connection_id,
        sizing_method=request.sizing_method,
        sizing_value=request.sizing_value,
        max_position_notional=request.max_position_notional,
        max_orders_per_run=request.max_orders_per_run,
        max_notional_per_run=request.max_notional_per_run,
    )


def _require_live_profile_service(
    *, service: LiveStrategyProfileService | None
) -> LiveStrategyProfileService:
    if service is None:
        from trading.platform.errors import RoehubError

        raise RoehubError(
            code="live_strategy_profile_unavailable",
            message="Live strategy profile service is not configured",
            details={},
        )
    return service


def _recent_auth_confirmed(
    *,
    request: Request,
    principal_dependency: CurrentUserPrincipalDependency | None,
) -> bool:
    if principal_dependency is None:
        return False
    principal = principal_dependency(request)
    if principal.session_created_at is None:
        return False
    now = datetime.now(tz=principal.session_created_at.tzinfo)
    return principal.session_created_at + _RECENT_AUTH_WINDOW >= now



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


def _to_live_strategy_profile_response(
    *, profile: LiveStrategyProfile
) -> LiveStrategyProfileResponse:
    return LiveStrategyProfileResponse(
        profile_id=profile.profile_id,
        owner_user_id=profile.owner_user_id.value,
        strategy_id=profile.strategy_id,
        mode=profile.mode,
        exchange_connection_id=profile.exchange_connection_id,
        sizing_method=profile.sizing_method,
        sizing_value=profile.sizing_value,
        max_position_notional=profile.max_position_notional,
        max_orders_per_run=profile.max_orders_per_run,
        max_notional_per_run=profile.max_notional_per_run,
        readiness_status=profile.readiness_status,
        readiness_reason=profile.readiness_reason,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
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
