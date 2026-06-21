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

from fastapi import APIRouter, Depends, Header, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request

from apps.api.monitoring import (
    record_live_strategy_profile_readiness,
    record_market_data_readiness,
    record_strategy_variant_compatibility,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.live_execution.application import (
    CapitalReservationPaperAccountingService,
    CreateExecutionIntentCommand,
    ExecutionDispatchService,
    ExecutionIngressService,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.domain import (
    PAPER_VIRTUAL_EXCHANGE_CONNECTION_ID,
    CapitalReservationBlockedError,
    ExecutionRiskContext,
    ExecutionSourceValidationError,
)
from trading.contexts.strategy.application.ports.current_user import (
    CurrentUser,
    CurrentUserProvider,
)
from trading.contexts.strategy.application.ports.exchange_connection_readiness import (
    ExchangeConnectionReadinessContext,
)
from trading.contexts.strategy.application.ports.repositories import (
    LiveStrategyProfileRepository,
    StrategyRunRepository,
)
from trading.contexts.strategy.application.use_cases import (
    SCENARIO_MATRIX_LAUNCH_RISK_MODES_V1,
    SCENARIO_MATRIX_MIN_NOTIONAL_USD_V1,
    SCENARIO_MATRIX_MODES_V1,
    SCENARIO_MATRIX_SYMBOL_SCOPE_V1,
    CloneStrategyUseCase,
    CreateStrategyFromBacktestVariantUseCase,
    CreateStrategyUseCase,
    DeleteStrategyUseCase,
    GetMyStrategyUseCase,
    ListMyStrategiesUseCase,
    LiveStrategyProfileConfig,
    LiveStrategyProfileService,
    RestartStrategyUseCase,
    RunStrategyUseCase,
    StopStrategyUseCase,
    StrategyCompatibilityReadinessService,
)
from trading.contexts.strategy.domain.entities import LiveStrategyProfile, Strategy, StrategyRun
from trading.platform.errors import RoehubError

CurrentUserProviderDependency = Callable[[Request], CurrentUserProvider]
CurrentUserPrincipalDependency = Callable[[Request], CurrentUserPrincipal]
_RECENT_AUTH_WINDOW = timedelta(minutes=10)
_DEFAULT_LAUNCH_CAPITAL_USD = Decimal("50")
_MIN_BTCUSDT_NOTIONAL_USD = SCENARIO_MATRIX_MIN_NOTIONAL_USD_V1
_ALLOWED_LAUNCH_MODES = frozenset(SCENARIO_MATRIX_MODES_V1)
_ALLOWED_LAUNCH_MARKET_TYPES = frozenset({"spot", "futures"})
_ALLOWED_LAUNCH_ENTRY_SIZING = frozenset({"fixed_quote", "fixed_equity_pct"})
_ALLOWED_LAUNCH_RISK_MODES = frozenset(SCENARIO_MATRIX_LAUNCH_RISK_MODES_V1)
_ALLOWED_LAUNCH_DIRECTIONS = frozenset({"long", "short", "long_short_reversal"})


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

    mode: Literal["monitor_only", "paper", "live", "testnet"] = "monitor_only"
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
    mode: Literal["monitor_only", "paper", "live", "testnet"]
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


class StrategyCompatibilityReadinessResponse(BaseModel):
    compatibility_check_id: UUID
    market_data_requirement_id: UUID
    strategy_id: UUID | None
    source_job_id: UUID | None
    source_variant_key: str | None
    strategy_spec_hash: str
    instrument_key: str
    market_type: str
    timeframe: str
    compatibility_state: Literal["launchable", "not_launchable", "degraded"]
    compatibility_reason_codes: list[str]
    market_data_state: Literal["ready", "missing", "stale", "pending"]
    market_data_reason_codes: list[str]
    market_data_stream_name: str
    market_data_stream_length: int | None
    market_data_last_message_id: str | None
    market_data_last_observed_at: datetime | None
    market_data_age_seconds: int | None
    launch_blocked: bool
    launch_blocked_reason: str
    checked_at: datetime


class BacktestVariantLaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    variant_key: str
    mode: str = "paper"
    exchange_connection_id: UUID | None = None
    market_type: str = "spot"
    symbol: str = "BTCUSDT"
    capital_allocation_usd: Decimal = Field(default=_DEFAULT_LAUNCH_CAPITAL_USD, ge=0)
    entry_sizing: str = "fixed_quote"
    risk_mode: str = "single_position_cap"
    direction: str = "long"


class BacktestVariantLaunchProvenanceResponse(BaseModel):
    source_job_id: UUID
    source_variant_key: str
    source_variant_hash: str
    source_indicator_variant_hash: str | None
    strategy_spec_hash: str
    launch_request_hash: str


class BacktestVariantLaunchResponse(BaseModel):
    status: Literal["started"]
    duplicate_strategy: bool
    duplicate_reason: str | None
    strategy: StrategyResponse
    profile: LiveStrategyProfileResponse
    run: StrategyRunResponse
    provenance: BacktestVariantLaunchProvenanceResponse
    launch_config: dict[str, Any]


class ManualStrategyExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: str | None = Field(default=None, min_length=1, max_length=128)
    quote_notional: Decimal | None = Field(default=None, gt=0)
    reference_price: Decimal | None = Field(default=None, gt=0)


class ManualStrategyExecutionResponse(BaseModel):
    status: Literal["pending", "accepted", "rejected", "unknown"]
    action: Literal["entry", "exit"]
    duplicate: bool
    source_event_id: UUID
    intent_id: UUID
    risk_status: str
    risk_reason: str
    intent_status: str
    intent_status_reason: str
    dispatch_stream_name: str | None
    dispatch_redis_message_id: str | None
    paper_accounting_id: UUID | None = None
    paper_order_state: str | None = None
    outcome_reason: str


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
    compatibility_readiness_service: StrategyCompatibilityReadinessService | None = None,
    create_strategy_from_variant_use_case: CreateStrategyFromBacktestVariantUseCase | None = None,
    strategy_run_repository: StrategyRunRepository | None = None,
    live_profile_repository: LiveStrategyProfileRepository | None = None,
    execution_ingress_service: ExecutionIngressService | None = None,
    execution_dispatch_service: ExecutionDispatchService | None = None,
    paper_accounting_service: CapitalReservationPaperAccountingService | None = None,
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

    @router.post(
        "/strategies/launch-from-backtest-variant",
        response_model=BacktestVariantLaunchResponse,
        status_code=201,
    )
    def post_strategy_launch_from_backtest_variant(
        payload: BacktestVariantLaunchRequest,
        request: Request,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> BacktestVariantLaunchResponse:
        launch_config = _validated_backtest_variant_launch_config(payload=payload)
        create_from_variant = _require_create_strategy_from_variant_use_case(
            use_case=create_strategy_from_variant_use_case,
        )
        profile_service = _require_live_profile_service(service=live_profile_service)
        current_user = current_user_provider.require_current_user()
        create_result = create_from_variant.execute(
            current_user=current_user,
            job_id=payload.job_id,
            variant_key=payload.variant_key,
            idempotency_key=idempotency_key,
            launch_config=launch_config,
        )
        if create_result.duplicate:
            response.status_code = 200
        profile = profile_service.update_profile(
            strategy_id=create_result.strategy.strategy_id,
            current_user=current_user,
            config=LiveStrategyProfileConfig(
                mode=launch_config["mode"],  # type: ignore[arg-type]
                exchange_connection_id=payload.exchange_connection_id,
                sizing_method=launch_config["entry_sizing"],  # type: ignore[arg-type]
                sizing_value=payload.capital_allocation_usd,
                max_position_notional=payload.capital_allocation_usd,
                max_orders_per_run=1,
                max_notional_per_run=payload.capital_allocation_usd,
                readiness_context=ExchangeConnectionReadinessContext(
                    mode=str(launch_config["mode"]),
                    market_type=str(launch_config["market_type"]),
                    symbol=str(launch_config["symbol"]),
                    direction=str(launch_config["direction"]),
                    notional=payload.capital_allocation_usd,
                ),
            ),
            recent_auth_confirmed=_recent_auth_confirmed(
                request=request,
                principal_dependency=current_user_principal_dependency,
            ),
        )
        if profile.readiness_status != "ready":
            raise RoehubError(
                code="strategy_launch.readiness_blocked",
                message="Strategy launch is blocked by readiness",
                details={
                    "reason": profile.readiness_reason,
                    "profile_id": str(profile.profile_id),
                    "strategy_id": str(create_result.strategy.strategy_id),
                },
            )
        run = run_use_case.execute(
            strategy_id=create_result.strategy.strategy_id,
            current_user=current_user,
            metadata_json={
                "schema": "strategy_backtest_variant_launch_run_v1",
                "launch_config": launch_config,
                "provenance": {
                    "source_job_id": str(create_result.provenance.source_job_id),
                    "source_variant_key": create_result.provenance.source_variant_key,
                    "strategy_spec_hash": create_result.provenance.strategy_spec_hash,
                    "launch_request_hash": create_result.provenance.launch_request_hash,
                },
            },
        )
        return BacktestVariantLaunchResponse(
            status="started",
            duplicate_strategy=create_result.duplicate,
            duplicate_reason=create_result.duplicate_reason,
            strategy=_to_strategy_response(strategy=create_result.strategy),
            profile=_to_live_strategy_profile_response(profile=profile),
            run=_to_strategy_run_response(run=run),
            provenance=BacktestVariantLaunchProvenanceResponse(
                source_job_id=create_result.provenance.source_job_id,
                source_variant_key=create_result.provenance.source_variant_key,
                source_variant_hash=create_result.provenance.source_variant_hash,
                source_indicator_variant_hash=(
                    create_result.provenance.source_indicator_variant_hash
                ),
                strategy_spec_hash=create_result.provenance.strategy_spec_hash,
                launch_request_hash=create_result.provenance.launch_request_hash,
            ),
            launch_config=launch_config,
        )

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

    @router.get(
        "/strategies/{strategy_id}/compatibility-readiness",
        response_model=StrategyCompatibilityReadinessResponse,
    )
    def get_strategy_compatibility_readiness(
        strategy_id: UUID,
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> StrategyCompatibilityReadinessResponse:
        service = _require_compatibility_readiness_service(
            service=compatibility_readiness_service,
        )
        current_user = current_user_provider.require_current_user()
        report = service.check_strategy(strategy_id=strategy_id, current_user=current_user)
        _record_compatibility_readiness(report=report)
        return _to_compatibility_readiness_response(report=report)

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

    @router.post(
        "/strategies/{strategy_id}/manual-entry",
        response_model=ManualStrategyExecutionResponse,
    )
    def post_strategy_manual_entry(
        strategy_id: UUID,
        request: Request,
        response: Response,
        payload: ManualStrategyExecutionRequest | None = None,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> ManualStrategyExecutionResponse:
        current_user = current_user_provider.require_current_user()
        result = _execute_manual_strategy_action(
            action="entry",
            strategy_id=strategy_id,
            request=request,
            payload=payload or ManualStrategyExecutionRequest(),
            idempotency_key=idempotency_key,
            current_user=current_user,
            get_use_case=get_use_case,
            run_repository=_require_strategy_run_repository(
                repository=strategy_run_repository,
            ),
            profile_repository=_require_live_profile_repository(
                repository=live_profile_repository,
            ),
            compatibility_readiness_service=compatibility_readiness_service,
            ingress_service=_require_execution_ingress_service(
                service=execution_ingress_service,
            ),
            dispatch_service=execution_dispatch_service,
            paper_accounting_service=paper_accounting_service,
            principal_dependency=current_user_principal_dependency,
        )
        if result.duplicate:
            response.status_code = 200
        return result

    @router.post(
        "/strategies/{strategy_id}/manual-exit",
        response_model=ManualStrategyExecutionResponse,
    )
    def post_strategy_manual_exit(
        strategy_id: UUID,
        request: Request,
        response: Response,
        payload: ManualStrategyExecutionRequest | None = None,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        current_user_provider: CurrentUserProvider = Depends(current_user_provider_dependency),
    ) -> ManualStrategyExecutionResponse:
        current_user = current_user_provider.require_current_user()
        result = _execute_manual_strategy_action(
            action="exit",
            strategy_id=strategy_id,
            request=request,
            payload=payload or ManualStrategyExecutionRequest(),
            idempotency_key=idempotency_key,
            current_user=current_user,
            get_use_case=get_use_case,
            run_repository=_require_strategy_run_repository(
                repository=strategy_run_repository,
            ),
            profile_repository=_require_live_profile_repository(
                repository=live_profile_repository,
            ),
            compatibility_readiness_service=compatibility_readiness_service,
            ingress_service=_require_execution_ingress_service(
                service=execution_ingress_service,
            ),
            dispatch_service=execution_dispatch_service,
            paper_accounting_service=paper_accounting_service,
            principal_dependency=current_user_principal_dependency,
        )
        if result.duplicate:
            response.status_code = 200
        return result

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


def _require_compatibility_readiness_service(
    *, service: StrategyCompatibilityReadinessService | None
) -> StrategyCompatibilityReadinessService:
    if service is None:
        raise RoehubError(
            code="strategy_compatibility.unavailable",
            message="Strategy compatibility readiness service is not configured",
            details={"reason": "strategy_compatibility_unavailable"},
        )
    return service


def _require_create_strategy_from_variant_use_case(
    *,
    use_case: CreateStrategyFromBacktestVariantUseCase | None,
) -> CreateStrategyFromBacktestVariantUseCase:
    if use_case is None:
        raise RoehubError(
            code="strategy_launch.unavailable",
            message="Backtest variant launch service is not configured",
            details={"reason": "strategy_launch_unavailable"},
        )
    return use_case


def _require_strategy_run_repository(
    *, repository: StrategyRunRepository | None
) -> StrategyRunRepository:
    if repository is None:
        raise RoehubError(
            code="strategy_manual_execution.unavailable",
            message="Manual execution is not configured",
            details={"reason": "strategy_run_repository_unavailable"},
        )
    return repository


def _require_live_profile_repository(
    *, repository: LiveStrategyProfileRepository | None
) -> LiveStrategyProfileRepository:
    if repository is None:
        raise RoehubError(
            code="strategy_manual_execution.unavailable",
            message="Manual execution is not configured",
            details={"reason": "live_profile_repository_unavailable"},
        )
    return repository


def _require_execution_ingress_service(
    *, service: ExecutionIngressService | None
) -> ExecutionIngressService:
    if service is None:
        raise RoehubError(
            code="strategy_manual_execution.unavailable",
            message="Manual execution is not configured",
            details={"reason": "execution_ingress_unavailable"},
        )
    return service


def _execute_manual_strategy_action(
    *,
    action: Literal["entry", "exit"],
    strategy_id: UUID,
    request: Request,
    payload: ManualStrategyExecutionRequest,
    idempotency_key: str | None,
    current_user: CurrentUser,
    get_use_case: GetMyStrategyUseCase,
    run_repository: StrategyRunRepository,
    profile_repository: LiveStrategyProfileRepository,
    compatibility_readiness_service: StrategyCompatibilityReadinessService | None,
    ingress_service: ExecutionIngressService,
    dispatch_service: ExecutionDispatchService | None,
    paper_accounting_service: CapitalReservationPaperAccountingService | None,
    principal_dependency: CurrentUserPrincipalDependency | None,
) -> ManualStrategyExecutionResponse:
    strategy = get_use_case.execute(strategy_id=strategy_id, current_user=current_user)
    active_run = run_repository.find_active_for_strategy(
        user_id=current_user.user_id,
        strategy_id=strategy.strategy_id,
    )
    if active_run is None:
        raise RoehubError(
            code="strategy_manual_execution.blocked",
            message="Manual execution is blocked",
            details={"reason": "strategy_run_inactive"},
        )
    profile = profile_repository.get_for_strategy(
        owner_user_id=current_user.user_id,
        strategy_id=strategy.strategy_id,
    )
    if profile is None:
        raise RoehubError(
            code="strategy_manual_execution.blocked",
            message="Manual execution is blocked",
            details={"reason": "live_profile_missing"},
        )
    request_key = (idempotency_key or payload.client_request_id or "").strip()
    if not request_key:
        raise RoehubError(
            code="strategy_manual_execution.idempotency_required",
            message="Manual execution requires an idempotency key",
            details={"reason": "idempotency_key_required"},
        )
    direction = _manual_direction_from_run_metadata(metadata=active_run.metadata_json)
    if direction == "long_short_reversal":
        raise RoehubError(
            code="strategy_manual_execution.blocked",
            message="Manual execution is blocked",
            details={"reason": "manual_direction_ambiguous"},
        )
    side = _manual_side(action=action, direction=direction)
    quote_notional = _manual_quote_notional(payload=payload, profile=profile)
    reference_price = payload.reference_price or Decimal("1")
    exchange_connection_id = (
        profile.exchange_connection_id
        if profile.exchange_connection_id is not None
        else PAPER_VIRTUAL_EXCHANGE_CONNECTION_ID
    )
    source_key = _manual_source_idempotency_key(
        strategy_id=strategy.strategy_id,
        run_id=active_run.run_id,
        action=action,
        request_key=request_key,
    )
    try:
        source_result = ingress_service.record_source_event(
            command=RecordExecutionSourceEventCommand(
                owner_user_id=current_user.user_id,
                source_type="manual_request",
                source_event_ref=f"manual:{action}:{active_run.run_id}",
                source_ref_json={
                    "strategy_id": str(strategy.strategy_id),
                    "strategy_run_id": str(active_run.run_id),
                    "action": action,
                    "mode": profile.mode,
                    "instrument_key": strategy.spec.instrument_key,
                },
                strategy_signal_id=None,
                idempotency_key=source_key,
            )
        )
        intent_result = ingress_service.create_intent(
            command=CreateExecutionIntentCommand(
                owner_user_id=current_user.user_id,
                source_event_id=source_result.event.source_event_id,
                idempotency_key=f"{source_key}|intent",
                exchange_connection_id=exchange_connection_id,
                market_type=strategy.spec.market_type,
                instrument_key=strategy.spec.instrument_key,
                order_type="market",
                side=side,
                quantity=None,
                quote_notional=quote_notional,
                limit_price=None,
                advanced_order_flags={},
                risk_context=_manual_risk_context(
                    mode=profile.mode,
                    profile_ready=profile.readiness_status == "ready",
                    recent_auth=_recent_auth_confirmed(
                        request=request,
                        principal_dependency=principal_dependency,
                    ),
                    compatibility_readiness_service=compatibility_readiness_service,
                    strategy_id=strategy.strategy_id,
                    current_user=current_user,
                ),
            )
        )
    except ExecutionSourceValidationError as error:
        raise _manual_execution_request_error(reason=error.reason) from error

    intent = intent_result.intent
    if dispatch_service is not None:
        intent = dispatch_service.dispatch_intent(intent=intent).intent

    paper_accounting_id = None
    paper_order_state = None
    if (
        profile.mode == "paper"
        and intent.risk_status == "rejected"
        and intent.risk_reason == "paper_no_exchange_submit"
    ):
        if paper_accounting_service is None:
            raise RoehubError(
                code="strategy_manual_execution.unavailable",
                message="Manual execution is not configured",
                details={"reason": "paper_accounting_unavailable"},
            )
        try:
            accounting = paper_accounting_service.record_manual_paper_execution(
                owner_user_id=current_user.user_id,
                strategy_id=strategy.strategy_id,
                live_profile_id=profile.profile_id,
                strategy_run_id=active_run.run_id,
                source_event_id=source_result.event.source_event_id,
                instrument_key=strategy.spec.instrument_key,
                market_type=strategy.spec.market_type,
                side=side,
                quote_notional=quote_notional,
                reference_price=reference_price,
                now=intent.created_at,
            )
        except CapitalReservationBlockedError as error:
            raise RoehubError(
                code="strategy_manual_execution.blocked",
                message="Manual execution is blocked",
                details={"reason": error.reason},
            ) from error
        paper_accounting_id = accounting.accounting_id
        paper_order_state = "filled"

    status = _manual_response_status(
        intent_status=intent.status,
        risk_status=intent.risk_status,
        paper_order_state=paper_order_state,
    )
    return ManualStrategyExecutionResponse(
        status=status,
        action=action,
        duplicate=source_result.duplicate or intent_result.duplicate,
        source_event_id=source_result.event.source_event_id,
        intent_id=intent.intent_id,
        risk_status=intent.risk_status,
        risk_reason=intent.risk_reason,
        intent_status=intent.status,
        intent_status_reason=intent.status_reason,
        dispatch_stream_name=intent.dispatch_stream_name,
        dispatch_redis_message_id=intent.dispatch_redis_message_id,
        paper_accounting_id=paper_accounting_id,
        paper_order_state=paper_order_state,
        outcome_reason=paper_order_state or intent.risk_reason or intent.status_reason,
    )


def _manual_direction_from_run_metadata(*, metadata: Mapping[str, Any]) -> str:
    launch_config = metadata.get("launch_config")
    if isinstance(launch_config, Mapping):
        direction = str(launch_config.get("direction", "long")).strip().casefold()
        if direction in {"long", "short", "long_short_reversal"}:
            return direction
    return "long"


def _manual_side(*, action: Literal["entry", "exit"], direction: str) -> Literal["buy", "sell"]:
    if action == "entry":
        return "sell" if direction == "short" else "buy"
    return "buy" if direction == "short" else "sell"


def _manual_quote_notional(*, payload: ManualStrategyExecutionRequest, profile) -> Decimal:
    if payload.quote_notional is not None:
        return payload.quote_notional
    if profile.max_position_notional is not None and profile.max_position_notional > 0:
        return profile.max_position_notional
    if profile.sizing_value > 0:
        return profile.sizing_value
    return _DEFAULT_LAUNCH_CAPITAL_USD


def _manual_source_idempotency_key(
    *,
    strategy_id: UUID,
    run_id: UUID,
    action: str,
    request_key: str,
) -> str:
    return "|".join(("manual_request", str(strategy_id), str(run_id), action, request_key))


def _manual_risk_context(
    *,
    mode: str,
    profile_ready: bool,
    recent_auth: bool,
    compatibility_readiness_service: StrategyCompatibilityReadinessService | None,
    strategy_id: UUID,
    current_user: CurrentUser,
) -> ExecutionRiskContext:
    market_data_state = "ready"
    variant_compatible = True
    if compatibility_readiness_service is not None:
        report = compatibility_readiness_service.check_strategy(
            strategy_id=strategy_id,
            current_user=current_user,
        )
        market_data_state = report.market_data_state
        variant_compatible = report.compatibility_state == "launchable"
    if mode == "paper":
        return ExecutionRiskContext(
            exchange_connection_active=True,
            secret_custody_ready=True,
            source_authorized=True,
            strategy_variant_compatible=variant_compatible,
            market_data_state=market_data_state,
            strategy_binding_active=True,
            strategy_live_profile_ready=profile_ready,
            strategy_run_active=True,
            position_ownership_active=True,
            capital_reservation_active=True,
            capital_reservation_sufficient=True,
            paper_accounting_ready=True,
            paper_no_exchange_submit=True,
            manual_recent_auth=recent_auth,
            kill_switch_open=True,
            environment_policy_allows=True,
            max_order_size_ok=True,
            daily_limit_ok=True,
        )
    allows_testnet = mode == "testnet"
    return ExecutionRiskContext(
        exchange_connection_active=profile_ready and allows_testnet,
        secret_custody_ready=profile_ready and allows_testnet,
        source_authorized=True,
        strategy_variant_compatible=variant_compatible,
        market_data_state=market_data_state,
        strategy_binding_active=profile_ready and allows_testnet,
        strategy_live_profile_ready=profile_ready and allows_testnet,
        strategy_run_active=True,
        exchange_config_verified=profile_ready and allows_testnet,
        account_state_fresh=profile_ready and allows_testnet,
        position_ownership_active=profile_ready and allows_testnet,
        capital_reservation_active=profile_ready and allows_testnet,
        capital_reservation_sufficient=profile_ready and allows_testnet,
        manual_recent_auth=recent_auth,
        kill_switch_open=True,
        environment_policy_allows=allows_testnet,
        max_order_size_ok=True,
        daily_limit_ok=True,
    )


def _manual_response_status(
    *, intent_status: str, risk_status: str, paper_order_state: str | None
) -> Literal["pending", "accepted", "rejected", "unknown"]:
    if paper_order_state == "filled":
        return "accepted"
    if risk_status == "rejected" or intent_status == "rejected":
        return "rejected"
    if intent_status in {"accepted", "dispatching", "dispatched", "retry"}:
        return "pending"
    return "unknown"


def _manual_execution_request_error(*, reason: str) -> RoehubError:
    return RoehubError(
        code="strategy_manual_execution.invalid_request",
        message="Manual execution request is invalid",
        details={"reason": reason},
    )


def _validated_backtest_variant_launch_config(
    *, payload: BacktestVariantLaunchRequest
) -> dict[str, Any]:
    mode = payload.mode.strip().casefold()
    market_type = payload.market_type.strip().casefold()
    symbol = payload.symbol.strip().upper()
    entry_sizing = payload.entry_sizing.strip().casefold()
    risk_mode = payload.risk_mode.strip().casefold()
    direction = payload.direction.strip().casefold()
    if mode not in _ALLOWED_LAUNCH_MODES:
        raise _strategy_launch_validation_error(reason="invalid_mode", field="mode")
    if symbol != SCENARIO_MATRIX_SYMBOL_SCOPE_V1:
        raise _strategy_launch_validation_error(reason="unsupported_symbol", field="symbol")
    if market_type not in _ALLOWED_LAUNCH_MARKET_TYPES:
        raise _strategy_launch_validation_error(
            reason="invalid_market_type", field="market_type"
        )
    if entry_sizing not in _ALLOWED_LAUNCH_ENTRY_SIZING:
        raise _strategy_launch_validation_error(
            reason="invalid_entry_sizing", field="entry_sizing"
        )
    if risk_mode not in _ALLOWED_LAUNCH_RISK_MODES:
        raise _strategy_launch_validation_error(reason="invalid_risk_mode", field="risk_mode")
    if direction not in _ALLOWED_LAUNCH_DIRECTIONS:
        raise _strategy_launch_validation_error(reason="invalid_direction", field="direction")
    if mode == "testnet" and payload.exchange_connection_id is None:
        raise _strategy_launch_validation_error(
            reason="exchange_connection_required",
            field="exchange_connection_id",
        )
    if (
        mode == "testnet" and
        market_type == "spot" and
        direction in {"short", "long_short_reversal"}
    ):
        raise _strategy_launch_validation_error(
            reason="spot_short_not_supported",
            field="direction",
        )
    if payload.capital_allocation_usd < _MIN_BTCUSDT_NOTIONAL_USD:
        raise _strategy_launch_validation_error(
            reason="insufficient_allocation_min_notional",
            field="capital_allocation_usd",
        )
    return {
        "schema": "strategy_backtest_variant_launch_config_v1",
        "mode": mode,
        "exchange_connection_id": (
            str(payload.exchange_connection_id)
            if payload.exchange_connection_id is not None
            else None
        ),
        "market_type": market_type,
        "symbol": symbol,
        "capital_allocation_usd": str(payload.capital_allocation_usd),
        "entry_sizing": entry_sizing,
        "risk_mode": risk_mode,
        "direction": direction,
        "allowlist_scope": "paper_testnet_btcusdt_v1",
        "mainnet": False,
    }


def _strategy_launch_validation_error(*, reason: str, field: str) -> RoehubError:
    return RoehubError(
        code="strategy_launch.invalid_config",
        message="Strategy launch config is invalid",
        details={"reason": reason, "field": field},
    )


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


def _to_compatibility_readiness_response(
    *, report
) -> StrategyCompatibilityReadinessResponse:
    return StrategyCompatibilityReadinessResponse(
        compatibility_check_id=report.compatibility_check_id,
        market_data_requirement_id=report.market_data_requirement_id,
        strategy_id=report.strategy_id,
        source_job_id=report.source_job_id,
        source_variant_key=report.source_variant_key,
        strategy_spec_hash=report.strategy_spec_hash,
        instrument_key=report.instrument_key,
        market_type=report.market_type,
        timeframe=report.timeframe,
        compatibility_state=report.compatibility_state,
        compatibility_reason_codes=list(report.compatibility_reason_codes),
        market_data_state=report.market_data_state,
        market_data_reason_codes=list(report.market_data_reason_codes),
        market_data_stream_name=report.market_data_stream_name,
        market_data_stream_length=report.market_data_stream_length,
        market_data_last_message_id=report.market_data_last_message_id,
        market_data_last_observed_at=report.market_data_last_observed_at,
        market_data_age_seconds=report.market_data_age_seconds,
        launch_blocked=report.launch_blocked,
        launch_blocked_reason=report.launch_blocked_reason,
        checked_at=report.checked_at,
    )


def _record_compatibility_readiness(*, report) -> None:
    record_strategy_variant_compatibility(
        state=report.compatibility_state,
        reason=report.compatibility_reason_codes[0],
    )
    record_market_data_readiness(
        state=report.market_data_state,
        reason=report.market_data_reason_codes[0],
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
