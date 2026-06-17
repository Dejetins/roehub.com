"""
Composition helpers for Strategy API module.

Docs:
  - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal
from typing import Any, Literal, Mapping
from uuid import UUID, uuid4

from fastapi import APIRouter
from starlette.requests import Request

from apps.api.exchange_control_client import (
    ExchangeControlAccountStateSnapshot,
    ExchangeControlClient,
    ExchangeControlClientError,
    build_exchange_control_client_from_environ,
)
from apps.api.monitoring import (
    record_strategy_capital_reservation,
    record_strategy_paper_accounting,
    record_strategy_position_ownership,
)
from apps.api.routes import build_strategies_router
from trading.contexts.backtest.adapters.outbound import (
    PostgresBacktestJobRepository,
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.live_execution.adapters.outbound import (
    InMemoryPaperAccountingRepository,
    InMemoryStrategyPositionOwnershipRepository,
    PostgresExchangeAccountProjectionRepository,
    PostgresPaperAccountingRepository,
    PostgresStrategyPositionOwnershipRepository,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import (
    CapitalReservationPaperAccountingService,
    ExchangeAccountProjectionService,
    StrategyPositionOwnershipService,
)
from trading.contexts.live_execution.domain import (
    ExchangeAccountProjection,
    ExchangeBalanceSnapshot,
    ExchangeInstrumentFilterSnapshot,
    ExchangeOpenOrderSnapshot,
    ExchangePositionSnapshot,
    ExpectedInstrumentConfig,
)
from trading.contexts.strategy.adapters.outbound import (
    InMemoryLiveStrategyProfileRepository,
    InMemoryStrategyEventRepository,
    InMemoryStrategyRepository,
    InMemoryStrategyRunRepository,
    InMemoryStrategySignalRepository,
    PostgresLiveStrategyProfileRepository,
    PostgresStrategyBacktestVariantProvenanceRepository,
    PostgresStrategyCompatibilityReadinessRepository,
    PostgresStrategyEventRepository,
    PostgresStrategyRepository,
    PostgresStrategyRunRepository,
    PostgresStrategySignalRepository,
    PsycopgStrategyPostgresGateway,
    RedisMarketDataReadinessReader,
    RedisStrategyLiveCandleStreamConfig,
    SystemStrategyClock,
    load_strategy_runtime_config,
    resolve_strategy_config_path,
)
from trading.contexts.strategy.application import (
    BacktestVariantLaunchReader,
    BacktestVariantLaunchSnapshot,
    CloneStrategyUseCase,
    CreateStrategyFromBacktestVariantUseCase,
    CreateStrategyUseCase,
    CurrentUser,
    CurrentUserProvider,
    DeleteStrategyUseCase,
    ExchangeConnectionReadiness,
    ExchangeConnectionReadinessChecker,
    ExchangeConnectionReadinessContext,
    GetMyStrategyUseCase,
    ListMyStrategiesUseCase,
    LiveStrategyProfileRepository,
    LiveStrategyProfileService,
    RestartStrategyUseCase,
    RunStrategyUseCase,
    StopStrategyUseCase,
    StrategyCompatibilityReadinessService,
    StrategyEventRepository,
    StrategyPositionOwnershipCoordinator,
    StrategyRepository,
    StrategyRunRepository,
    StrategySignalRepository,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId

_ENV_NAME_KEY = "ROEHUB_ENV"
_STRATEGY_FAIL_FAST_KEY = "STRATEGY_FAIL_FAST"
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"
_ALLOWED_ENVS = ("dev", "prod", "test")


@dataclass(frozen=True, slots=True)
class StrategyRuntimeSettings:
    """
    StrategyRuntimeSettings — runtime policy for Strategy API module wiring.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/wiring/modules/strategy.py
      - apps/api/main/app.py
      - apps/api/routes/strategies.py
    """

    env_name: str
    fail_fast: bool
    postgres_dsn: str

    def __post_init__(self) -> None:
        """
        Validate strategy runtime settings invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Settings are normalized by resolver before dataclass construction.
        Raises:
            ValueError: If one of invariants is violated.
        Side Effects:
            None.
        """
        if self.env_name not in _ALLOWED_ENVS:
            raise ValueError(
                f"StrategyRuntimeSettings.env_name must be one of {_ALLOWED_ENVS}, "
                f"got {self.env_name!r}"
            )


class IdentityPrincipalCurrentUserProvider(CurrentUserProvider):
    """
    IdentityPrincipalCurrentUserProvider — adapter from identity principal to Strategy CurrentUser.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/current_user.py
      - src/trading/contexts/identity/application/ports/current_user.py
      - apps/api/wiring/modules/strategy.py
    """

    def __init__(self, *, principal: CurrentUserPrincipal) -> None:
        """
        Store identity principal for current request scope.

        Args:
            principal: Identity current-user principal resolved from cookie.
        Returns:
            None.
        Assumptions:
            Principal identity has already passed authentication checks.
        Raises:
            ValueError: If principal is missing.
        Side Effects:
            None.
        """
        if principal is None:  # type: ignore[truthy-bool]
            raise ValueError("IdentityPrincipalCurrentUserProvider requires principal")
        self._principal = principal

    def require_current_user(self) -> CurrentUser:
        """
        Convert identity principal into Strategy CurrentUser port object.

        Args:
            None.
        Returns:
            CurrentUser: Strategy-layer current user value object.
        Assumptions:
            User id from identity principal is canonical shared-kernel UserId.
        Raises:
            ValueError: If principal payload is invalid.
        Side Effects:
            None.
        """
        return CurrentUser(user_id=self._principal.user_id)


class StrategyCurrentUserProviderDependency:
    """
    StrategyCurrentUserProviderDependency — FastAPI dependency resolving
    Strategy CurrentUserProvider.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/routes/strategies.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
      - src/trading/contexts/strategy/application/ports/current_user.py
    """

    def __init__(self, *, current_user_dependency: RequireCurrentUserDependency) -> None:
        """
        Initialize dependency bridge with identity current-user resolver.

        Args:
            current_user_dependency: Identity dependency resolving authenticated principal.
        Returns:
            None.
        Assumptions:
            Identity dependency raises deterministic unauthorized payloads when needed.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        if current_user_dependency is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "StrategyCurrentUserProviderDependency requires current_user_dependency"
            )
        self._current_user_dependency = current_user_dependency

    def __call__(self, request: Request) -> CurrentUserProvider:
        """
        Resolve request-scoped Strategy CurrentUserProvider from identity principal dependency.

        Args:
            request: FastAPI request object.
        Returns:
            CurrentUserProvider: Request-scoped provider for Strategy use-cases.
        Assumptions:
            Request contains authentication cookie consumed by identity dependency.
        Raises:
            HTTPException: Propagated unauthorized response from identity dependency.
        Side Effects:
            None.
        """
        principal = self._current_user_dependency(request)
        return IdentityPrincipalCurrentUserProvider(principal=principal)


@dataclass(frozen=True, slots=True)
class ExchangeControlReadinessChecker(ExchangeConnectionReadinessChecker):
    client: ExchangeControlClient
    account_projection_service: ExchangeAccountProjectionService | None = None

    def check_trading_ready(
        self,
        *,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        context: ExchangeConnectionReadinessContext | None = None,
    ) -> ExchangeConnectionReadiness:
        try:
            rows = self.client.list_connections(
                owner_user_id=str(owner_user_id),
                request_id="apps-api-live-profile-read-connection",
            )
        except ExchangeControlClientError:
            return ExchangeConnectionReadiness(
                eligible=False,
                reason="exchange_control_unavailable",
            )
        connection = next(
            (row for row in rows if row.connection_id == str(exchange_connection_id)),
            None,
        )
        if connection is None:
            return ExchangeConnectionReadiness(
                eligible=False,
                reason="exchange_connection_not_found",
            )
        if connection.status != "active":
            return ExchangeConnectionReadiness(
                eligible=False,
                reason="exchange_connection_not_active",
                exchange_name=connection.exchange_name,
                market_type=connection.market_type,
            )
        if context is not None and context.mode == "testnet":
            binding_blocker = _testnet_binding_blocker(
                connection_exchange=connection.exchange_name,
                connection_market_type=connection.market_type,
                connection_environment=connection.environment,
                context=context,
            )
            if binding_blocker is not None:
                return ExchangeConnectionReadiness(
                    eligible=False,
                    reason=binding_blocker,
                    exchange_name=connection.exchange_name,
                    market_type=connection.market_type,
                )
        if (
            connection.effective_capability != "trading"
            or connection.connection_readiness != "ready_for_trading"
        ):
            return ExchangeConnectionReadiness(
                eligible=False,
                reason=connection.connection_readiness_reason
                or "exchange_connection_not_ready_for_trading",
                exchange_name=connection.exchange_name,
                market_type=connection.market_type,
            )
        if _requires_safe_futures_short_guard(context=context):
            return self._check_safe_futures_short(
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
                exchange_name=connection.exchange_name,
                market_type=connection.market_type,
                context=context,
            )
        return ExchangeConnectionReadiness(
            eligible=True,
            reason="ready_for_trading",
            exchange_name=connection.exchange_name,
            market_type=connection.market_type,
        )

    def _check_safe_futures_short(
        self,
        *,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        exchange_name: str,
        market_type: str,
        context: ExchangeConnectionReadinessContext | None,
    ) -> ExchangeConnectionReadiness:
        if context is None:
            return ExchangeConnectionReadiness(
                eligible=False,
                reason="unsafe_futures_short",
                exchange_name=exchange_name,
                market_type=market_type,
            )
        if self.account_projection_service is None:
            return ExchangeConnectionReadiness(
                eligible=False,
                reason="account_projection_repository_unavailable",
                exchange_name=exchange_name,
                market_type=market_type,
            )
        instrument_key = f"{exchange_name}:{market_type}:{context.symbol}"
        requirement = ExpectedInstrumentConfig(
            instrument_key=instrument_key,
            market_type=market_type,
            side="short",
            expected_margin_mode="isolated",
            required_leverage=Decimal("1"),
            order_notional=context.notional,
            required_balance_asset="USDT",
        )
        try:
            self.account_projection_service.sync_connection(
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
                reader=_ExchangeControlAccountProjectionReader(
                    client=self.client,
                    instrument_keys=(instrument_key,),
                ),
                requirements=(requirement,),
            )
            readiness = self.account_projection_service.get_readiness(
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
                requirement=requirement,
            )
        except ExchangeControlClientError as error:
            return ExchangeConnectionReadiness(
                eligible=False,
                reason=str(error) or "exchange_account_state_read_failed",
                exchange_name=exchange_name,
                market_type=market_type,
            )
        if readiness.ready_for_risk:
            return ExchangeConnectionReadiness(
                eligible=True,
                reason="safe_testnet_futures_short_1x_isolated_verified",
                exchange_name=exchange_name,
                market_type=market_type,
            )
        return ExchangeConnectionReadiness(
            eligible=False,
            reason=readiness.reason_codes[0] if readiness.reason_codes else readiness.status,
            exchange_name=exchange_name,
            market_type=market_type,
        )


@dataclass(frozen=True, slots=True)
class _ExchangeControlAccountProjectionReader:
    client: ExchangeControlClient
    instrument_keys: tuple[str, ...]

    def read_account_projection(
        self, *, owner_user_id: UserId, exchange_connection_id: UUID
    ) -> ExchangeAccountProjection:
        snapshot = self.client.read_account_state(
            owner_user_id=str(owner_user_id),
            connection_id=str(exchange_connection_id),
            instrument_keys=self.instrument_keys,
            request_id="apps-api-safe-testnet-binding-account-read",
        )
        return _projection_from_exchange_control_snapshot(
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
            snapshot=snapshot,
        )


def _testnet_binding_blocker(
    *,
    connection_exchange: str,
    connection_market_type: str,
    connection_environment: str,
    context: ExchangeConnectionReadinessContext,
) -> str | None:
    if connection_exchange not in {"binance", "bybit"}:
        return "unsupported_exchange_connection"
    if connection_environment != "testnet":
        return "testnet_connection_required"
    if connection_market_type != context.market_type:
        return "exchange_connection_market_type_mismatch"
    return None


def _requires_safe_futures_short_guard(
    *, context: ExchangeConnectionReadinessContext | None
) -> bool:
    return (
        context is not None
        and context.mode == "testnet"
        and context.market_type == "futures"
        and context.direction == "short"
    )


def _projection_from_exchange_control_snapshot(
    *,
    owner_user_id: UserId,
    exchange_connection_id: UUID,
    snapshot: ExchangeControlAccountStateSnapshot,
) -> ExchangeAccountProjection:
    observed_at = snapshot.observed_at
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    else:
        observed_at = observed_at.astimezone(UTC)
    return ExchangeAccountProjection(
        account_snapshot_id=uuid4(),
        owner_user_id=owner_user_id,
        exchange_connection_id=exchange_connection_id,
        exchange_name=snapshot.exchange_name,
        market_type=snapshot.market_type,
        environment=snapshot.environment,
        account_mode=snapshot.account_mode,
        balances=tuple(
            ExchangeBalanceSnapshot(
                asset=item.asset,
                free=item.free,
                locked=item.locked,
                total=item.total,
            )
            for item in snapshot.balances
        ),
        positions=tuple(
            ExchangePositionSnapshot(
                instrument_key=item.instrument_key,
                side=_position_side(item.side),
                quantity=item.quantity,
                entry_price=item.entry_price,
                leverage=item.leverage,
                margin_mode=item.margin_mode,
                position_mode=item.position_mode,
            )
            for item in snapshot.positions
        ),
        open_orders=tuple(
            ExchangeOpenOrderSnapshot(
                instrument_key=item.instrument_key,
                exchange_order_ref=item.exchange_order_ref,
                side=_order_side(item.side),
                order_type=item.order_type,
                quantity=item.quantity,
                price=item.price,
                status=item.status,
            )
            for item in snapshot.open_orders
        ),
        instrument_filters=tuple(
            ExchangeInstrumentFilterSnapshot(
                instrument_key=item.instrument_key,
                tick_size=item.tick_size,
                step_size=item.step_size,
                min_qty=item.min_qty,
                min_notional=item.min_notional,
                max_leverage=item.max_leverage,
            )
            for item in snapshot.instrument_filters
        ),
        source_hash=snapshot.source_hash,
        observed_at=observed_at,
        synced_at=observed_at,
        sync_status="fresh" if snapshot.sync_status == "fresh" else "degraded",
        sync_reason=snapshot.sync_reason,
        metadata={"source": "exchange_control_account_state"},
    )


def _position_side(value: str) -> Literal["long", "short", "net"]:
    normalized = value.strip().casefold()
    if normalized == "long":
        return "long"
    if normalized == "short":
        return "short"
    return "net"


def _order_side(value: str) -> Literal["buy", "sell"]:
    normalized = value.strip().casefold()
    if normalized == "sell":
        return "sell"
    return "buy"



def is_strategy_api_enabled(*, environ: Mapping[str, str]) -> bool:
    """
    Resolve Strategy API enable toggle from source-of-truth runtime config.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - apps/api/main/app.py
      - configs/dev/strategy.yaml
      - src/trading/contexts/strategy/adapters/outbound/config/strategy_runtime_config.py

    Args:
        environ: Runtime environment mapping.
    Returns:
        bool: True when Strategy API router should be included.
    Assumptions:
        Config path resolution uses `ROEHUB_STRATEGY_CONFIG` or
        `configs/<ROEHUB_ENV>/strategy.yaml`.
    Raises:
        FileNotFoundError: If resolved config path does not exist.
        ValueError: If strategy config payload or scalar overrides are invalid.
    Side Effects:
        Reads one YAML config file from disk.
    """
    config_path = resolve_strategy_config_path(environ=environ)
    runtime_config = load_strategy_runtime_config(config_path, environ=environ)
    return runtime_config.api.enabled


def build_strategy_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    """
    Build fully wired Strategy router from runtime settings and shared identity dependency.

    Args:
        environ: Runtime environment mapping.
        current_user_dependency: Shared identity dependency resolving authenticated principal.
    Returns:
        APIRouter: Strategy API router.
    Assumptions:
        Strategy persistence can use Postgres or deterministic in-memory
        fallback depending on settings.
    Raises:
        ValueError: If fail-fast settings require Postgres DSN but it is missing.
    Side Effects:
        None.
    """
    settings = _resolve_strategy_runtime_settings(environ=environ)
    strategy_repository, run_repository, event_repository = _build_repositories(settings=settings)
    profile_repository = _build_live_profile_repository(settings=settings)
    position_ownership_coordinator = _build_position_ownership_coordinator(
        settings=settings,
    )
    paper_accounting_service = _build_paper_accounting_service(settings=settings)
    clock = SystemStrategyClock()
    compatibility_readiness_service = _build_compatibility_readiness_service(
        environ=environ,
        settings=settings,
        strategy_repository=strategy_repository,
        event_repository=event_repository,
        clock=clock,
    )
    account_projection_service = _build_account_projection_service(settings=settings)

    create_use_case = CreateStrategyUseCase(
        repository=strategy_repository,
        clock=clock,
        event_repository=event_repository,
    )
    clone_use_case = CloneStrategyUseCase(
        repository=strategy_repository,
        clock=clock,
        event_repository=event_repository,
    )
    list_use_case = ListMyStrategiesUseCase(repository=strategy_repository)
    get_use_case = GetMyStrategyUseCase(repository=strategy_repository)
    run_use_case = RunStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        clock=clock,
        event_repository=event_repository,
        compatibility_readiness_checker=compatibility_readiness_service,
        live_profile_repository=profile_repository,
        position_ownership_coordinator=position_ownership_coordinator,
        capital_reservation_coordinator=paper_accounting_service,
    )
    stop_use_case = StopStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        clock=clock,
        event_repository=event_repository,
        position_ownership_coordinator=position_ownership_coordinator,
    )
    restart_use_case = RestartStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        clock=clock,
        event_repository=event_repository,
        position_ownership_coordinator=position_ownership_coordinator,
    )
    delete_use_case = DeleteStrategyUseCase(
        repository=strategy_repository,
        clock=clock,
        event_repository=event_repository,
    )
    exchange_client = build_exchange_control_client_from_environ(environ=environ)
    live_profile_service = LiveStrategyProfileService(
        strategy_repository=strategy_repository,
        profile_repository=profile_repository,
        clock=clock,
        event_repository=event_repository,
        exchange_connection_checker=(
            ExchangeControlReadinessChecker(
                client=exchange_client,
                account_projection_service=account_projection_service,
            )
            if exchange_client is not None
            else None
        ),
        compatibility_readiness_checker=compatibility_readiness_service,
    )
    create_strategy_from_variant_use_case = _build_create_strategy_from_variant_use_case(
        settings=settings,
        job_repository=_build_backtest_job_repository(settings=settings),
    )

    current_user_provider_dependency = StrategyCurrentUserProviderDependency(
        current_user_dependency=current_user_dependency,
    )

    return build_strategies_router(
        create_use_case=create_use_case,
        clone_use_case=clone_use_case,
        list_use_case=list_use_case,
        get_use_case=get_use_case,
        run_use_case=run_use_case,
        stop_use_case=stop_use_case,
        restart_use_case=restart_use_case,
        delete_use_case=delete_use_case,
        current_user_provider_dependency=current_user_provider_dependency,
        live_profile_service=live_profile_service,
        current_user_principal_dependency=current_user_dependency,
        compatibility_readiness_service=compatibility_readiness_service,
        create_strategy_from_variant_use_case=create_strategy_from_variant_use_case,
    )



def _build_repositories(
    *,
    settings: StrategyRuntimeSettings,
) -> tuple[StrategyRepository, StrategyRunRepository, StrategyEventRepository]:
    """
    Build strategy repositories using Postgres when configured or deterministic in-memory fallback.

    Args:
        settings: Resolved strategy runtime settings.
    Returns:
        tuple[StrategyRepository, StrategyRunRepository, StrategyEventRepository]:
            Repository adapters.
    Assumptions:
        In-memory fallback is allowed only when fail-fast mode is disabled.
    Raises:
        ValueError: If fail-fast mode requires Postgres DSN and it is absent.
    Side Effects:
        None.
    """
    if settings.postgres_dsn:
        gateway = PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)
        return (
            PostgresStrategyRepository(gateway=gateway),
            PostgresStrategyRunRepository(gateway=gateway),
            PostgresStrategyEventRepository(gateway=gateway),
        )

    if settings.fail_fast:
        raise ValueError(
            f"{_STRATEGY_PG_DSN_KEY} is required when strategy fail-fast mode is enabled"
        )

    return (
        InMemoryStrategyRepository(),
        InMemoryStrategyRunRepository(),
        InMemoryStrategyEventRepository(),
    )


def _build_live_profile_repository(
    *,
    settings: StrategyRuntimeSettings,
) -> LiveStrategyProfileRepository:
    if settings.postgres_dsn:
        return PostgresLiveStrategyProfileRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)
        )
    if settings.fail_fast:
        raise ValueError(
            f"{_STRATEGY_PG_DSN_KEY} is required when strategy fail-fast mode is enabled"
        )
    return InMemoryLiveStrategyProfileRepository()


def _build_backtest_job_repository(
    *,
    settings: StrategyRuntimeSettings,
) -> BacktestJobRepository | None:
    if not settings.postgres_dsn:
        return None
    return PostgresBacktestJobRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=settings.postgres_dsn)
    )


def _build_create_strategy_from_variant_use_case(
    *,
    settings: StrategyRuntimeSettings,
    job_repository: BacktestJobRepository | None,
) -> CreateStrategyFromBacktestVariantUseCase | None:
    if not settings.postgres_dsn or job_repository is None:
        return None
    strategy_gateway = PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)
    return CreateStrategyFromBacktestVariantUseCase(
        variant_reader=_BacktestJobRepositoryVariantLaunchReader(repository=job_repository),
        strategy_repository=PostgresStrategyRepository(gateway=strategy_gateway),
        provenance_repository=PostgresStrategyBacktestVariantProvenanceRepository(
            gateway=strategy_gateway,
        ),
        event_repository=PostgresStrategyEventRepository(gateway=strategy_gateway),
        clock=SystemStrategyClock(),
    )


def _build_position_ownership_coordinator(
    *,
    settings: StrategyRuntimeSettings,
) -> StrategyPositionOwnershipCoordinator:
    if settings.postgres_dsn:
        return StrategyPositionOwnershipService(
            repository=PostgresStrategyPositionOwnershipRepository(
                gateway=PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn),
            ),
            on_transition=lambda result, reason: record_strategy_position_ownership(
                result=result,
                reason=reason,
            ),
        )
    return StrategyPositionOwnershipService(
        repository=InMemoryStrategyPositionOwnershipRepository(),
        on_transition=lambda result, reason: record_strategy_position_ownership(
            result=result,
            reason=reason,
        ),
    )


def _build_paper_accounting_service(
    *,
    settings: StrategyRuntimeSettings,
) -> CapitalReservationPaperAccountingService:
    if settings.postgres_dsn:
        gateway = PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)
        return CapitalReservationPaperAccountingService(
            repository=PostgresPaperAccountingRepository(gateway=gateway),
            account_projection_repository=PostgresExchangeAccountProjectionRepository(
                gateway=gateway
            ),
            clock=SystemLiveExecutionClock(),
            on_capital_reservation=lambda result, reason: record_strategy_capital_reservation(
                result=result,
                reason=reason,
            ),
            on_paper_accounting=lambda result, reason: record_strategy_paper_accounting(
                result=result,
                reason=reason,
            ),
        )
    return CapitalReservationPaperAccountingService(
        repository=InMemoryPaperAccountingRepository(),
        account_projection_repository=None,
        clock=SystemLiveExecutionClock(),
        on_capital_reservation=lambda result, reason: record_strategy_capital_reservation(
            result=result,
            reason=reason,
        ),
        on_paper_accounting=lambda result, reason: record_strategy_paper_accounting(
            result=result,
            reason=reason,
        ),
    )


def _build_account_projection_service(
    *,
    settings: StrategyRuntimeSettings,
) -> ExchangeAccountProjectionService | None:
    if not settings.postgres_dsn:
        return None
    return ExchangeAccountProjectionService(
        repository=PostgresExchangeAccountProjectionRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)
        ),
        clock=SystemLiveExecutionClock(),
    )


def _build_compatibility_readiness_service(
    *,
    environ: Mapping[str, str],
    settings: StrategyRuntimeSettings,
    strategy_repository: StrategyRepository | None,
    event_repository: StrategyEventRepository | None,
    clock: SystemStrategyClock,
) -> StrategyCompatibilityReadinessService:
    repository = None
    if settings.postgres_dsn:
        repository = PostgresStrategyCompatibilityReadinessRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)
        )
    redis_reader = None
    try:
        runtime_config = load_strategy_runtime_config(
            resolve_strategy_config_path(environ=environ),
            environ=environ,
        )
        redis_config = runtime_config.live_worker.redis_streams
        if redis_config.enabled:
            redis_reader = RedisMarketDataReadinessReader(
                config=RedisStrategyLiveCandleStreamConfig(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.db,
                    password_env=redis_config.password_env,
                    socket_timeout_s=redis_config.socket_timeout_s,
                    connect_timeout_s=redis_config.connect_timeout_s,
                    stream_prefix=redis_config.stream_prefix,
                    consumer_group=redis_config.consumer_group,
                    consumer_name="api-readiness",
                    read_count=1,
                    block_ms=0,
                ),
                environ=environ,
            )
    except Exception:
        redis_reader = None
    return StrategyCompatibilityReadinessService(
        strategy_repository=strategy_repository,
        compatibility_repository=repository,
        market_data_reader=redis_reader,
        event_repository=event_repository,
        clock=clock,
    )


def _build_signal_repository(
    *,
    settings: StrategyRuntimeSettings,
) -> StrategySignalRepository:
    if settings.postgres_dsn:
        return PostgresStrategySignalRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)
        )
    if settings.fail_fast:
        raise ValueError(
            f"{_STRATEGY_PG_DSN_KEY} is required when strategy fail-fast mode is enabled"
        )
    return InMemoryStrategySignalRepository()


class _BacktestJobRepositoryVariantLaunchReader(BacktestVariantLaunchReader):
    def __init__(self, *, repository: BacktestJobRepository) -> None:
        self._repository = repository

    def get(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestVariantLaunchSnapshot:
        job = self._repository.get(job_id=job_id)
        if job is None:
            raise RoehubError(
                code="strategy_variant_launch.not_found",
                message="Backtest job was not found",
                details={"reason": "not_found", "job_id": str(job_id)},
            )
        if job.user_id != user_id:
            raise RoehubError(
                code="strategy_variant_launch.forbidden",
                message="Backtest job does not belong to current user",
                details={"reason": "forbidden", "job_id": str(job_id)},
            )
        row = self._repository.get_top_variant_by_public_key(
            job_id=job_id,
            public_variant_key=variant_key,
        )
        if row is None:
            raise RoehubError(
                code="strategy_variant_launch.not_found",
                message="Backtest variant was not found",
                details={
                    "reason": "not_found",
                    "job_id": str(job_id),
                    "variant_key": variant_key,
                },
            )
        request = dict(job.request_json)
        coordinates = _mapping(request.get("coordinates"))
        payload = dict(row.payload_json)
        if job.market_id is None:
            raise RoehubError(
                code="strategy_variant_launch.not_launchable",
                message="Backtest job has no launchable market id",
                details={"reason": "not_launchable", "job_id": str(job_id)},
            )
        return BacktestVariantLaunchSnapshot(
            job_id=job.job_id,
            owner_user_id=job.user_id,
            job_state=job.state,
            request_hash=job.request_hash,
            result_config_hash=job.engine_params_hash,
            market_id=int(job.market_id),
            exchange=str(coordinates.get("exchange", "binance")),
            market_type=str(coordinates.get("market_type", "spot")),
            symbol=str(coordinates.get("symbol", job.symbol)),
            timeframe=str(job.timeframe),
            variant_key=str(payload.get("public_variant_key") or variant_key),
            variant_hash=str(payload.get("variant_hash") or row.variant_key),
            indicator_variant_hash=(
                str(payload.get("indicator_variant_hash") or row.indicator_variant_key)
                if (payload.get("indicator_variant_hash") or row.indicator_variant_key)
                else None
            ),
            rank=row.rank,
            summary_metrics=dict(row.summary_metrics_json),
            canonical_variant_params=_mapping(payload.get("canonical_variant_params")),
            readable_params=_mapping(payload.get("readable_params")),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}



def _resolve_strategy_runtime_settings(*, environ: Mapping[str, str]) -> StrategyRuntimeSettings:
    """
    Resolve Strategy runtime settings with environment-aware fail-fast policy.

    Args:
        environ: Runtime environment mapping.
    Returns:
        StrategyRuntimeSettings: Normalized settings object.
    Assumptions:
        Missing `ROEHUB_ENV` defaults to `dev`.
    Raises:
        ValueError: If env values are invalid.
    Side Effects:
        None.
    """
    env_name = _resolve_env_name(environ=environ)
    fail_fast = _resolve_fail_fast(environ=environ, env_name=env_name)
    postgres_dsn = environ.get(_STRATEGY_PG_DSN_KEY, "").strip()

    return StrategyRuntimeSettings(
        env_name=env_name,
        fail_fast=fail_fast,
        postgres_dsn=postgres_dsn,
    )



def _resolve_env_name(*, environ: Mapping[str, str]) -> str:
    """
    Resolve normalized runtime environment name for Strategy wiring.

    Args:
        environ: Runtime environment mapping.
    Returns:
        str: Environment literal (`dev`, `prod`, `test`).
    Assumptions:
        Missing env variable defaults to `dev`.
    Raises:
        ValueError: If env value is not in allowed set.
    Side Effects:
        None.
    """
    raw_env = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if raw_env not in _ALLOWED_ENVS:
        raise ValueError(
            f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {raw_env!r}"
        )
    return raw_env



def _resolve_fail_fast(*, environ: Mapping[str, str], env_name: str) -> bool:
    """
    Resolve Strategy fail-fast mode from explicit override or environment default policy.

    Args:
        environ: Runtime environment mapping.
        env_name: Normalized environment name.
    Returns:
        bool: True when fail-fast mode should be enabled.
    Assumptions:
        Default policy enables fail-fast in `prod` and disables in `dev`/`test`.
    Raises:
        ValueError: If explicit override value is invalid.
    Side Effects:
        None.
    """
    raw_value = environ.get(_STRATEGY_FAIL_FAST_KEY)
    if raw_value is None:
        return env_name == "prod"

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_STRATEGY_FAIL_FAST_KEY} must be boolean-like value, got {raw_value!r}"
    )
