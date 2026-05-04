"""
Composition helpers for Strategy API module.

Docs:
  - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from dataclasses import dataclass
from typing import Mapping

from fastapi import APIRouter
from starlette.requests import Request

from apps.api.routes import build_strategies_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.adapters.outbound import (
    InMemoryStrategyEventRepository,
    InMemoryStrategyRepository,
    InMemoryStrategyRunRepository,
    PostgresStrategyEventRepository,
    PostgresStrategyRepository,
    PostgresStrategyRunRepository,
    PsycopgStrategyPostgresGateway,
    SystemStrategyClock,
    load_strategy_runtime_config,
    resolve_strategy_config_path,
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
    StrategyEventRepository,
    StrategyRepository,
    StrategyRunRepository,
)

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
    clock = SystemStrategyClock()

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
    )
    stop_use_case = StopStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        clock=clock,
        event_repository=event_repository,
    )
    delete_use_case = DeleteStrategyUseCase(
        repository=strategy_repository,
        clock=clock,
        event_repository=event_repository,
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
        delete_use_case=delete_use_case,
        current_user_provider_dependency=current_user_provider_dependency,
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
