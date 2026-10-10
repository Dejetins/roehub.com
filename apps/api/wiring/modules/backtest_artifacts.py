"""Opt-in artifact HTTP wiring; constructs adapters without activating workers."""

from pathlib import Path
from uuid import UUID

from apps.api.routes.backtest_artifacts import build_artifact_router
from apps.api.wiring.artifact_service import ArtifactHttpService
from apps.api.wiring.modules.market_data_reference import build_market_data_reference_use_cases
from apps.api.wiring.modules.research_tenancy import build_research_organization_scope_resolver
from trading.contexts.backtest_artifacts.adapters.outbound.config.backtest_artifacts_runtime_config import (  # noqa: E501
    load_backtest_artifacts_runtime_config,
)
from trading.contexts.backtest_artifacts.adapters.outbound.defaults import (
    YamlBacktestGridDefaultsProvider,
)
from trading.contexts.backtest_artifacts.adapters.outbound.defaults.management_catalog import (
    ManagementCatalog,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.gateway import (
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.management_repository import (  # noqa: E501
    PostgresArtifactManagementRepository,
)
from trading.contexts.backtest_artifacts.application.services.management import ManagementError
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
    artifact_market_id_from_coordinates_v2,
)
from trading.contexts.indicators.adapters.outbound.registry import YamlIndicatorRegistry
from trading.contexts.indicators.domain.definitions import all_defs
from trading.shared_kernel.primitives import MarketId


class CatalogReadAccess:
    def __init__(self, search):
        self.search = search

    def require(self, *, organization_id, actor_id, coordinate):
        # Membership is checked transactionally by the repository, including for viewers.
        market = artifact_market_id_from_coordinates_v2(ArtifactCoordinatesV2(*coordinate.key))
        rows = self.search.execute(market_id=MarketId(market), q=coordinate.symbol, limit=100)
        if not any(str(row.symbol) == coordinate.symbol for row in rows):
            raise ManagementError("coordinate_forbidden")


def build_artifact_repository(environ, access):
    dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    config_path = environ.get(
        "ROEHUB_BACKTEST_ARTIFACTS_CONFIG", "/etc/roehub/backtest-artifacts.yaml"
    )
    indicator_path = environ.get("ROEHUB_INDICATORS_CONFIG", "/etc/roehub/indicators.yaml")
    if not dsn or not Path(config_path).is_file() or not Path(indicator_path).is_file():
        return None
    config = load_backtest_artifacts_runtime_config(config_path)
    catalog = ManagementCatalog(
        defaults=YamlBacktestGridDefaultsProvider.from_yaml(config_path=Path(indicator_path)),
        registry=YamlIndicatorRegistry.from_yaml(defs=all_defs(), config_path=indicator_path),
        config=config,
    )
    return PostgresArtifactManagementRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=dsn), catalog=catalog, access=access
    )


def build_artifact_http_router(*, environ, current_user_dependency):
    service = None
    if environ.get("ROEHUB_ARTIFACT_MANAGEMENT_INSTALLATION_ID"):
        reference = build_market_data_reference_use_cases(environ=environ)
        repo = build_artifact_repository(
            environ, CatalogReadAccess(reference.search_enabled_tradable_instruments)
        )
        service = (
            ArtifactHttpService(repo, UUID(environ["ROEHUB_ARTIFACT_MANAGEMENT_INSTALLATION_ID"]))
            if repo
            else None
        )
    return build_artifact_router(
        service=service,
        current_user_dependency=current_user_dependency,
        scope_resolver=build_research_organization_scope_resolver(environ=environ),
    )
