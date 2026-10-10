"""Use the existing installation configuration; never enable management or provision identity."""

from datetime import date, time
from uuid import UUID

from apps.control_agent.artifact_storage import ArtifactStorageService
from trading.contexts.backtest_artifacts.adapters.outbound.config.backtest_artifacts_runtime_config import (  # noqa: E501
    load_backtest_artifacts_runtime_config,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.gateway import (
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.management_repository import (  # noqa: E501
    PostgresArtifactManagementRepository,
)
from trading.contexts.backtest_artifacts.application.services.management import (
    ManagementError,
    Schedule,
)


class _NoPublicCommands:
    def require(self, **kwargs):
        raise ManagementError("forbidden")

    def validate(self, profile):
        raise ManagementError("forbidden")


def build_artifact_storage_service(environ):
    if not environ.get("ROEHUB_ARTIFACT_MANAGEMENT_INSTALLATION_ID"):
        return None
    config = load_backtest_artifacts_runtime_config(
        environ.get("ROEHUB_BACKTEST_ARTIFACTS_CONFIG", "/etc/roehub/backtest-artifacts.yaml")
    )
    dsn = environ.get("ROEHUB_STORAGE_POSTGRES_DSN") or environ["STRATEGY_PG_DSN"]
    repository = PostgresArtifactManagementRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=dsn),
        catalog=_NoPublicCommands(),
        access=_NoPublicCommands(),
    )
    # The rendered worker config is the registry. Additional roots must be provisioned
    # by an explicit installation config workflow, never by an HTTP path override.
    return ArtifactStorageService(
        repository=repository,
        installation_id=UUID(environ["ROEHUB_ARTIFACT_MANAGEMENT_INSTALLATION_ID"]),
        roots={"backtest-primary": config.artifact_root_path().absolute()},
        default_schedule=Schedule(
            interval_days=1,
            timezone="UTC",
            local_time=time(
                config.publish_schedule.full_rebuild_hour_utc,
                config.publish_schedule.full_rebuild_minute_utc,
            ),
            anchor=date.today(),
        ),
    )
