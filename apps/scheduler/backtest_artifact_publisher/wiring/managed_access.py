"""Worker reauthorization using active membership and the current Market Data catalog."""

from dataclasses import dataclass

from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.backtest_artifacts.application.services.management import ManagementError
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
    artifact_market_id_from_coordinates_v2,
)
from trading.contexts.market_data.application.use_cases.search_enabled_tradable_instruments import (
    SearchEnabledTradableInstrumentsUseCase,
)
from trading.shared_kernel.primitives import MarketId


@dataclass(frozen=True)
class ManagedCoordinateAccess:
    gateway: BacktestPostgresGateway
    search: SearchEnabledTradableInstrumentsUseCase

    def require(self, *, organization_id, actor_id, coordinate):
        member = self.gateway.fetch_one(
            query="""SELECT m.role FROM identity_memberships m
            JOIN identity_organizations o USING(organization_id)
            WHERE m.organization_id=%(org)s AND m.user_id=%(actor)s
            AND m.status='active' AND o.status='active'""",
            parameters={"org": organization_id, "actor": actor_id},
        )
        if member is None or member["role"] not in ("owner", "admin", "operator"):
            raise ManagementError("forbidden")
        market = artifact_market_id_from_coordinates_v2(ArtifactCoordinatesV2(*coordinate.key))
        candidates = self.search.execute(market_id=MarketId(market), q=coordinate.symbol, limit=100)
        if not any(str(c.symbol) == coordinate.symbol for c in candidates):
            raise ManagementError("coordinate_forbidden")
