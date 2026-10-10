"""Server-side authorization/catalog boundaries for the managed artifact inbox."""

from typing import Protocol
from uuid import UUID

from trading.contexts.backtest_artifacts.application.services.management import Coordinate, Profile


class ProfileCatalog(Protocol):
    def validate(self, profile: Profile) -> Profile: ...


class CoordinateAccess(Protocol):
    def require(self, *, organization_id: UUID, actor_id: UUID, coordinate: Coordinate) -> None:
        """Resolve current server-side catalog access or raise before any write."""
        ...
