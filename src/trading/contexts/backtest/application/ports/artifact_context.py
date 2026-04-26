from __future__ import annotations

from typing import Protocol

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
)


class BacktestArtifactContextUnavailable(RuntimeError):
    """
    Raised when the trusted artifact current pointer or manifests cannot be resolved.
    """


class BacktestArtifactContextResolver(Protocol):
    """
    Application port for selecting the currently published artifact context.
    """

    def resolve_context(
        self,
        *,
        coordinates: BacktestCoordinates,
    ) -> BacktestArtifactMetadata:
        """
        Resolve trusted artifact metadata for normalized coordinates.

        Args:
            coordinates: Normalized public backtest coordinates.
        Returns:
            BacktestArtifactMetadata: Active slot/current-pointer metadata.
        Assumptions:
            Implementations derive filesystem paths from trusted runtime config only.
        Raises:
            BacktestArtifactContextUnavailable: If required current/manifests are unavailable.
        Side Effects:
            Adapter-defined filesystem reads may occur.
        """
        ...


__all__ = [
    "BacktestArtifactContextResolver",
    "BacktestArtifactContextUnavailable",
]
