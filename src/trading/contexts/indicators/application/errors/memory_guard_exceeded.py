from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

from trading.contexts.indicators.domain.errors import GridValidationError


class EstimateMemoryGuardExceeded(GridValidationError):
    """
    Raised when batch estimate exceeds `max_compute_bytes_total`.

    Docs: docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related: trading.contexts.indicators.application.services.grid_builder,
      apps.api.routes.indicators
    """

    def __init__(
        self,
        *,
        estimated_memory_bytes: int,
        max_compute_bytes_total: int,
    ) -> None:
        """
        Build deterministic memory-guard error payload.

        Args:
            estimated_memory_bytes: Actual estimated total memory bytes for request batch.
            max_compute_bytes_total: Configured total compute bytes limit.
        Returns:
            None.
        Assumptions:
            Caller passes integer values from deterministic estimation policy.
        Raises:
            None.
        Side Effects:
            Stores deterministic ordered details for API serialization.
        """
        self._details: Mapping[str, int | str] = OrderedDict(
            [
                ("error", "max_compute_bytes_total_exceeded"),
                ("estimated_memory_bytes", int(estimated_memory_bytes)),
                ("max_compute_bytes_total", int(max_compute_bytes_total)),
            ]
        )
        message = (
            "memory guard exceeded: "
            f"estimated_memory_bytes={estimated_memory_bytes} > "
            f"max_compute_bytes_total={max_compute_bytes_total}"
        )
        super().__init__(message)

    @property
    def details(self) -> Mapping[str, int | str]:
        """
        Return deterministic payload for HTTP 422 mapping.

        Args:
            None.
        Returns:
            Mapping[str, int | str]: Ordered details payload.
        Assumptions:
            Key order is preserved for deterministic API responses.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._details
