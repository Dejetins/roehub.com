from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

from trading.contexts.indicators.domain.errors import GridValidationError


class EstimateVariantsGuardExceeded(GridValidationError):
    """
    Raised when batch `total_variants` exceeds `max_variants_per_compute`.

    Docs: docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related: trading.contexts.indicators.application.services.grid_builder,
      apps.api.routes.indicators
    """

    def __init__(
        self,
        *,
        total_variants: int,
        max_variants_per_compute: int,
    ) -> None:
        """
        Build deterministic variants-guard error payload.

        Args:
            total_variants: Actual computed variants for the request batch.
            max_variants_per_compute: Configured maximum allowed variants per request.
        Returns:
            None.
        Assumptions:
            Caller passes integer values produced by deterministic estimator logic.
        Raises:
            None.
        Side Effects:
            Stores deterministic ordered details for API serialization.
        """
        self._details: Mapping[str, int | str] = OrderedDict(
            [
                ("error", "max_variants_per_compute_exceeded"),
                ("total_variants", int(total_variants)),
                ("max_variants_per_compute", int(max_variants_per_compute)),
            ]
        )
        message = (
            "variants guard exceeded: "
            f"total_variants={total_variants} > max_variants_per_compute={max_variants_per_compute}"
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
