from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

from .grid_validation_error import GridValidationError


class ComputeBudgetExceeded(GridValidationError):
    """
    Raised when total compute memory estimate exceeds configured budget.

    Docs: docs/architecture/indicators/indicators-compute-engine-core.md
    Related: trading.contexts.indicators.adapters.outbound.compute_numba.kernels._common,
      trading.contexts.indicators.adapters.outbound.compute_numba.engine
    """

    def __init__(
        self,
        *,
        t: int,
        variants: int,
        bytes_out: int,
        bytes_total_est: int,
        max_compute_bytes_total: int,
    ) -> None:
        """
        Build deterministic budget-exceeded payload and message.

        Args:
            t: Time dimension length.
            variants: Variant count across all axes.
            bytes_out: Estimated output tensor bytes.
            bytes_total_est: Estimated total compute bytes including workspace.
            max_compute_bytes_total: Configured maximum total compute budget.
        Returns:
            None.
        Assumptions:
            All values are positive integers produced by guard helpers.
        Raises:
            None.
        Side Effects:
            Stores deterministic error details for downstream API mapping.
        """
        self._details: Mapping[str, int] = OrderedDict(
            [
                ("T", int(t)),
                ("V", int(variants)),
                ("bytes_out", int(bytes_out)),
                ("bytes_total_est", int(bytes_total_est)),
                ("max_compute_bytes_total", int(max_compute_bytes_total)),
            ]
        )
        message = (
            "compute budget exceeded: "
            f"total_est={bytes_total_est} > max_compute_bytes_total={max_compute_bytes_total}"
        )
        super().__init__(message)

    @property
    def details(self) -> Mapping[str, int]:
        """
        Return stable ordered details payload for diagnostics and API layer.

        Args:
            None.
        Returns:
            Mapping[str, int]: Deterministic details mapping.
        Assumptions:
            Mapping keys order is preserved for predictable error rendering.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._details
