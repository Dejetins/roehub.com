from __future__ import annotations

from typing import Protocol

from trading.contexts.indicators.application.dto import (
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
)
from trading.contexts.indicators.domain.specifications import GridSpec


class IndicatorCompute(Protocol):
    """
    Port for indicator estimate/compute execution.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: ....application.dto.compute_request, ....domain.specifications.grid_spec
    """

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Estimate axis cardinality and variant count without running compute.

        Args:
            grid: Grid specification for one indicator.
            max_variants_guard: Upper guard for allowed variant count.
        Returns:
            EstimateResult: Materialized estimate metadata.
        Assumptions:
            Guard is enforced by implementation before producing a successful estimate.
        Raises:
            GridValidationError: If the grid is invalid or exceeds the guard.
            UnknownIndicatorError: If the indicator id is not supported.
        Side Effects:
            None.
        """
        ...

    def compute(self, req: ComputeRequest) -> IndicatorTensor:
        """
        Compute the indicator tensor for the provided request.

        Args:
            req: Compute request with candles, grid, and guard.
        Returns:
            IndicatorTensor: Tensor output in declared layout and float32 dtype.
        Assumptions:
            NaN values from candles are propagated according to nan policy.
        Raises:
            GridValidationError: If the request grid is invalid or exceeds guard.
            UnknownIndicatorError: If the indicator id is not supported.
            MissingInputSeriesError: If required input series are not available.
        Side Effects:
            None.
        """
        ...

    def warmup(self) -> None:
        """
        Trigger optional compute-engine warmup on process startup.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup is idempotent or safely repeatable by implementation.
        Raises:
            GridValidationError: If warmup configuration is invalid.
        Side Effects:
            May initialize internal caches or JIT state in implementation.
        """
        ...
