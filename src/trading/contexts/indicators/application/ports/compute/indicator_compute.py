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

    Docs:
      - docs/architecture/indicators/indicators-overview.md
      - docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
      - docs/architecture/indicators/README.md
    Related:
      - src/trading/contexts/indicators/application/dto/compute_request.py
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
      - src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py
    """

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Estimate axis cardinality and variant count without running compute.

        Docs:
            - docs/architecture/indicators/indicators-overview.md
            - docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md

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

        Docs:
            - docs/architecture/indicators/indicators-overview.md
            - docs/architecture/indicators/README.md
            - docs/runbooks/indicators-why-nan.md

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

        Docs:
            - docs/architecture/indicators/indicators-overview.md
            - docs/runbooks/indicators-numba-warmup-jit.md
            - docs/runbooks/indicators-numba-cache-and-threads.md

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
