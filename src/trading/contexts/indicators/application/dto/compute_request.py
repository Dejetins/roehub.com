from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.indicators.domain.specifications import GridSpec

from .candle_arrays import CandleArrays


@dataclass(frozen=True, slots=True)
class ComputeRequest:
    """
    Application request envelope for indicator compute.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .candle_arrays, .indicator_tensor
    """

    candles: CandleArrays
    grid: GridSpec
    max_variants_guard: int
    dtype: str = "float32"

    def __post_init__(self) -> None:
        """
        Validate request-level compute invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            v1 contract only supports float32 output dtype.
        Raises:
            ValueError: If required payload is missing, guard is invalid, or dtype is unsupported.
        Side Effects:
            None.
        """
        if self.candles is None:  # type: ignore[truthy-bool]
            raise ValueError("ComputeRequest requires candles")
        if self.grid is None:  # type: ignore[truthy-bool]
            raise ValueError("ComputeRequest requires grid")
        if self.max_variants_guard <= 0:
            raise ValueError("ComputeRequest requires max_variants_guard > 0")
        if self.dtype != "float32":
            raise ValueError("ComputeRequest currently supports only dtype='float32'")
