from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId


@dataclass(frozen=True, slots=True)
class EstimateResult:
    """
    Compute estimate result without executing indicator kernels.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: ...domain.specifications.grid_spec, ..ports.compute.indicator_compute
    """

    indicator_id: IndicatorId
    axes: tuple[AxisDef, ...]
    variants: int
    max_variants_guard: int

    def __post_init__(self) -> None:
        """
        Validate estimate consistency against guard constraints.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variants are already computed from materialized axes by the caller.
        Raises:
            ValueError: If required fields are missing, counts are non-positive,
                or variants exceed guard.
        Side Effects:
            None.
        """
        if self.indicator_id is None:  # type: ignore[truthy-bool]
            raise ValueError("EstimateResult requires indicator_id")
        if self.max_variants_guard <= 0:
            raise ValueError("EstimateResult requires max_variants_guard > 0")
        if self.variants <= 0:
            raise ValueError("EstimateResult requires variants > 0")
        if self.variants > self.max_variants_guard:
            raise ValueError("EstimateResult variants exceed max_variants_guard")
