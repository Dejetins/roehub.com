from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId, Layout


@dataclass(frozen=True, slots=True)
class TensorMeta:
    """
    Minimal metadata envelope for indicator tensor transport.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .compute_request, .estimate_result
    """

    t: int
    variants: int
    nan_policy: str = "propagate"
    compute_ms: int | None = None

    def __post_init__(self) -> None:
        """
        Validate tensor metadata invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `t` and `variants` are positive, and NaN policy names are stable identifiers.
        Raises:
            ValueError: If shape metadata is non-positive or optional timing is invalid.
        Side Effects:
            Normalizes `nan_policy` by stripping spaces.
        """
        if self.t <= 0:
            raise ValueError("TensorMeta requires t > 0")
        if self.variants <= 0:
            raise ValueError("TensorMeta requires variants > 0")

        normalized_policy = self.nan_policy.strip()
        object.__setattr__(self, "nan_policy", normalized_policy)
        if not normalized_policy:
            raise ValueError("TensorMeta requires a non-empty nan_policy")

        if self.compute_ms is not None and self.compute_ms < 0:
            raise ValueError("TensorMeta compute_ms must be >= 0 when provided")


@dataclass(frozen=True, slots=True)
class IndicatorTensor:
    """
    Indicator compute output as tensor values plus axis metadata.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .estimate_result, ...domain.entities.layout
    """

    indicator_id: IndicatorId
    layout: Layout
    axes: tuple[AxisDef, ...]
    values: np.ndarray
    meta: TensorMeta

    def __post_init__(self) -> None:
        """
        Validate dtype, axis-cardinality, and layout-to-shape consistency.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `values` stores one time dimension and one or more variant
                dimensions according to layout.
        Raises:
            ValueError: If metadata is missing, dtype is not float32, or shape
                does not match layout/meta.
        Side Effects:
            None.
        """
        if self.indicator_id is None:  # type: ignore[truthy-bool]
            raise ValueError("IndicatorTensor requires indicator_id")
        if self.meta is None:  # type: ignore[truthy-bool]
            raise ValueError("IndicatorTensor requires meta")

        try:
            if self.values.ndim < 2:
                raise ValueError("IndicatorTensor values must have at least 2 dimensions")
            if self.values.dtype != np.float32:
                raise ValueError("IndicatorTensor values dtype must be float32")
        except AttributeError as error:
            raise ValueError("IndicatorTensor values must be a numpy ndarray") from error

        axis_variants = 1
        for axis in self.axes:
            axis_variants = axis_variants * axis.length()
        if axis_variants != self.meta.variants:
            raise ValueError("IndicatorTensor meta.variants must equal product of axis lengths")

        if self.layout is Layout.TIME_MAJOR:
            if self.values.shape[0] != self.meta.t:
                raise ValueError("TIME_MAJOR tensor must have values.shape[0] == meta.t")
            trailing_product = np.prod(self.values.shape[1:], dtype=np.int64)
            if trailing_product != self.meta.variants:
                raise ValueError(
                    "TIME_MAJOR tensor trailing shape product must equal "
                    "meta.variants"
                )
            return

        if self.layout is Layout.VARIANT_MAJOR:
            if self.values.shape[-1] != self.meta.t:
                raise ValueError("VARIANT_MAJOR tensor must have values.shape[-1] == meta.t")
            leading_product = np.prod(self.values.shape[:-1], dtype=np.int64)
            if leading_product != self.meta.variants:
                raise ValueError(
                    "VARIANT_MAJOR tensor leading shape product must equal "
                    "meta.variants"
                )
            return

        raise ValueError(f"Unsupported layout '{self.layout}'")
