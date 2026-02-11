from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

GridValue = int | float | str


class GridParamSpec(Protocol):
    """
    Contract for materializing one grid parameter axis.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .grid_spec, ..entities.param_def
    """

    name: str

    def materialize(self) -> tuple[GridValue, ...]:
        """
        Materialize axis values in deterministic order.

        Args:
            None.
        Returns:
            tuple[GridValue, ...]: Ordered values for this parameter axis.
        Assumptions:
            Implementations return non-empty tuples.
        Raises:
            ValueError: If implementation-level materialization cannot produce valid values.
        Side Effects:
            None.
        """
        ...


@dataclass(frozen=True, slots=True)
class ExplicitValuesSpec:
    """
    Explicit materialized values for one parameter axis.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .grid_spec, ..entities.axis_def
    """

    name: str
    values: tuple[GridValue, ...]

    def __post_init__(self) -> None:
        """
        Validate explicit values.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Explicit values are already in the desired deterministic order.
        Raises:
            ValueError: If name is blank, values are empty, or values contain duplicates.
        Side Effects:
            Normalizes `name` by stripping spaces.
        """
        normalized_name = self.name.strip()
        object.__setattr__(self, "name", normalized_name)
        if not normalized_name:
            raise ValueError("ExplicitValuesSpec requires a non-empty parameter name")

        if len(self.values) == 0:
            raise ValueError("ExplicitValuesSpec requires at least one value")
        if len(set(self.values)) != len(self.values):
            raise ValueError("ExplicitValuesSpec values must be unique")

    def materialize(self) -> tuple[GridValue, ...]:
        """
        Return explicit values as-is.

        Args:
            None.
        Returns:
            tuple[GridValue, ...]: Explicit deterministic value set.
        Assumptions:
            Values were validated during object construction.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.values


@dataclass(frozen=True, slots=True)
class RangeValuesSpec:
    """
    Inclusive range-based materialization for a numeric parameter axis.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .grid_spec, ..entities.param_def
    """

    name: str
    start: int | float
    stop_inclusive: int | float
    step: int | float

    def __post_init__(self) -> None:
        """
        Validate range specification invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Range semantics are inclusive on the stop value.
        Raises:
            ValueError: If name is blank, step is non-positive, or start is greater than stop.
        Side Effects:
            Normalizes `name` by stripping spaces.
        """
        normalized_name = self.name.strip()
        object.__setattr__(self, "name", normalized_name)
        if not normalized_name:
            raise ValueError("RangeValuesSpec requires a non-empty parameter name")
        if self.step <= 0:
            raise ValueError("RangeValuesSpec requires step > 0")
        if self.start > self.stop_inclusive:
            raise ValueError("RangeValuesSpec requires start <= stop_inclusive")

    def materialize(self) -> tuple[GridValue, ...]:
        """
        Materialize an inclusive range with deterministic step increments.

        Args:
            None.
        Returns:
            tuple[GridValue, ...]: Inclusive axis values from start to stop.
        Assumptions:
            A small epsilon is acceptable for float stop comparisons.
        Raises:
            ValueError: If generated sequence is empty or exceeds safety iteration bounds.
        Side Effects:
            None.
        """
        values: list[int | float] = []
        current = self.start
        epsilon = abs(self.step) * 1e-9
        while current <= self.stop_inclusive + epsilon:
            values.append(current)
            if len(values) > 1_000_000:
                raise ValueError("RangeValuesSpec generated too many values")
            current = current + self.step

        if len(values) == 0:
            raise ValueError("RangeValuesSpec materialized an empty sequence")
        return tuple(values)
