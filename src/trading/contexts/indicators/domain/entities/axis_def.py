from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AxisDef:
    """
    Materialized axis values for one tensor dimension.

    Exactly one value family must be provided.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .indicator_def, ..specifications.grid_param_spec
    """

    name: str
    values_int: tuple[int, ...] | None = None
    values_float: tuple[float, ...] | None = None
    values_enum: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """
        Validate one-of semantics and axis value invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Axis values are immutable tuples and must have at least one value.
        Raises:
            ValueError: If no value family is provided, more than one is
                provided, or values are invalid.
        Side Effects:
            Normalizes `name` and enum values by stripping spaces.
        """
        normalized_name = self.name.strip()
        object.__setattr__(self, "name", normalized_name)
        if not normalized_name:
            raise ValueError("AxisDef requires a non-empty name")

        provided = 0
        for candidate in (self.values_int, self.values_float, self.values_enum):
            if candidate is not None:
                provided += 1
        if provided != 1:
            raise ValueError(
                "AxisDef requires exactly one of values_int, "
                "values_float, values_enum"
            )

        if self.values_int is not None:
            self._validate_numeric_tuple(self.values_int, "values_int")
            return
        if self.values_float is not None:
            self._validate_numeric_tuple(self.values_float, "values_float")
            return
        self._validate_enum_tuple()

    def length(self) -> int:
        """
        Return axis cardinality.

        Args:
            None.
        Returns:
            int: Number of values in the active axis family.
        Assumptions:
            Invariants guarantee that one family is present.
        Raises:
            RuntimeError: If the object somehow violates one-of invariants after construction.
        Side Effects:
            None.
        """
        if self.values_int is not None:
            return len(self.values_int)
        if self.values_float is not None:
            return len(self.values_float)
        if self.values_enum is not None:
            return len(self.values_enum)
        raise RuntimeError("AxisDef is in an invalid state without values")

    def _validate_numeric_tuple(
        self, values: tuple[int, ...] | tuple[float, ...], field: str
    ) -> None:
        """
        Validate non-empty and unique numeric axis values.

        Args:
            values: Axis numeric values.
            field: Field name for error messages.
        Returns:
            None.
        Assumptions:
            Values are hashable and preserve caller-provided deterministic ordering.
        Raises:
            ValueError: If values are empty or duplicated.
        Side Effects:
            None.
        """
        if len(values) == 0:
            raise ValueError(f"AxisDef {field} must be non-empty")
        if len(set(values)) != len(values):
            raise ValueError(f"AxisDef {field} must contain unique values")

    def _validate_enum_tuple(self) -> None:
        """
        Validate non-empty and unique enum axis values.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Enum values are provided as plain strings.
        Raises:
            ValueError: If enum values are empty, blank, or duplicated.
        Side Effects:
            Normalizes enum values by stripping spaces.
        """
        if self.values_enum is None or len(self.values_enum) == 0:
            raise ValueError("AxisDef values_enum must be non-empty")

        normalized: list[str] = []
        for raw in self.values_enum:
            value = raw.strip()
            if not value:
                raise ValueError("AxisDef values_enum must not contain blank values")
            normalized.append(value)

        if len(set(normalized)) != len(normalized):
            raise ValueError("AxisDef values_enum must contain unique values")

        object.__setattr__(self, "values_enum", tuple(normalized))
