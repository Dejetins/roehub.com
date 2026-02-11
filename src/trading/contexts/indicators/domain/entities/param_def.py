from __future__ import annotations

from dataclasses import dataclass

from .param_kind import ParamKind


@dataclass(frozen=True, slots=True)
class ParamDef:
    """
    Domain declaration of a single indicator parameter.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .param_kind, ..specifications.grid_param_spec
    """

    name: str
    kind: ParamKind
    hard_min: float | int | None = None
    hard_max: float | int | None = None
    step: float | int | None = None
    enum_values: tuple[str, ...] | None = None
    default: float | int | str | None = None

    def __post_init__(self) -> None:
        """
        Validate parameter definition invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Enum and numeric parameter families are mutually exclusive.
        Raises:
            ValueError: If name is invalid or kind-specific invariants are violated.
        Side Effects:
            Normalizes `name` and enum values by stripping spaces.
        """
        normalized_name = self.name.strip()
        object.__setattr__(self, "name", normalized_name)
        if not normalized_name:
            raise ValueError("ParamDef requires a non-empty name")

        if (
            self.hard_min is not None
            and self.hard_max is not None
            and self.hard_min > self.hard_max
        ):
            raise ValueError("ParamDef requires hard_min <= hard_max")

        if self.kind is ParamKind.ENUM:
            self._validate_enum_param()
            return

        self._validate_numeric_param()

    def _validate_numeric_param(self) -> None:
        """
        Validate numeric parameter invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Numeric kinds require positive step and do not accept enum values.
        Raises:
            ValueError: If numeric constraints are violated.
        Side Effects:
            None.
        """
        if self.enum_values is not None:
            raise ValueError("Numeric ParamDef must not define enum_values")
        if self.step is None:
            raise ValueError("Numeric ParamDef requires step")
        if self.step <= 0:
            raise ValueError("Numeric ParamDef requires step > 0")

        if self.default is None:
            return

        if self.hard_min is not None and self.default < self.hard_min:
            raise ValueError("ParamDef default must be >= hard_min")
        if self.hard_max is not None and self.default > self.hard_max:
            raise ValueError("ParamDef default must be <= hard_max")

    def _validate_enum_param(self) -> None:
        """
        Validate enum parameter invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Enum kind disallows numeric bounds and requires non-empty enum values.
        Raises:
            ValueError: If enum constraints are violated.
        Side Effects:
            Normalizes enum values by stripping spaces.
        """
        if self.hard_min is not None or self.hard_max is not None or self.step is not None:
            raise ValueError("Enum ParamDef does not allow hard_min, hard_max, or step")

        if self.enum_values is None or len(self.enum_values) == 0:
            raise ValueError("Enum ParamDef requires non-empty enum_values")

        normalized: list[str] = []
        for raw in self.enum_values:
            value = raw.strip()
            if not value:
                raise ValueError("Enum ParamDef values must be non-empty strings")
            normalized.append(value)

        if len(set(normalized)) != len(normalized):
            raise ValueError("Enum ParamDef values must be unique")

        if self.default is not None and self.default not in normalized:
            raise ValueError("Enum ParamDef default must belong to enum_values")

        object.__setattr__(self, "enum_values", tuple(normalized))
