from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """
    Output declaration for an indicator definition.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .indicator_def, ...application.dto.indicator_tensor
    """

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        """
        Validate output names.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Each output component name is unique and non-empty.
        Raises:
            ValueError: If no names are provided or a name is blank or duplicated.
        Side Effects:
            Normalizes names by stripping surrounding spaces.
        """
        if not self.names:
            raise ValueError("OutputSpec requires at least one output name")

        normalized: list[str] = []
        for name in self.names:
            value = name.strip()
            if not value:
                raise ValueError("OutputSpec names must be non-empty")
            normalized.append(value)

        if len(set(normalized)) != len(normalized):
            raise ValueError("OutputSpec names must be unique")

        object.__setattr__(self, "names", tuple(normalized))
