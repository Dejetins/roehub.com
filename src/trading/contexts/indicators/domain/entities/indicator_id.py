from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndicatorId:
    """
    Stable identifier for an indicator definition.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .indicator_def, ..specifications.grid_spec
    """

    value: str

    def __post_init__(self) -> None:
        """
        Normalize and validate the identifier.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The identifier is provided as text and is stable once created.
        Raises:
            ValueError: If the normalized identifier is empty or contains unsupported symbols.
        Side Effects:
            Normalizes `value` by stripping spaces and converting to lowercase.
        """
        normalized = self.value.strip().lower()
        object.__setattr__(self, "value", normalized)
        if not normalized:
            raise ValueError("IndicatorId must be non-empty")

        compact = normalized.replace("_", "").replace("-", "").replace(".", "")
        if not compact.isalnum():
            raise ValueError(
                "IndicatorId may contain only letters, digits, underscore, dash, and dot"
            )

    def __str__(self) -> str:
        """
        Return the canonical text representation.

        Args:
            None.
        Returns:
            str: Canonical indicator identifier.
        Assumptions:
            `value` has already passed normalization.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.value
