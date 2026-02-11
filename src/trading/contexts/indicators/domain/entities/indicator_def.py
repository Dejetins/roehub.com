from __future__ import annotations

from dataclasses import dataclass

from .indicator_id import IndicatorId
from .input_series import InputSeries
from .output_spec import OutputSpec
from .param_def import ParamDef

_ALLOWED_SPECIAL_AXES = ("source",)


@dataclass(frozen=True, slots=True)
class IndicatorDef:
    """
    Full domain definition of an indicator and its parameter space.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .param_def, ..specifications.grid_spec
    """

    indicator_id: IndicatorId
    title: str
    inputs: tuple[InputSeries, ...]
    params: tuple[ParamDef, ...]
    axes: tuple[str, ...]
    output: OutputSpec

    def __post_init__(self) -> None:
        """
        Validate indicator definition consistency.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Axis names map to parameter names or explicitly allowed special axes.
        Raises:
            ValueError: If title is blank, lists are inconsistent, or axes are invalid.
        Side Effects:
            Normalizes title and axis names by stripping spaces.
        """
        if self.indicator_id is None:  # type: ignore[truthy-bool]
            raise ValueError("IndicatorDef requires indicator_id")
        if self.output is None:  # type: ignore[truthy-bool]
            raise ValueError("IndicatorDef requires output")

        normalized_title = self.title.strip()
        object.__setattr__(self, "title", normalized_title)
        if not normalized_title:
            raise ValueError("IndicatorDef requires a non-empty title")

        if len(self.inputs) == 0:
            raise ValueError("IndicatorDef requires at least one input series")

        param_names: list[str] = []
        for param in self.params:
            param_names.append(param.name)
        if len(set(param_names)) != len(param_names):
            raise ValueError("IndicatorDef parameter names must be unique")

        normalized_axes: list[str] = []
        for axis_name in self.axes:
            normalized_axis = axis_name.strip()
            if not normalized_axis:
                raise ValueError("IndicatorDef axes must be non-empty names")
            normalized_axes.append(normalized_axis)

        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("IndicatorDef axes must be unique")

        allowed_axes = set(param_names + list(_ALLOWED_SPECIAL_AXES))
        for axis_name in normalized_axes:
            if axis_name not in allowed_axes:
                raise ValueError(
                    f"IndicatorDef axis '{axis_name}' must reference "
                    "a parameter or supported special axis"
                )

        object.__setattr__(self, "axes", tuple(normalized_axes))
