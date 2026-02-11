"""
Pydantic response models for GET /indicators.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from trading.contexts.indicators.application.dto import (
    DefaultSpec,
    ExplicitDefaultSpec,
    MergedIndicatorView,
    MergedInputView,
    MergedParamView,
    RangeDefaultSpec,
)


class ExplicitDefaultSpecResponse(BaseModel):
    """
    API representation for explicit defaults mode.
    """

    mode: Literal["explicit"]
    values: list[int | float | str]


class RangeDefaultSpecResponse(BaseModel):
    """
    API representation for range defaults mode.
    """

    mode: Literal["range"]
    start: int | float
    stop_incl: int | float
    step: int | float


DefaultSpecResponse = ExplicitDefaultSpecResponse | RangeDefaultSpecResponse


class InputAxisResponse(BaseModel):
    """
    API representation for one configurable input axis.
    """

    name: str
    allowed_values: list[str]
    default: DefaultSpecResponse | None


class ParamAxisResponse(BaseModel):
    """
    API representation for one indicator parameter.
    """

    name: str
    kind: str
    hard_min: int | float | None
    hard_max: int | float | None
    step: int | float | None
    enum_values: list[str] | None
    default: DefaultSpecResponse | None


class IndicatorResponse(BaseModel):
    """
    API representation for one merged indicator definition.
    """

    indicator_id: str
    group: str
    title: str
    required_inputs: list[str]
    inputs: list[InputAxisResponse]
    params: list[ParamAxisResponse]
    output_names: list[str]
    default_layout: str


class IndicatorsResponse(BaseModel):
    """
    API response wrapper for merged indicator registry view.
    """

    schema_version: int
    items: list[IndicatorResponse]


def build_indicators_response(
    *,
    views: tuple[MergedIndicatorView, ...],
) -> IndicatorsResponse:
    """
    Convert merged registry views to API response model.

    Args:
        views: Deterministic merged registry items.
    Returns:
        IndicatorsResponse: Serialized API response payload.
    Assumptions:
        Views are already sorted and validated by registry adapter.
    Raises:
        None.
    Side Effects:
        None.
    """
    return IndicatorsResponse(
        schema_version=1,
        items=[_to_indicator_response(view=view) for view in views],
    )


def _to_indicator_response(*, view: MergedIndicatorView) -> IndicatorResponse:
    """
    Convert one merged indicator view to API response item.

    Args:
        view: One merged indicator view.
    Returns:
        IndicatorResponse: API-compatible response item.
    Assumptions:
        Nested defaults are valid by adapter validation.
    Raises:
        None.
    Side Effects:
        None.
    """
    return IndicatorResponse(
        indicator_id=view.indicator_id,
        group=view.group,
        title=view.title,
        required_inputs=list(view.required_inputs),
        inputs=[_to_input_axis_response(item=item) for item in view.inputs],
        params=[_to_param_axis_response(item=item) for item in view.params],
        output_names=list(view.output_names),
        default_layout=view.default_layout.value,
    )


def _to_input_axis_response(*, item: MergedInputView) -> InputAxisResponse:
    """
    Convert one merged input axis to API response model.

    Args:
        item: Merged input axis view.
    Returns:
        InputAxisResponse: API-compatible input axis object.
    Assumptions:
        Allowed values are deterministic.
    Raises:
        None.
    Side Effects:
        None.
    """
    return InputAxisResponse(
        name=item.name,
        allowed_values=list(item.allowed_values),
        default=_to_default_spec_response(default=item.default),
    )


def _to_param_axis_response(*, item: MergedParamView) -> ParamAxisResponse:
    """
    Convert one merged parameter axis to API response model.

    Args:
        item: Merged parameter axis view.
    Returns:
        ParamAxisResponse: API-compatible parameter axis object.
    Assumptions:
        Param kind is a domain enum with `.value` serialization.
    Raises:
        None.
    Side Effects:
        None.
    """
    return ParamAxisResponse(
        name=item.name,
        kind=item.kind.value,
        hard_min=item.hard_min,
        hard_max=item.hard_max,
        step=item.step,
        enum_values=list(item.enum_values) if item.enum_values is not None else None,
        default=_to_default_spec_response(default=item.default),
    )


def _to_default_spec_response(
    *,
    default: DefaultSpec | None,
) -> DefaultSpecResponse | None:
    """
    Convert internal defaults spec DTO to API response union.

    Args:
        default: Internal default spec or None.
    Returns:
        DefaultSpecResponse | None: Pydantic union member or None.
    Assumptions:
        Union contains only explicit/range variants.
    Raises:
        None.
    Side Effects:
        None.
    """
    if default is None:
        return None

    if isinstance(default, ExplicitDefaultSpec):
        return ExplicitDefaultSpecResponse(mode="explicit", values=list(default.values))

    if isinstance(default, RangeDefaultSpec):
        return RangeDefaultSpecResponse(
            mode="range",
            start=default.start,
            stop_incl=default.stop_incl,
            step=default.step,
        )

    raise ValueError(f"unsupported default spec type: {type(default).__name__}")
