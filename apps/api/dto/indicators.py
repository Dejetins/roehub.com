"""
Pydantic API models and converters for indicators endpoints.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md,
  docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from trading.contexts.indicators.application.dto import (
    BatchEstimateResult,
    DefaultSpec,
    ExplicitDefaultSpec,
    ExplicitValuesSpec,
    GridParamSpec,
    GridSpec,
    MergedIndicatorView,
    MergedInputView,
    MergedParamView,
    RangeDefaultSpec,
    RangeValuesSpec,
)
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.shared_kernel.primitives import Timeframe, TimeRange, UtcTimestamp


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


class EstimateExplicitAxisSpecRequest(BaseModel):
    """
    Request contract for explicit axis mode in `POST /indicators/estimate`.
    """

    mode: Literal["explicit"]
    values: list[int | float | str]


class EstimateRangeAxisSpecRequest(BaseModel):
    """
    Request contract for inclusive range axis mode in `POST /indicators/estimate`.
    """

    mode: Literal["range"]
    start: int | float
    stop_incl: int | float
    step: int | float


EstimateAxisSpecRequest = Annotated[
    EstimateExplicitAxisSpecRequest | EstimateRangeAxisSpecRequest,
    Field(discriminator="mode"),
]


class EstimateIndicatorRequest(BaseModel):
    """
    One indicator block in `POST /indicators/estimate` request.
    """

    indicator_id: str
    params: dict[str, EstimateAxisSpecRequest] = Field(default_factory=dict)
    source: EstimateAxisSpecRequest | None = None


class EstimateRiskRequest(BaseModel):
    """
    SL/TP block for total variants estimation.
    """

    sl: EstimateAxisSpecRequest
    tp: EstimateAxisSpecRequest


class EstimateTimeRangeRequest(BaseModel):
    """
    Request time-range payload with half-open semantics `[start, end)`.
    """

    start: datetime
    end: datetime


class IndicatorsEstimateRequest(BaseModel):
    """
    API request contract for `POST /indicators/estimate`.
    """

    timeframe: str
    time_range: EstimateTimeRangeRequest
    indicators: list[EstimateIndicatorRequest]
    risk: EstimateRiskRequest


class IndicatorsEstimateResponse(BaseModel):
    """
    Totals-only API response contract for `POST /indicators/estimate`.
    """

    schema_version: int
    total_variants: int
    estimated_memory_bytes: int


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


def build_indicator_grid_specs(
    *,
    request: IndicatorsEstimateRequest,
) -> tuple[GridSpec, ...]:
    """
    Convert estimate request indicator blocks into domain `GridSpec` objects.

    Args:
        request: Parsed API request for `POST /indicators/estimate`.
    Returns:
        tuple[GridSpec, ...]: Indicator grid specs preserving explicit values ordering.
    Assumptions:
        Unknown indicator ids are validated downstream against registry in service layer.
    Raises:
        ValueError: If one of the specs cannot be converted into domain DTOs.
    Side Effects:
        None.
    """
    return tuple(
        _build_grid_spec_from_indicator_request(item=item)
        for item in request.indicators
    )


def build_risk_specs(*, request: IndicatorsEstimateRequest) -> tuple[GridParamSpec, GridParamSpec]:
    """
    Convert API risk block into SL/TP grid parameter specs.

    Args:
        request: Parsed API request for `POST /indicators/estimate`.
    Returns:
        tuple[GridParamSpec, GridParamSpec]: `(sl_spec, tp_spec)` pair.
    Assumptions:
        Risk values are validated as numeric later in application service layer.
    Raises:
        ValueError: If one of risk specs has unsupported shape.
    Side Effects:
        None.
    """
    return (
        _to_grid_param_spec(name="sl", spec=request.risk.sl),
        _to_grid_param_spec(name="tp", spec=request.risk.tp),
    )


def build_time_range(*, request: IndicatorsEstimateRequest) -> TimeRange:
    """
    Convert API request time-range object into shared-kernel `TimeRange`.

    Args:
        request: Parsed API request for `POST /indicators/estimate`.
    Returns:
        TimeRange: Half-open UTC range `[start, end)`.
    Assumptions:
        Input datetimes are timezone-aware or will be rejected by `UtcTimestamp`.
    Raises:
        ValueError: If datetimes are invalid or do not satisfy `start < end`.
    Side Effects:
        None.
    """
    start = UtcTimestamp(request.time_range.start)
    end = UtcTimestamp(request.time_range.end)
    return TimeRange(start=start, end=end)


def build_timeframe(*, request: IndicatorsEstimateRequest) -> Timeframe:
    """
    Convert API request timeframe string into shared-kernel `Timeframe`.

    Args:
        request: Parsed API request for `POST /indicators/estimate`.
    Returns:
        Timeframe: Validated timeframe primitive.
    Assumptions:
        Timeframe code follows shared-kernel allowed set.
    Raises:
        ValueError: If timeframe code is unsupported.
    Side Effects:
        None.
    """
    return Timeframe(request.timeframe)


def build_indicators_estimate_response(
    *,
    result: BatchEstimateResult,
) -> IndicatorsEstimateResponse:
    """
    Convert application batch estimate result into API totals-only response.

    Args:
        result: Application DTO with totals-only estimate values.
    Returns:
        IndicatorsEstimateResponse: Response containing `schema_version`, totals, and nothing else.
    Assumptions:
        `BatchEstimateResult` has already validated schema invariants.
    Raises:
        None.
    Side Effects:
        None.
    """
    return IndicatorsEstimateResponse(
        schema_version=result.schema_version,
        total_variants=result.total_variants,
        estimated_memory_bytes=result.estimated_memory_bytes,
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


def _build_grid_spec_from_indicator_request(*, item: EstimateIndicatorRequest) -> GridSpec:
    """
    Convert one indicator block from API request into domain `GridSpec`.

    Args:
        item: One indicator request block.
    Returns:
        GridSpec: Domain grid specification.
    Assumptions:
        Field-level request validation is already done by Pydantic.
    Raises:
        ValueError: If indicator id is invalid or axis spec conversion fails.
    Side Effects:
        None.
    """
    params: dict[str, GridParamSpec] = {}
    for param_name, spec in item.params.items():
        params[param_name] = _to_grid_param_spec(name=param_name, spec=spec)

    source_spec: GridParamSpec | None = None
    if item.source is not None:
        source_spec = _to_grid_param_spec(name="source", spec=item.source)

    return GridSpec(
        indicator_id=IndicatorId(item.indicator_id),
        params=params,
        source=source_spec,
    )


def _to_grid_param_spec(*, name: str, spec: EstimateAxisSpecRequest) -> GridParamSpec:
    """
    Convert API axis union object into domain grid-parameter specification.

    Args:
        name: Axis name for resulting domain spec.
        spec: API axis spec union member (`explicit` or `range`).
    Returns:
        GridParamSpec: Domain-compatible axis specification.
    Assumptions:
        Caller needs deterministic materialization semantics from domain specs.
    Raises:
        ValueError: If spec member type is unsupported.
    Side Effects:
        None.
    """
    if isinstance(spec, EstimateExplicitAxisSpecRequest):
        return ExplicitValuesSpec(name=name, values=tuple(spec.values))

    if isinstance(spec, EstimateRangeAxisSpecRequest):
        return RangeValuesSpec(
            name=name,
            start=spec.start,
            stop_inclusive=spec.stop_incl,
            step=spec.step,
        )

    raise ValueError(f"unsupported estimate axis spec type: {type(spec).__name__}")
