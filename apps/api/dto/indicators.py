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
    CandleArrays,
    ComputeRequest,
    DefaultSpec,
    ExplicitDefaultSpec,
    ExplicitValuesSpec,
    GridParamSpec,
    GridSpec,
    IndicatorTensor,
    MergedIndicatorView,
    MergedInputView,
    MergedParamView,
    RangeDefaultSpec,
    RangeValuesSpec,
)
from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId, Layout
from trading.shared_kernel.primitives import (
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
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


class IndicatorsComputeRequest(BaseModel):
    """
    API request contract for `POST /indicators/compute` (one indicator per request).

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.contexts.indicators.application.dto.compute_request
    """

    market_id: int
    symbol: str
    timeframe: str
    time_range: EstimateTimeRangeRequest
    indicator: EstimateIndicatorRequest
    layout: Literal["time_major", "variant_major"] | None = None
    max_variants_guard: int | None = Field(default=None, gt=0)


class ComputeAxisResponse(BaseModel):
    """
    API response axis descriptor for computed tensor metadata.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: trading.contexts.indicators.domain.entities.axis_def,
      trading.contexts.indicators.application.dto.indicator_tensor
    """

    name: str
    values: list[int | float | str]


class ComputeMetaResponse(BaseModel):
    """
    API response metadata for computed tensor.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: trading.contexts.indicators.application.dto.indicator_tensor,
      apps.api.routes.indicators
    """

    t: int
    variants: int
    nan_policy: str
    compute_ms: int | None = None


class IndicatorsComputeResponse(BaseModel):
    """
    Compact API response contract for `POST /indicators/compute`.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.contexts.indicators.application.dto.indicator_tensor
    """

    schema_version: int
    indicator_id: str
    layout: str
    axes: list[ComputeAxisResponse]
    shape: list[int]
    dtype: str
    c_contiguous: bool
    meta: ComputeMetaResponse


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


def build_compute_time_range(*, request: IndicatorsComputeRequest) -> TimeRange:
    """
    Convert compute request time-range into shared-kernel `TimeRange`.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.shared_kernel.primitives.time_range

    Args:
        request: Parsed API request for `POST /indicators/compute`.
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


def build_compute_timeframe(*, request: IndicatorsComputeRequest) -> Timeframe:
    """
    Convert compute request timeframe string into shared-kernel `Timeframe`.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.shared_kernel.primitives.timeframe

    Args:
        request: Parsed API request for `POST /indicators/compute`.
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


def build_compute_market_id(*, request: IndicatorsComputeRequest) -> MarketId:
    """
    Convert compute request market id into shared-kernel `MarketId`.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.shared_kernel.primitives.market_id

    Args:
        request: Parsed API request for `POST /indicators/compute`.
    Returns:
        MarketId: Validated market identifier value object.
    Assumptions:
        `market_id` is integer-like and positive.
    Raises:
        ValueError: If market id violates shared-kernel invariants.
    Side Effects:
        None.
    """
    return MarketId(request.market_id)


def build_compute_symbol(*, request: IndicatorsComputeRequest) -> Symbol:
    """
    Convert compute request symbol string into shared-kernel `Symbol`.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.shared_kernel.primitives.symbol

    Args:
        request: Parsed API request for `POST /indicators/compute`.
    Returns:
        Symbol: Normalized symbol value object.
    Assumptions:
        Symbol normalization follows shared-kernel rules.
    Raises:
        ValueError: If symbol is blank after normalization.
    Side Effects:
        None.
    """
    return Symbol(request.symbol)


def build_compute_grid_spec(*, request: IndicatorsComputeRequest) -> GridSpec:
    """
    Convert compute request indicator block into one domain `GridSpec`.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.contexts.indicators.domain.specifications.grid_spec

    Args:
        request: Parsed API request for `POST /indicators/compute`.
    Returns:
        GridSpec: Domain grid specification for one indicator.
    Assumptions:
        Request contains exactly one indicator block in `request.indicator`.
    Raises:
        ValueError: If indicator id is invalid or layout code is unsupported.
    Side Effects:
        None.
    """
    layout_preference: Layout | None = None
    if request.layout is not None:
        layout_preference = Layout(request.layout)
    return _build_grid_spec_from_indicator_request(
        item=request.indicator,
        layout_preference=layout_preference,
    )


def build_compute_request(
    *,
    candles: CandleArrays,
    request: IndicatorsComputeRequest,
    max_variants_guard: int,
) -> ComputeRequest:
    """
    Build application `ComputeRequest` from API compute payload and candles.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.contexts.indicators.application.dto.compute_request

    Args:
        candles: Dense candle arrays loaded via `CandleFeed`.
        request: Parsed API request for `POST /indicators/compute`.
        max_variants_guard: Effective variants guard to enforce during compute.
    Returns:
        ComputeRequest: Application-layer compute request DTO.
    Assumptions:
        Candle arrays are already validated by candle feed adapter.
    Raises:
        ValueError: If DTO invariants are violated.
    Side Effects:
        None.
    """
    return ComputeRequest(
        candles=candles,
        grid=build_compute_grid_spec(request=request),
        max_variants_guard=max_variants_guard,
    )


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


def build_indicators_compute_response(
    *,
    tensor: IndicatorTensor,
) -> IndicatorsComputeResponse:
    """
    Convert application tensor output into compact compute API response.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.contexts.indicators.application.dto.indicator_tensor

    Args:
        tensor: Computed indicator tensor from application compute adapter.
    Returns:
        IndicatorsComputeResponse: Metadata-focused response without raw tensor payload.
    Assumptions:
        Tensor invariants are validated by `IndicatorTensor` dataclass.
    Raises:
        ValueError: If tensor axes contain unsupported value families.
    Side Effects:
        None.
    """
    return IndicatorsComputeResponse(
        schema_version=1,
        indicator_id=tensor.indicator_id.value,
        layout=tensor.layout.value,
        axes=[_to_compute_axis_response(axis=axis) for axis in tensor.axes],
        shape=[int(dim) for dim in tensor.values.shape],
        dtype=str(tensor.values.dtype),
        c_contiguous=bool(tensor.values.flags["C_CONTIGUOUS"]),
        meta=ComputeMetaResponse(
            t=tensor.meta.t,
            variants=tensor.meta.variants,
            nan_policy=tensor.meta.nan_policy,
            compute_ms=tensor.meta.compute_ms,
        ),
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


def _to_compute_axis_response(*, axis: AxisDef) -> ComputeAxisResponse:
    """
    Convert domain `AxisDef` into compact API axis response shape.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: trading.contexts.indicators.domain.entities.axis_def,
      trading.contexts.indicators.application.dto.indicator_tensor

    Args:
        axis: Domain axis definition from computed tensor metadata.
    Returns:
        ComputeAxisResponse: Response axis with one values list.
    Assumptions:
        `AxisDef` enforces one-of value family semantics.
    Raises:
        ValueError: If axis has no active values family.
    Side Effects:
        None.
    """
    if axis.values_int is not None:
        return ComputeAxisResponse(name=axis.name, values=[int(item) for item in axis.values_int])
    if axis.values_float is not None:
        return ComputeAxisResponse(
            name=axis.name,
            values=[float(item) for item in axis.values_float],
        )
    if axis.values_enum is not None:
        return ComputeAxisResponse(name=axis.name, values=[str(item) for item in axis.values_enum])
    raise ValueError(f"axis '{axis.name}' has no active values family")


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


def _build_grid_spec_from_indicator_request(
    *,
    item: EstimateIndicatorRequest,
    layout_preference: Layout | None = None,
) -> GridSpec:
    """
    Convert one indicator block from API request into domain `GridSpec`.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.routes.indicators,
      trading.contexts.indicators.domain.specifications.grid_spec

    Args:
        item: One indicator request block.
        layout_preference: Optional explicit tensor layout preference.
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
        layout_preference=layout_preference,
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
