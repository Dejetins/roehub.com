"""
Indicators API routes.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md,
  docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from apps.api.dto import (
    IndicatorsComputeRequest,
    IndicatorsComputeResponse,
    IndicatorsEstimateRequest,
    IndicatorsEstimateResponse,
    IndicatorsResponse,
    build_compute_grid_spec,
    build_compute_market_id,
    build_compute_request,
    build_compute_symbol,
    build_compute_time_range,
    build_compute_timeframe,
    build_indicator_grid_specs,
    build_indicators_compute_response,
    build_indicators_estimate_response,
    build_indicators_response,
    build_risk_specs,
    build_time_range,
    build_timeframe,
)
from trading.contexts.indicators.application.dto import BatchEstimateResult, ExplicitValuesSpec
from trading.contexts.indicators.application.errors import (
    EstimateMemoryGuardExceeded,
    EstimateVariantsGuardExceeded,
)
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.ports.feeds import CandleFeed
from trading.contexts.indicators.application.ports.registry import IndicatorRegistry
from trading.contexts.indicators.application.services import (
    MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    MAX_VARIANTS_PER_COMPUTE_DEFAULT,
    BatchEstimator,
    GridBuilder,
    enforce_batch_guards,
)
from trading.contexts.indicators.domain.errors import (
    ComputeBudgetExceeded,
    GridValidationError,
    UnknownIndicatorError,
)
from trading.shared_kernel.primitives import Timeframe

_TIMEFRAME_1M = Timeframe("1m")


def build_indicators_router(
    *,
    registry: IndicatorRegistry,
    compute: IndicatorCompute | None = None,
    candle_feed: CandleFeed | None = None,
    max_variants_per_compute: int = MAX_VARIANTS_PER_COMPUTE_DEFAULT,
    max_compute_bytes_total: int = MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
) -> APIRouter:
    """
    Build router exposing indicators registry and estimate endpoints.

    Docs: docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related: apps.api.dto.indicators,
      trading.contexts.indicators.application.services.grid_builder

    Args:
        registry: Indicator registry application port implementation.
        compute: Optional compute adapter for `POST /indicators/compute`.
        candle_feed: Optional candle feed adapter for `POST /indicators/compute`.
        max_variants_per_compute: Public variants guard for estimate/compute endpoints.
        max_compute_bytes_total: Public memory guard for estimate/compute endpoints.
    Returns:
        APIRouter: Router exposing indicators registry, estimate, and compute endpoints.
    Assumptions:
        Registry and guard settings are initialized in composition root.
    Raises:
        ValueError: If guards are configured with non-positive values.
    Side Effects:
        None.
    """
    if max_variants_per_compute <= 0:
        raise ValueError(
            "max_variants_per_compute must be > 0, "
            f"got {max_variants_per_compute}"
        )
    if max_compute_bytes_total <= 0:
        raise ValueError(
            "max_compute_bytes_total must be > 0, "
            f"got {max_compute_bytes_total}"
        )

    grid_builder = GridBuilder(registry=registry)
    estimator = BatchEstimator(grid_builder=grid_builder)
    router = APIRouter(tags=["indicators"])

    @router.get("/indicators", response_model=IndicatorsResponse)
    def get_indicators() -> IndicatorsResponse:
        """
        Return merged registry view (hard defs + defaults).

        Args:
            None.
        Returns:
            IndicatorsResponse: Deterministic merged registry payload.
        Assumptions:
            Registry order is deterministic and defaults already validated.
        Raises:
            None.
        Side Effects:
            None.
        """
        views = registry.list_merged()
        return build_indicators_response(views=views)

    @router.post("/indicators/estimate", response_model=IndicatorsEstimateResponse)
    def post_indicators_estimate(request: IndicatorsEstimateRequest) -> IndicatorsEstimateResponse:
        """
        Estimate total variants and memory usage for full indicators batch preflight.

        Args:
            request: API payload with indicators grids, risk SL/TP axes, and time settings.
        Returns:
            IndicatorsEstimateResponse: Totals-only payload with schema version
                and guard-safe values.
        Assumptions:
            Endpoint must not return axis previews or materialized value lists.
        Raises:
            HTTPException: With 422 payload for grid/guard violations.
        Side Effects:
            None.
        """
        try:
            indicator_grids = build_indicator_grid_specs(request=request)
            sl_spec, tp_spec = build_risk_specs(request=request)
            estimate = estimator.estimate_batch(
                indicator_grids=indicator_grids,
                sl_spec=sl_spec,
                tp_spec=tp_spec,
                time_range=build_time_range(request=request),
                timeframe=build_timeframe(request=request),
            )
            enforce_batch_guards(
                estimate=estimate,
                max_variants_per_compute=max_variants_per_compute,
                max_compute_bytes_total=max_compute_bytes_total,
            )
        except (
            EstimateMemoryGuardExceeded,
            EstimateVariantsGuardExceeded,
            GridValidationError,
            UnknownIndicatorError,
            ValueError,
        ) as error:
            raise HTTPException(
                status_code=422,
                detail=_estimate_error_payload(error=error),
            ) from error

        result = BatchEstimateResult(
            schema_version=1,
            total_variants=estimate.total_variants,
            estimated_memory_bytes=estimate.estimated_memory_bytes,
        )
        return build_indicators_estimate_response(result=result)

    @router.post("/indicators/compute", response_model=IndicatorsComputeResponse)
    def post_indicators_compute(
        request: IndicatorsComputeRequest,
        http_request: Request,
    ) -> IndicatorsComputeResponse:
        """
        Compute one indicator tensor using CandleFeed + compute adapter.

        Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
        Related: apps.api.dto.indicators,
          trading.contexts.indicators.application.ports.compute.indicator_compute,
          trading.contexts.indicators.application.ports.feeds.candle_feed

        Args:
            request: API payload with one indicator grid and market/time scope.
        Returns:
            IndicatorsComputeResponse: Compact tensor metadata response.
        Assumptions:
            Endpoint v1 computes exactly one indicator per request.
        Raises:
            HTTPException: With 422 payload for validation/guard errors, 503 if wiring is missing.
        Side Effects:
            Loads dense candles from CandleFeed and executes compute adapter.
        """
        effective_compute = compute or getattr(http_request.app.state, "indicators_compute", None)
        effective_candle_feed = candle_feed or getattr(
            http_request.app.state,
            "indicators_candle_feed",
            None,
        )
        if effective_compute is None or effective_candle_feed is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "compute_not_configured",
                    "message": "POST /indicators/compute is not configured",
                },
            )

        effective_variants_guard = _effective_compute_variants_guard(
            request_guard=request.max_variants_guard,
            max_variants_per_compute=max_variants_per_compute,
        )

        try:
            timeframe = build_compute_timeframe(request=request)
            if timeframe != _TIMEFRAME_1M:
                raise GridValidationError(
                    "POST /indicators/compute supports timeframe='1m' only"
                )
            time_range = build_compute_time_range(request=request)

            grid = build_compute_grid_spec(request=request)

            estimate = effective_compute.estimate(grid=grid, max_variants_guard=2_147_483_647)
            if estimate.variants > max_variants_per_compute:
                raise EstimateVariantsGuardExceeded(
                    total_variants=estimate.variants,
                    max_variants_per_compute=max_variants_per_compute,
                )
            if estimate.variants > effective_variants_guard:
                raise EstimateVariantsGuardExceeded(
                    total_variants=estimate.variants,
                    max_variants_per_compute=effective_variants_guard,
                )

            preflight = estimator.estimate_batch(
                indicator_grids=(grid,),
                sl_spec=ExplicitValuesSpec(name="sl", values=(1.0,)),
                tp_spec=ExplicitValuesSpec(name="tp", values=(1.0,)),
                time_range=time_range,
                timeframe=timeframe,
            )
            enforce_batch_guards(
                estimate=preflight,
                max_variants_per_compute=effective_variants_guard,
                max_compute_bytes_total=max_compute_bytes_total,
            )

            candles = effective_candle_feed.load_1m_dense(
                market_id=build_compute_market_id(request=request),
                symbol=build_compute_symbol(request=request),
                time_range=time_range,
            )
            compute_request = build_compute_request(
                candles=candles,
                request=request,
                max_variants_guard=effective_variants_guard,
            )
            tensor = effective_compute.compute(compute_request)
        except (
            ComputeBudgetExceeded,
            EstimateMemoryGuardExceeded,
            EstimateVariantsGuardExceeded,
            GridValidationError,
            UnknownIndicatorError,
            ValueError,
        ) as error:
            raise HTTPException(
                status_code=422,
                detail=_compute_error_payload(error=error),
            ) from error

        return build_indicators_compute_response(tensor=tensor)

    return router


def _estimate_error_payload(*, error: Exception) -> dict[str, object]:
    """
    Build deterministic `422` payload for estimate endpoint failures.

    Args:
        error: Raised exception from request conversion, grid validation, or guard checks.
    Returns:
        dict[str, object]: Deterministic payload with stable keys.
    Assumptions:
        Guard errors provide structured `details` mapping.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(error, (EstimateVariantsGuardExceeded, EstimateMemoryGuardExceeded)):
        return dict(error.details)
    return {
        "error": "grid_validation_error",
        "message": str(error),
    }


def _compute_error_payload(*, error: Exception) -> dict[str, object]:
    """
    Build deterministic `422` payload for compute endpoint failures.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.dto.indicators,
      trading.contexts.indicators.domain.errors.compute_budget_exceeded

    Args:
        error: Raised exception from request conversion, grid validation, or guard checks.
    Returns:
        dict[str, object]: Deterministic payload with stable keys.
    Assumptions:
        Guard and budget errors provide structured details mappings.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(error, (EstimateVariantsGuardExceeded, EstimateMemoryGuardExceeded)):
        return dict(error.details)
    if isinstance(error, ComputeBudgetExceeded):
        return dict(error.details)
    return {
        "error": "grid_validation_error",
        "message": str(error),
    }


def _effective_compute_variants_guard(
    *,
    request_guard: int | None,
    max_variants_per_compute: int,
) -> int:
    """
    Resolve effective compute variants guard without exceeding router-level maximum.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related: apps.api.dto.indicators,
      trading.contexts.indicators.application.errors.variants_guard_exceeded

    Args:
        request_guard: Optional request-scoped variants guard override.
        max_variants_per_compute: Router-level configured upper bound.
    Returns:
        int: Effective guard value for compute request.
    Assumptions:
        Router-level guard is validated as positive at router construction.
    Raises:
        ValueError: If router-level guard is non-positive.
    Side Effects:
        None.
    """
    if max_variants_per_compute <= 0:
        raise ValueError("max_variants_per_compute must be > 0")
    if request_guard is None:
        return max_variants_per_compute
    return min(request_guard, max_variants_per_compute)
