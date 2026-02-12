"""
Indicators API routes.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md,
  docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.dto import (
    IndicatorsEstimateRequest,
    IndicatorsEstimateResponse,
    IndicatorsResponse,
    build_indicator_grid_specs,
    build_indicators_estimate_response,
    build_indicators_response,
    build_risk_specs,
    build_time_range,
    build_timeframe,
)
from trading.contexts.indicators.application.dto import BatchEstimateResult
from trading.contexts.indicators.application.errors import (
    EstimateMemoryGuardExceeded,
    EstimateVariantsGuardExceeded,
)
from trading.contexts.indicators.application.ports.registry import IndicatorRegistry
from trading.contexts.indicators.application.services import (
    MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    MAX_VARIANTS_PER_COMPUTE_DEFAULT,
    BatchEstimator,
    GridBuilder,
    enforce_batch_guards,
)
from trading.contexts.indicators.domain.errors import GridValidationError, UnknownIndicatorError


def build_indicators_router(
    *,
    registry: IndicatorRegistry,
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
        max_variants_per_compute: Public variants guard for `POST /indicators/estimate`.
        max_compute_bytes_total: Public memory guard for `POST /indicators/estimate`.
    Returns:
        APIRouter: Router exposing `GET /indicators` and `POST /indicators/estimate`.
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
