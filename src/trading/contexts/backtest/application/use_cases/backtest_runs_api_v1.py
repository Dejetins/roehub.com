from __future__ import annotations

from typing import Any, Mapping, Protocol

from trading.contexts.backtest.application.dto import (
    RunBacktestRequest,
    RunBacktestResponse,
)
from trading.contexts.backtest.application.ports import CurrentUser
from trading.contexts.backtest.application.services.run_control_v1 import BacktestRunControlV1
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.backtest_artifacts.application.services.v2.execution_profile_v2 import (
    validate_execution_profile_mode_v2,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, Timeframe

from .backtest_jobs_api_v1 import CreateBacktestJobCommand


class BacktestRunsApiUseCase(Protocol):
    """
    Structural contract for `/backtests` launch orchestration at the API boundary.
    """

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        request_payload: Mapping[str, Any] | None = None,
        run_control: BacktestRunControlV1 | None = None,
    ) -> RunBacktestResponse:
        ...


class BacktestRunPreflightUseCase(Protocol):
    """
    Structural contract for deterministic staged-budget preflight without execution.
    """

    def preflight(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        run_control: BacktestRunControlV1 | None = None,
    ) -> None:
        ...


class BacktestBackgroundJobCreateUseCase(Protocol):
    """
    Structural contract for creating queued persisted runs for background execution.
    """

    def execute(
        self,
        *,
        command: CreateBacktestJobCommand,
        current_user: CurrentUser,
    ) -> BacktestJob:
        ...


class LaunchBacktestGatewayUseCase:
    """
    Thin gateway launch use-case used by production API wiring.
    """

    def __init__(
        self,
        *,
        background_create_use_case: BacktestBackgroundJobCreateUseCase,
        engine_version: str,
        default_execution_profile_mode: str | None = None,
    ) -> None:
        if background_create_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "LaunchBacktestGatewayUseCase requires background_create_use_case"
            )
        normalized_engine_version = engine_version.strip()
        if not normalized_engine_version:
            raise ValueError("LaunchBacktestGatewayUseCase requires engine_version")

        normalized_execution_profile_mode = None
        if default_execution_profile_mode is not None:
            normalized_execution_profile_mode = validate_execution_profile_mode_v2(
                value=default_execution_profile_mode
            )

        self._background_create_use_case = background_create_use_case
        self._engine_version = normalized_engine_version
        self._default_execution_profile_mode = normalized_execution_profile_mode

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        request_payload: Mapping[str, Any] | None = None,
        run_control: BacktestRunControlV1 | None = None,
    ) -> RunBacktestResponse:
        if request is None:  # type: ignore[truthy-bool]
            raise ValueError("LaunchBacktestGatewayUseCase.execute requires request")
        if current_user is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "LaunchBacktestGatewayUseCase.execute requires current_user"
            )
        if request_payload is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "LaunchBacktestGatewayUseCase.execute requires request_payload"
            )
        _ = run_control

        created_run = self._background_create_use_case.execute(
            command=CreateBacktestJobCommand(
                run_request=request,
                request_payload=request_payload,
                execution_mode="background_auto",
                execution_profile_mode=self._default_execution_profile_mode,
            ),
            current_user=current_user,
        )
        return _build_background_auto_launch_response(
            request=request,
            created_run=created_run,
            engine_version=self._engine_version,
        )


def _build_background_auto_launch_response(
    *,
    request: RunBacktestRequest,
    created_run: BacktestJob,
    engine_version: str,
) -> RunBacktestResponse:
    artifact_pin = _require_job_artifact_pin(created_run=created_run)
    if created_run.execution_mode != "background_auto":
        raise BacktestValidationError(
            "background_auto launch response requires background_auto execution_mode"
        )
    return RunBacktestResponse(
        mode=request.mode,
        instrument_id=InstrumentId(
            market_id=MarketId(_require_job_market_id(created_run=created_run)),
            symbol=Symbol(_require_job_symbol(created_run=created_run)),
        ),
        timeframe=Timeframe(_require_job_timeframe(created_run=created_run)),
        strategy_id=request.strategy_id,
        top_k=_require_positive_int_request_json(
            request_json=created_run.request_json,
            field_name="top_k",
        ),
        preselect=_require_positive_int_request_json(
            request_json=created_run.request_json,
            field_name="preselect",
        ),
        variants=tuple(),
        total_indicator_compute_calls=0,
        run_id=created_run.job_id,
        state=created_run.state,
        execution_mode=created_run.execution_mode,
        execution_profile_mode=_require_persisted_execution_profile_mode(
            created_run=created_run
        ),
        engine_version=engine_version,
        artifact_slot=artifact_pin.artifact_slot,
        artifact_slot_generation=artifact_pin.artifact_slot_generation,
        artifact_asof_date=artifact_pin.artifact_asof_date,
        artifact_manifest_hash=artifact_pin.artifact_manifest_hash,
        spec_hash=created_run.spec_hash,
        spec_payload_json=created_run.spec_payload_json,
        engine_params_hash=created_run.engine_params_hash,
    )


def _require_job_artifact_pin(*, created_run: BacktestJob) -> BacktestJobArtifactPin:
    if created_run.artifact_pin is None:
        raise BacktestValidationError("persisted run requires artifact pin metadata")
    return created_run.artifact_pin


def _require_positive_int_request_json(
    *,
    request_json: Mapping[str, Any],
    field_name: str,
) -> int:
    raw_value = request_json.get(field_name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
        raise BacktestValidationError(
            f"persisted run request_json requires positive integer field {field_name!r}"
        )
    return raw_value


def _require_persisted_execution_profile_mode(
    *,
    created_run: BacktestJob,
) -> str:
    raw_mode = created_run.effective_execution_profile_mode
    if raw_mode is None:
        raw_mode = created_run.execution_profile_mode_hint
    if raw_mode is None:
        legacy_mode = created_run.request_json.get("execution_profile_mode")
        raw_mode = legacy_mode if isinstance(legacy_mode, str) else None
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        raise BacktestValidationError(
            "persisted run metadata requires additive execution-profile fields"
        )
    try:
        return validate_execution_profile_mode_v2(value=raw_mode)
    except ValueError as error:
        raise BacktestValidationError(
            "persisted run metadata requires valid additive execution-profile fields"
        ) from error


def _require_job_market_id(*, created_run: BacktestJob) -> int:
    market_id = created_run.market_id
    if market_id is None or market_id <= 0:
        raise BacktestValidationError("persisted run requires positive market_id metadata")
    return market_id


def _require_job_symbol(*, created_run: BacktestJob) -> str:
    symbol = created_run.symbol
    if symbol is None or not symbol.strip():
        raise BacktestValidationError("persisted run requires symbol metadata")
    return symbol


def _require_job_timeframe(*, created_run: BacktestJob) -> str:
    timeframe = created_run.timeframe
    if timeframe is None or not timeframe.strip():
        raise BacktestValidationError("persisted run requires timeframe metadata")
    return timeframe


__all__ = [
    "BacktestBackgroundJobCreateUseCase",
    "BacktestRunPreflightUseCase",
    "BacktestRunsApiUseCase",
    "LaunchBacktestGatewayUseCase",
]
