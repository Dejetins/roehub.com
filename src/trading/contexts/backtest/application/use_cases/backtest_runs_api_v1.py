from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol, cast
from uuid import UUID, uuid4

from trading.contexts.backtest.application.dto import (
    BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1,
    BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1,
    BacktestReportV1,
    RunBacktestRequest,
    RunBacktestResponse,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository, CurrentUser
from trading.contexts.backtest.application.services import (
    BacktestJobTopVariantCandidateV1,
    build_finalized_snapshot_rows,
)
from trading.contexts.backtest.application.services.run_control_v1 import BacktestRunControlV1
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobMode,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError

from .backtest_jobs_api_v1 import _build_sha256_from_payload, _normalize_json_mapping

NowProvider = Callable[[], datetime]
RunIdFactory = Callable[[], UUID]
_SYNC_INLINE_LOCKED_BY = "sync-inline"


class BacktestRunsApiUseCase(Protocol):
    """
    Structural contract for `/backtests` sync execution and lazy report API orchestration.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - apps/api/routes/backtests.py
    """

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        request_payload: Mapping[str, Any] | None = None,
        run_control: BacktestRunControlV1 | None = None,
    ) -> RunBacktestResponse:
        """
        Execute one sync backtest flow with optional canonical request payload snapshot.

        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            request_payload: Optional strict API payload snapshot for persisted-run orchestrators.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            RunBacktestResponse: Deterministic sync response DTO.
        Assumptions:
            Implementations may ignore `request_payload` when persistence is not required.
        Raises:
            Exception: Domain/application errors are implementation-specific.
        Side Effects:
            May execute sync compute and optionally persist run metadata.
        """
        ...

    def build_variant_report(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        variant_payload: Any,
        include_trades: bool = False,
        run_control: BacktestRunControlV1 | None = None,
    ) -> BacktestReportV1:
        """
        Build one lazy single-variant report for `/backtests/variant-report`.

        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            variant_payload: Explicit variant payload selected for on-demand detail generation.
            include_trades: Include-trades flag for report generation.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            BacktestReportV1: Deterministic report payload.
        Assumptions:
            Report generation reuses the same runtime semantics as sync execution.
        Raises:
            Exception: Domain/application errors are implementation-specific.
        Side Effects:
            Executes report generation logic and may perform compute IO.
        """
        ...


class CreateAndRunBacktestSyncInlineUseCase:
    """
    Create-and-execute persisted sync-inline Backtest flow over the unified jobs storage family.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - apps/api/routes/backtests.py
    """

    def __init__(
        self,
        *,
        run_use_case: BacktestRunsApiUseCase,
        job_repository: BacktestJobRepository,
        backtest_runtime_config_hash: str,
        engine_version: str,
        now_provider: NowProvider | None = None,
        run_id_factory: RunIdFactory | None = None,
    ) -> None:
        """
        Initialize sync-inline persisted-run orchestrator dependencies.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/wiring/modules/backtest.py
        Args:
            run_use_case: Existing sync backtest use-case performing internal preflight + execute.
            job_repository: Persisted-run repository over the unified jobs storage family.
            backtest_runtime_config_hash: Canonical result-affecting runtime config hash.
            engine_version: Stable engine/runtime literal exposed in sync responses.
            now_provider: Optional UTC clock provider for deterministic tests.
            run_id_factory: Optional persisted run id factory for deterministic tests.
        Returns:
            None.
        Assumptions:
            Runtime config hash and engine-version literal are validated on startup wiring.
        Raises:
            ValueError: If one dependency or invariant is invalid.
        Side Effects:
            None.
        """
        if run_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateAndRunBacktestSyncInlineUseCase requires run_use_case")
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateAndRunBacktestSyncInlineUseCase requires job_repository")
        normalized_engine_version = engine_version.strip()
        if not normalized_engine_version:
            raise ValueError("CreateAndRunBacktestSyncInlineUseCase requires engine_version")
        normalized_runtime_hash = backtest_runtime_config_hash.strip().lower()
        if len(normalized_runtime_hash) != 64:
            raise ValueError(
                "CreateAndRunBacktestSyncInlineUseCase requires 64-char "
                "backtest_runtime_config_hash"
            )

        self._run_use_case = run_use_case
        self._job_repository = job_repository
        self._backtest_runtime_config_hash = normalized_runtime_hash
        self._engine_version = normalized_engine_version
        self._now = now_provider or _utc_now
        self._run_id_factory = run_id_factory or uuid4

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        request_payload: Mapping[str, Any] | None = None,
        run_control: BacktestRunControlV1 | None = None,
    ) -> RunBacktestResponse:
        """
        Execute sync run inline, then persist final run row and summary-only top rows atomically.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
          - apps/api/routes/backtests.py
        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            request_payload: Strict API payload snapshot used to persist canonical `request_json`.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            RunBacktestResponse: Sync response enriched with persisted run identity metadata.
        Assumptions:
            Internal preflight remains delegated to the existing sync `RunBacktestUseCase`.
        Raises:
            BacktestValidationError: If persisted metadata cannot be built deterministically.
            RoehubError: Propagates canonical validation/not-found/forbidden failures
                from the inner sync use-case.
        Side Effects:
            Executes sync backtest compute and writes one terminal run row plus summary-only top
            rows into the unified Postgres storage family.
        """
        if request is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateAndRunBacktestSyncInlineUseCase.execute requires request")
        if current_user is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "CreateAndRunBacktestSyncInlineUseCase.execute requires current_user"
            )
        if request_payload is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "CreateAndRunBacktestSyncInlineUseCase.execute requires request_payload"
            )

        created_at = self._now()
        base_response = self._run_use_case.execute(
            request=request,
            current_user=current_user,
            run_control=run_control,
        )
        finished_at = self._now()

        artifact_pin = _artifact_pin_from_response(response=base_response)
        request_json = _build_request_json_payload(
            request=request,
            request_payload=request_payload,
            response=base_response,
        )
        engine_params_hash = _build_engine_params_hash(response=base_response)
        run = _build_terminal_sync_inline_run(
            run_id=self._run_id_factory(),
            created_at=created_at,
            finished_at=finished_at,
            current_user=current_user,
            request=request,
            response=base_response,
            request_json=request_json,
            artifact_pin=artifact_pin,
            backtest_runtime_config_hash=self._backtest_runtime_config_hash,
            engine_params_hash=engine_params_hash,
        )
        persisted_rows = _build_persisted_top_rows(
            job_id=run.job_id,
            persisted_at=finished_at,
            response=base_response,
        )
        persisted_run = self._job_repository.create_with_top_variants(
            job=run,
            top_variants=persisted_rows,
        )

        return replace(
            base_response,
            run_id=persisted_run.job_id,
            state=persisted_run.state,
            execution_mode=persisted_run.execution_mode,
            engine_version=self._engine_version,
            artifact_slot=artifact_pin.artifact_slot,
            artifact_slot_generation=artifact_pin.artifact_slot_generation,
            artifact_asof_date=artifact_pin.artifact_asof_date,
            artifact_manifest_hash=artifact_pin.artifact_manifest_hash,
            spec_hash=persisted_run.spec_hash,
            spec_payload_json=persisted_run.spec_payload_json,
            engine_params_hash=persisted_run.engine_params_hash,
        )

    def build_variant_report(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        variant_payload: Any,
        include_trades: bool = False,
        run_control: BacktestRunControlV1 | None = None,
    ) -> BacktestReportV1:
        """
        Delegate lazy single-variant report generation to the existing sync use-case.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/routes/backtests.py
        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            variant_payload: Explicit variant payload selected for lazy detail/report fetch.
            include_trades: Include-trades flag for report generation.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            BacktestReportV1: Deterministic lazy report payload.
        Assumptions:
            Persisted sync-inline cutover does not change lazy single-variant detail semantics.
        Raises:
            RoehubError: Propagates canonical errors from the inner sync use-case.
        Side Effects:
            Delegates to the existing report builder path without touching persisted storage.
        """
        return self._run_use_case.build_variant_report(
            request=request,
            current_user=current_user,
            variant_payload=variant_payload,
            include_trades=include_trades,
            run_control=run_control,
        )


def _artifact_pin_from_response(*, response: RunBacktestResponse) -> BacktestJobArtifactPin:
    """
    Convert sync run artifact metadata into the immutable persisted-run pin DTO.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        response: Completed sync response carrying slot-pinned artifact metadata.
    Returns:
        BacktestJobArtifactPin: Immutable artifact pin reused for persistence and API response.
    Assumptions:
        The inner sync run already executed against this exact pinned artifact identity.
    Raises:
        BacktestValidationError: If one artifact metadata field is missing from the sync response.
    Side Effects:
        None.
    """
    if (
        response.artifact_slot is None
        or response.artifact_slot_generation is None
        or response.artifact_asof_date is None
        or response.artifact_manifest_hash is None
    ):
        raise BacktestValidationError(
            "sync_inline persisted run requires slot-pinned artifact metadata"
        )
    return BacktestJobArtifactPin(
        artifact_slot=response.artifact_slot,
        artifact_slot_generation=response.artifact_slot_generation,
        artifact_manifest_hash=response.artifact_manifest_hash,
        artifact_asof_date=response.artifact_asof_date,
    )


def _build_request_json_payload(
    *,
    request: RunBacktestRequest,
    request_payload: Mapping[str, Any],
    response: RunBacktestResponse,
) -> Mapping[str, Any]:
    """
    Build canonical persisted `request_json` payload for one completed sync-inline run.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        request: Parsed application request DTO.
        request_payload: Strict API request payload snapshot.
        response: Completed sync response carrying resolved runtime defaults.
    Returns:
        Mapping[str, Any]: Deterministic JSON-compatible request snapshot for persistence.
    Assumptions:
        Sync-inline persistence keeps the same canonical request shape as the jobs storage family.
    Raises:
        BacktestValidationError: If template/saved mode payload cannot be reconstructed.
    Side Effects:
        None.
    """
    normalized_payload = _normalize_json_mapping(values=request_payload)
    normalized_payload["warmup_bars"] = response.warmup_bars
    normalized_payload["top_k"] = response.top_k
    normalized_payload["preselect"] = response.preselect
    normalized_payload["top_trades_n"] = response.top_trades_n

    direction_mode = _require_direction_mode(response=response)
    sizing_mode = _require_sizing_mode(response=response)
    execution_payload = _sorted_execution_payload(response=response)

    if request.mode == "template":
        raw_template = normalized_payload.get("template")
        if not isinstance(raw_template, Mapping):
            raise BacktestValidationError("template mode sync_inline persistence requires template")
        template_payload = dict(raw_template)
        template_payload["direction_mode"] = direction_mode
        template_payload["sizing_mode"] = sizing_mode
        template_payload["execution"] = execution_payload
        normalized_payload["template"] = template_payload
        normalized_payload.pop("overrides", None)
        return normalized_payload

    overrides_payload: dict[str, Any] = {}
    raw_overrides = normalized_payload.get("overrides")
    if isinstance(raw_overrides, Mapping):
        overrides_payload = dict(raw_overrides)
    overrides_payload["direction_mode"] = direction_mode
    overrides_payload["sizing_mode"] = sizing_mode
    overrides_payload["execution"] = execution_payload
    normalized_payload["overrides"] = overrides_payload
    if request.strategy_id is not None:
        normalized_payload["strategy_id"] = str(request.strategy_id)
    normalized_payload.pop("template", None)
    return normalized_payload


def _build_engine_params_hash(*, response: RunBacktestResponse) -> str:
    """
    Build deterministic engine-params hash from resolved sync execution payload.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/dto/backtests.py
    Args:
        response: Completed sync response with resolved execution metadata.
    Returns:
        str: Canonical SHA-256 hash of effective direction/sizing/execution settings.
    Assumptions:
        Response carries the same effective execution payload that was used during the sync run.
    Raises:
        BacktestValidationError: If response lacks effective execution metadata.
    Side Effects:
        None.
    """
    return _build_sha256_from_payload(
        payload={
            "direction_mode": _require_direction_mode(response=response),
            "sizing_mode": _require_sizing_mode(response=response),
            "execution": _sorted_execution_payload(response=response),
        }
    )


def _build_terminal_sync_inline_run(
    *,
    run_id: UUID,
    created_at: datetime,
    finished_at: datetime,
    current_user: CurrentUser,
    request: RunBacktestRequest,
    response: RunBacktestResponse,
    request_json: Mapping[str, Any],
    artifact_pin: BacktestJobArtifactPin,
    backtest_runtime_config_hash: str,
    engine_params_hash: str,
) -> BacktestJob:
    """
    Build final succeeded sync-inline run aggregate snapshot for one persisted sync response.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
    Args:
        run_id: Persisted run identifier.
        created_at: Sync run create timestamp captured before compute started.
        finished_at: Terminal timestamp captured after sync compute completed.
        current_user: Authenticated owner identity.
        request: Parsed application request DTO.
        response: Completed sync response carrying resolved runtime metadata.
        request_json: Canonical request snapshot payload for storage.
        artifact_pin: Immutable artifact pin used by the sync execution.
        backtest_runtime_config_hash: Canonical result-affecting runtime hash.
        engine_params_hash: Canonical effective execution hash.
    Returns:
        BacktestJob: Terminal succeeded run aggregate ready for persistence.
    Assumptions:
        Sync-inline persistence records one logical attempt represented by
        `queued -> running -> succeeded` domain transitions in memory.
    Raises:
        BacktestValidationError: If saved-mode response lacks required persisted spec metadata.
    Side Effects:
        None.
    """
    ranking_primary_metric = (
        request.ranking.primary_metric
        if request.ranking is not None
        else BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1
    )
    ranking_secondary_metric = (
        request.ranking.secondary_metric
        if request.ranking is not None
        else BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1
    )

    spec_hash = response.spec_hash if request.mode == "saved" else None
    spec_payload_json = response.spec_payload_json if request.mode == "saved" else None
    if request.mode == "saved" and (spec_hash is None or spec_payload_json is None):
        raise BacktestValidationError(
            "saved mode sync_inline persistence requires spec snapshot metadata"
        )

    queued = BacktestJob.create_queued(
        job_id=run_id,
        user_id=current_user.user_id,
        mode=cast(BacktestJobMode, request.mode),
        created_at=created_at,
        request_json=request_json,
        request_hash=_build_sha256_from_payload(payload=request_json),
        spec_hash=spec_hash,
        spec_payload_json=spec_payload_json,
        engine_params_hash=engine_params_hash,
        backtest_runtime_config_hash=backtest_runtime_config_hash,
        artifact_pin=artifact_pin,
        execution_mode="sync_inline",
        market_id=response.instrument_id.market_id.value,
        symbol=str(response.instrument_id.symbol),
        timeframe=str(response.timeframe),
        requested_top_n=response.top_k,
        ranking_primary_metric=ranking_primary_metric,
        ranking_secondary_metric=ranking_secondary_metric,
    )
    claimed = queued.claim(
        changed_at=created_at,
        locked_by=_SYNC_INLINE_LOCKED_BY,
        lease_expires_at=created_at + timedelta(seconds=1),
    )
    return claimed.finish(next_state="succeeded", changed_at=finished_at)


def _build_persisted_top_rows(
    *,
    job_id: UUID,
    persisted_at: datetime,
    response: RunBacktestResponse,
) -> tuple[BacktestJobTopVariant, ...]:
    """
    Convert sync response variants into summary-only persisted top rows via shared worker mappers.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
    Args:
        job_id: Persisted run identifier.
        persisted_at: Final persistence timestamp in UTC.
        response: Completed sync response with ranked variants.
    Returns:
        tuple[BacktestJobTopVariant, ...]: Summary-only persisted top rows ordered by rank.
    Assumptions:
        Response variants are already ordered by the final deterministic ranking contract.
    Raises:
        BacktestValidationError: If one ranked variant lacks explicit payload data.
    Side Effects:
        None.
    """
    candidates: list[BacktestJobTopVariantCandidateV1] = []
    for variant in response.variants:
        if variant.payload is None:  # pragma: no cover - guarded by DTO invariant
            raise BacktestValidationError("sync_inline persisted top rows require variant payload")
        candidates.append(
            BacktestJobTopVariantCandidateV1(
                variant_index=variant.variant_index,
                variant_key=variant.variant_key,
                indicator_variant_key=variant.indicator_variant_key,
                total_return_pct=variant.total_return_pct,
                indicator_selections=variant.payload.indicator_selections,
                signal_params=cast(
                    Mapping[str, Mapping[str, Any]],
                    variant.payload.signal_params or {},
                ),
                risk_params=cast(Mapping[str, Any], variant.payload.risk_params or {}),
                summary_metrics_json=variant.summary_metrics_json,
                best_tp_pct=variant.best_tp_pct,
                best_sl_pct=variant.best_sl_pct,
            )
        )
    return build_finalized_snapshot_rows(
        job_id=job_id,
        now=persisted_at,
        ranked_candidates=tuple(candidates),
        direction_mode=_require_direction_mode(response=response),
        sizing_mode=_require_sizing_mode(response=response),
        execution_params=_sorted_execution_payload(response=response),
        reports_by_variant_key={},
        trades_by_variant_key={},
    )


def _require_direction_mode(*, response: RunBacktestResponse) -> str:
    """
    Require deterministic effective direction mode on the completed sync response.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    Args:
        response: Completed sync response.
    Returns:
        str: Effective direction mode literal.
    Assumptions:
        `RunBacktestUseCase` always resolves and returns one effective direction mode.
    Raises:
        BacktestValidationError: If direction mode metadata is unexpectedly absent.
    Side Effects:
        None.
    """
    if response.direction_mode is None:
        raise BacktestValidationError("sync_inline persisted run requires direction_mode")
    return response.direction_mode


def _require_sizing_mode(*, response: RunBacktestResponse) -> str:
    """
    Require deterministic effective sizing mode on the completed sync response.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    Args:
        response: Completed sync response.
    Returns:
        str: Effective sizing mode literal.
    Assumptions:
        `RunBacktestUseCase` always resolves and returns one effective sizing mode.
    Raises:
        BacktestValidationError: If sizing mode metadata is unexpectedly absent.
    Side Effects:
        None.
    """
    if response.sizing_mode is None:
        raise BacktestValidationError("sync_inline persisted run requires sizing_mode")
    return response.sizing_mode


def _sorted_execution_payload(*, response: RunBacktestResponse) -> dict[str, Any]:
    """
    Require and normalize effective execution params from the completed sync response.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    Args:
        response: Completed sync response.
    Returns:
        dict[str, Any]: Deterministic execution mapping sorted by key.
    Assumptions:
        Effective execution params are resolved before compute and returned by the sync use-case.
    Raises:
        BacktestValidationError: If execution metadata is unexpectedly absent.
    Side Effects:
        None.
    """
    if response.execution_params is None:
        raise BacktestValidationError("sync_inline persisted run requires execution_params")
    return {key: response.execution_params[key] for key in sorted(response.execution_params.keys())}


def _utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp for persisted sync-inline lifecycle writes.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        None.
    Returns:
        datetime: Current UTC-aware timestamp.
    Assumptions:
        Persisted sync-inline rows use UTC timestamps consistently with the jobs storage family.
    Raises:
        None.
    Side Effects:
        None.
    """
    return datetime.now(timezone.utc)


__all__ = [
    "BacktestRunsApiUseCase",
    "CreateAndRunBacktestSyncInlineUseCase",
]
